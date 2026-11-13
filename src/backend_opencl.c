/* neuronte-gpu OpenCL backend, through the OpenCL ICD loader (libOpenCL.so.1) opened at run
 * time. Written for Arm Mali GPUs, whose vendor drivers offer OpenCL where they may offer no
 * Vulkan (Mali-T860: OpenCL 1.2 only; Mali-G610: OpenCL 3.0 and Vulkan), and usable with any
 * OpenCL 1.2 GPU. The kernel (kernels/cover.cl, embedded as source) is compiled by the driver at
 * open for the model set's hidden width and sample count. A device that is a CPU (such as PoCL)
 * is refused unless the caller allows it, and is then reported as a CPU device. */
#define _GNU_SOURCE
#include "internal.h"
#include "clmini.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { B_PARAMS, B_TILES, B_MASKS, B_SCALARS, B_OUT, NBUF };
/* kernel arguments, in the order of cover.cl */
enum { A_PARAMS, A_TILES, A_MASKS, A_SCALARS, A_OUT, A_NGLYPH, A_CELL_W, A_PLANE, A_GPB, A_PER_SAMPLE, A_CS, A_SV };

#define DEFAULT_BLOCK 64
/* The tile kernel from this hidden width up. Measured at 16x32 (Braille models): on the Mali-T860
   (Midgard, 16 vec4 registers per thread at most) the tile kernel won from H = 12 (1.3x to 2.7x);
   on the Mali-G610 (Valhall) the per-sample kernel won up to H = 24 and the tile kernel at H = 48
   (2.6x). */
#define TILE_MIN_H_MIDGARD 12
#define TILE_MIN_H 32
#define TILE_NS 64    /* samples per pass of the tile kernel, at most */

struct impl {
    struct clmini cl;
    neuronte_gpu *g;
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
    cl_kernel k;
    cl_mem bufs[NBUF];
    int host_divide; /* S*S is not a power of two: the kernel writes sums and the host divides */
    int ns;          /* the tile kernel's samples per pass; 0 = the per-sample kernel */
};

static void impl_close(void *p)
{
    struct impl *m = p;
    struct clmini *cl = &m->cl;
    int i;
    if (!m)
        return;
    if (cl->clReleaseMemObject)
        for (i = 0; i < NBUF; i++)
            if (m->bufs[i])
                cl->clReleaseMemObject(m->bufs[i]);
    if (m->k)
        cl->clReleaseKernel(m->k);
    if (m->prog)
        cl->clReleaseProgram(m->prog);
    if (m->q)
        cl->clReleaseCommandQueue(m->q);
    if (m->ctx)
        cl->clReleaseContext(m->ctx);
    if (cl->handle)
        dlclose(cl->handle);
    free(m);
}

static int impl_run(void *p, const uint32_t *scalars, int count, float *out, struct neuronte_gpu_timing *t)
{
    struct impl *m = p;
    struct clmini *cl = &m->cl;
    neuronte_gpu *g = m->g;
    size_t plane = (size_t)g->cell_w * g->cell_h, n = (size_t)count * plane, i;
    int groups = (count + g->gpb - 1) / g->gpb, nz = neuronte_gpu_splits(g, g->ntiles * groups);
    cl_int nglyph = count, per_sample = nz > 1;
    size_t global[3] = {(size_t)g->ntiles * (size_t)g->block, (size_t)groups, (size_t)nz};
    size_t local[3] = {(size_t)g->block, 1, 1};
    cl_event ev = NULL;
    cl_ulong start = 0, end = 0;
    uint64_t t0, t1, t2;
    int rc = -1;

    if (count <= 0)
        return 0;
    t0 = neuronte_gpu_now();
    if (cl->clEnqueueWriteBuffer(m->q, m->bufs[B_SCALARS], CL_FALSE, 0, (size_t)count * sizeof *scalars, scalars, 0,
                                 NULL, NULL) ||
        cl->clSetKernelArg(m->k, A_NGLYPH, sizeof nglyph, &nglyph) ||
        cl->clSetKernelArg(m->k, A_PER_SAMPLE, sizeof per_sample, &per_sample) ||
        cl->clEnqueueNDRangeKernel(m->q, m->k, 3, NULL, global, local, 0, NULL, &ev) || cl->clFlush(m->q))
        goto done;
    t1 = neuronte_gpu_now();
    if (cl->clWaitForEvents(1, &ev))
        goto done;
    t2 = neuronte_gpu_now();
    if (cl->clEnqueueReadBuffer(m->q, m->bufs[B_OUT], CL_TRUE, 0, n * sizeof *out, out, 0, NULL, NULL))
        goto done;
    if (m->host_divide) {
        const float d = (float)(g->samples * g->samples);
        for (i = 0; i < n; i++)
            out[i] = out[i] / d;
    }
    if (!cl->clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof start, &start, NULL) &&
        !cl->clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof end, &end, NULL) && end >= start)
        t->run += end - start;
    t->upload += t1 - t0;
    t->wait += t2 - t1;
    t->download += neuronte_gpu_now() - t2;
    t->launches++;
    rc = 0;
done:
    if (ev)
        cl->clReleaseEvent(ev);
    return rc;
}

static const struct backend_ops ops = {impl_close, impl_run};

/* The parameters as cover.cl reads them, from the core's packing (which holds the layer-1 tables
 * and the slopes exactly as the library computes them): per tile, the weight block with layers 2
 * and 3 transposed to [input][output], then the x and y tables, every row padded to NQ float4s.
 * recs receives the tile records with offsets in float4s. */
static float *pack_cl(const neuronte_gpu *g, size_t *n4_out, int32_t *recs)
{
    const int h = g->hidden, nq = (h + 3) / 4, h4 = 4 * nq, s = g->samples;
    const size_t block4 = (size_t)(CODE + 1) * nq + 2 * (size_t)h * nq + 3 * (size_t)nq + 1;
    size_t n4 = 0, off4 = 0;
    float *p;
    int t, i, o, q;
    for (t = 0; t < g->ntiles; t++)
        n4 += block4 + (size_t)(g->tiles[t].w + g->tiles[t].h) * s * nq;
    p = calloc(n4 * 4, sizeof *p);
    if (!p)
        return NULL;
    for (t = 0; t < g->ntiles; t++) {
        const struct gtile *tl = &g->tiles[t];
        const struct net *n = &tl->net;
        const int32_t *src = g->pk.tiles + t * T_INTS;
        int32_t *rec = recs + t * T_INTS;
        float *b = p + 4 * off4;
        const float *xt = g->pk.params + src[T_XTAB], *yt = g->pk.params + src[T_YTAB];
        const int sw = tl->w * s, sh = tl->h * s;
        memcpy(rec, src, T_INTS * sizeof *rec);
        rec[T_PARAM] = (int32_t)off4;
        for (i = 0; i < CODE; i++) /* code columns, [bit][output] */
            for (o = 0; o < h; o++)
                b[(size_t)i * h4 + o] = n->w1[(size_t)o * IN + POS + i];
        b += (size_t)CODE * h4;
        memcpy(b, n->b1, (size_t)h * sizeof *b);
        b += h4;
        for (q = 0; q < h; q++) /* w2: the file's [output][input], transposed */
            for (o = 0; o < h; o++)
                b[(size_t)q * h4 + o] = n->w2[(size_t)o * h + q];
        b += (size_t)h * h4;
        memcpy(b, n->b2, (size_t)h * sizeof *b);
        b += h4;
        for (q = 0; q < h; q++)
            for (o = 0; o < h; o++)
                b[(size_t)q * h4 + o] = n->w3[(size_t)o * h + q];
        b += (size_t)h * h4;
        memcpy(b, n->b3, (size_t)h * sizeof *b);
        b += h4;
        memcpy(b, n->w4, (size_t)h * sizeof *b);
        b += h4;
        b[0] = n->b4;
        b[1] = g->pk.params[src[T_PARAM] + (CODE + 1) * h + 2 * h * h + 3 * h + 1]; /* the slope */
        off4 += block4;
        rec[T_XTAB] = (int32_t)off4;
        for (i = 0; i < sw; i++)
            memcpy(p + 4 * (off4 + (size_t)i * nq), xt + (size_t)i * h, (size_t)h * sizeof *p);
        off4 += (size_t)sw * nq;
        rec[T_YTAB] = (int32_t)off4;
        for (i = 0; i < sh; i++)
            memcpy(p + 4 * (off4 + (size_t)i * nq), yt + (size_t)i * h, (size_t)h * sizeof *p);
        off4 += (size_t)sh * nq;
    }
    *n4_out = n4;
    return p;
}

static void dev_str(struct clmini *cl, cl_device_id d, cl_device_info what, char *buf, size_t len)
{
    buf[0] = 0;
    if (cl->clGetDeviceInfo(d, what, len - 1, buf, NULL))
        buf[0] = 0;
    buf[len - 1] = 0;
}

static int type_rank(cl_device_type type, int allow_cpu)
{
    if (type & CL_DEVICE_TYPE_GPU)
        return 3;
    if (type & CL_DEVICE_TYPE_CPU)
        return allow_cpu ? 1 : -1;
    return 2; /* accelerator or custom */
}

int neuronte_gpu_opencl_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len)
{
    struct impl *m = calloc(1, sizeof *m);
    struct clmini *cl;
    cl_platform_id plats[16], best_plat = NULL;
    cl_uint nplat = 0, p, i;
    cl_device_type best_type = 0;
    int best_rank = -1, refused = 0, allow_cpu = (opt->flags & NEURONTE_GPU_ALLOW_CPU_DEVICE) != 0;
    cl_int err = 0, cvals[3];
    char name[256], pver[256] = "", dver[256] = "", drv[256] = "", cver[256] = "", opts[512], plat_name[256] = "";
    cl_uint cu = 0, mhz = 0, unified = 0, lmem_type = 0;
    cl_ulong lmem = 0;
    cl_device_fp_config fp = 0;
    size_t kmax = 0, kmult = 0, dev_max = 0, n4 = 0, plane = (size_t)g->cell_w * g->cell_h;
    cl_ulong kpriv = 0, klocal = 0;
    int32_t recs[MAX_TILES * T_INTS];
    float *params = NULL;
    const char *src = neuronte_gpu_opencl_source;
    char *src_file = NULL;
    const int ss = g->samples * g->samples, nq = (g->hidden + 3) / 4;
    int l3b, gpb_est, pmin = 1 << 30, t;
    const char *extra, *kname;
    cl_mem_flags ro = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

    if (!m)
        return -1;
    m->g = g;
    cl = &m->cl;
    /* NEURONTE_GPU_OPENCL_SOURCE names a kernel source file to use instead of the embedded one
       (a development aid) */
    if (getenv("NEURONTE_GPU_OPENCL_SOURCE")) {
        FILE *f = fopen(getenv("NEURONTE_GPU_OPENCL_SOURCE"), "rb");
        long n = -1;
        if (f && fseek(f, 0, SEEK_END) == 0 && (n = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0 &&
            (src_file = calloc((size_t)n + 1, 1)) && fread(src_file, 1, (size_t)n, f) == (size_t)n)
            src = src_file;
        else
            n = -1;
        if (f)
            fclose(f);
        if (n < 0) {
            neuronte_gpu_say(why, len, "cannot read NEURONTE_GPU_OPENCL_SOURCE");
            goto fail;
        }
    }
    if (!src[0]) {
        neuronte_gpu_say(why, len, "no OpenCL kernel built");
        goto fail;
    }
    if (g->hidden > 64) {
        neuronte_gpu_say(why, len, "the OpenCL kernel supports hidden widths up to 64, not %d", g->hidden);
        goto fail;
    }
    if (clmini_load(cl, "libOpenCL.so.1", why, len))
        goto fail;
    if (cl->clGetPlatformIDs(16, plats, &nplat) || nplat == 0) {
        neuronte_gpu_say(why, len, "no OpenCL platform");
        goto fail;
    }
    if (nplat > 16)
        nplat = 16;
    /* the first GPU in platform order (or the first device whose name contains opt->device) */
    for (p = 0; p < nplat; p++) {
        cl_device_id devs[16];
        cl_uint ndev = 0;
        if (cl->clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 16, devs, &ndev))
            continue;
        if (ndev > 16)
            ndev = 16;
        for (i = 0; i < ndev; i++) {
            cl_device_type type = 0;
            int rank;
            dev_str(cl, devs[i], CL_DEVICE_NAME, name, sizeof name);
            if (cl->clGetDeviceInfo(devs[i], CL_DEVICE_TYPE, sizeof type, &type, NULL))
                continue;
            if (opt->device && !strstr(name, opt->device))
                continue;
            rank = type_rank(type, allow_cpu);
            if (rank < 0)
                refused++;
            if (rank > best_rank) {
                best_rank = rank;
                m->dev = devs[i];
                best_plat = plats[p];
                best_type = type;
            }
        }
    }
    if (best_rank < 0) {
        neuronte_gpu_say(why, len, "no OpenCL GPU%s%s%s", opt->device ? " named like " : "",
                         opt->device ? opt->device : "",
                         refused ? " (CPU devices are refused unless allowed)" : "");
        goto fail;
    }
    dev_str(cl, m->dev, CL_DEVICE_NAME, g->device, sizeof g->device);
    dev_str(cl, m->dev, CL_DEVICE_VERSION, dver, sizeof dver);
    dev_str(cl, m->dev, CL_DRIVER_VERSION, drv, sizeof drv);
    dev_str(cl, m->dev, CL_DEVICE_OPENCL_C_VERSION, cver, sizeof cver);
    if (cl->clGetPlatformInfo(best_plat, CL_PLATFORM_VERSION, sizeof pver - 1, pver, NULL))
        pver[0] = 0;
    if (cl->clGetPlatformInfo(best_plat, CL_PLATFORM_NAME, sizeof plat_name - 1, plat_name, NULL))
        plat_name[0] = 0;
    cl->clGetDeviceInfo(m->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
    cl->clGetDeviceInfo(m->dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof mhz, &mhz, NULL);
    cl->clGetDeviceInfo(m->dev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof unified, &unified, NULL);
    cl->clGetDeviceInfo(m->dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof lmem, &lmem, NULL);
    cl->clGetDeviceInfo(m->dev, CL_DEVICE_LOCAL_MEM_TYPE, sizeof lmem_type, &lmem_type, NULL);
    cl->clGetDeviceInfo(m->dev, CL_DEVICE_SINGLE_FP_CONFIG, sizeof fp, &fp, NULL);
    cl->clGetDeviceInfo(m->dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof dev_max, &dev_max, NULL);
    g->device_type = (best_type & CL_DEVICE_TYPE_CPU) && !(best_type & CL_DEVICE_TYPE_GPU) ? "cpu"
                     : !(best_type & CL_DEVICE_TYPE_GPU)                                 ? "accelerator"
                     : unified                                                         ? "integrated-gpu"
                                                                                       : "gpu";

    m->ctx = cl->clCreateContext(NULL, 1, &m->dev, NULL, NULL, &err);
    if (!m->ctx || err) {
        neuronte_gpu_say(why, len, "clCreateContext failed (%d)", (int)err);
        goto fail;
    }
    m->q = cl->clCreateCommandQueue(m->ctx, m->dev, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!m->q || err) {
        neuronte_gpu_say(why, len, "clCreateCommandQueue failed (%d)", (int)err);
        goto fail;
    }
    /* exact: sums of S*S samples are divided by S*S; that is a multiply by an exact reciprocal only
       when S*S is a power of two, and OpenCL's division need not be correctly rounded */
    m->host_divide = g->exact && (ss & (ss - 1)) != 0;
    /* layer-3 output blocks per pass of the per-sample kernel: one (more chains spilled registers
       on both Mali GPUs measured) */
    l3b = 1;
    /* The kernel: the tile kernel (layer by layer over NS samples, as matrix products) from
       TILE_MIN_H up, the per-sample kernel below it or when no pass of whole pixels fits in local
       memory. NS is the largest power of two up to TILE_NS that is a whole number of pixels and
       leaves room for the character parts of the glyphs a work-group is likely to take (with the
       default work-group size). NEURONTE_GPU_OPENCL_KERNEL=tile|sample overrides the choice (a
       tuning aid). */
    for (t = 0; t < g->ntiles; t++)
        if (g->tiles[t].w * g->tiles[t].h < pmin)
            pmin = g->tiles[t].w * g->tiles[t].h;
    gpb_est = (2 * DEFAULT_BLOCK + pmin - 1) / pmin;
    gpb_est = gpb_est < 1 ? 1 : gpb_est > 64 ? 64 : gpb_est;
    for (m->ns = TILE_NS; m->ns >= 4; m->ns /= 2)
        if (m->ns % ss == 0 && 2 * (cl_ulong)(4 * nq) * m->ns * 4 + (cl_ulong)(m->ns > DEFAULT_BLOCK ? m->ns : DEFAULT_BLOCK) * 4 +
                                       (cl_ulong)gpb_est * nq * 16 + 1024 <= lmem)
            break;
    if (m->ns < 4 || g->hidden < (strncmp(g->device, "Mali-T", 6) ? TILE_MIN_H : TILE_MIN_H_MIDGARD))
        m->ns = 0;
    kname = getenv("NEURONTE_GPU_OPENCL_KERNEL");
    if (kname && !strcmp(kname, "sample"))
        m->ns = 0;
    else if (kname && !strcmp(kname, "tile") && !m->ns && g->hidden >= 1)
        for (m->ns = TILE_NS; m->ns >= 4 && m->ns % ss; m->ns /= 2)
            ;
    if (m->ns < 4)
        m->ns = 0;
    /* NEURONTE_GPU_OPENCL_DEFINES adds compiler options (a tuning aid, e.g. "-DL3B=2"); later
       definitions override earlier ones */
    extra = getenv("NEURONTE_GPU_OPENCL_DEFINES");
    snprintf(opts, sizeof opts, "-cl-std=CL1.2 -DH=%d -DS=%d -DEXACT=%d -DHOST_DIVIDE=%d -DL3B=%d -DNS=%d%s%s%s",
             g->hidden, g->samples, g->exact, m->host_divide, l3b, m->ns ? m->ns : TILE_NS,
             g->exact ? "" : " -cl-mad-enable", extra ? " " : "", extra ? extra : "");
    m->prog = cl->clCreateProgramWithSource(m->ctx, 1, &src, NULL, &err);
    free(src_file);
    src_file = NULL;
    if (!m->prog || err) {
        neuronte_gpu_say(why, len, "clCreateProgramWithSource failed (%d)", (int)err);
        goto fail;
    }
    err = cl->clBuildProgram(m->prog, 1, &m->dev, opts, NULL, NULL);
    if (err) {
        char log[400] = "";
        cl->clGetProgramBuildInfo(m->prog, m->dev, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, NULL);
        log[sizeof log - 1] = 0;
        neuronte_gpu_say(why, len, "the OpenCL kernel did not build (%d): %s", (int)err, log);
        goto fail;
    }
    m->k = cl->clCreateKernel(m->prog, m->ns ? "cover_tile" : "cover", &err);
    if (!m->k || err) {
        neuronte_gpu_say(why, len, "clCreateKernel failed (%d)", (int)err);
        goto fail;
    }
    cl->clGetKernelWorkGroupInfo(m->k, m->dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof kmax, &kmax, NULL);
    cl->clGetKernelWorkGroupInfo(m->k, m->dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof kmult, &kmult,
                                 NULL);
    cl->clGetKernelWorkGroupInfo(m->k, m->dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof kpriv, &kpriv, NULL);
    cl->clGetKernelWorkGroupInfo(m->k, m->dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof klocal, &klocal, NULL);
    if (kmax == 0)
        kmax = dev_max ? dev_max : DEFAULT_BLOCK;

    /* the work-group size: asked for, else the default, within what the compiled kernel allows */
    neuronte_gpu_set_block(g, opt, DEFAULT_BLOCK < (int)kmax ? DEFAULT_BLOCK : (int)kmax);
    if ((size_t)g->block > kmax) {
        neuronte_gpu_say(why, len, "a work-group of %d is more than this kernel allows on %s (%zu)", g->block,
                         g->device, kmax);
        goto fail;
    }
    if ((cl_ulong)g->gpb * nq * 16 + (cl_ulong)(g->block > m->ns ? g->block : m->ns) * 4 + klocal > lmem) {
        neuronte_gpu_say(why, len, "%d glyphs per work-group need more local memory than %s has", g->gpb, g->device);
        goto fail;
    }
    g->fill_blocks = 8 * (int)(cu ? cu : 4);

    params = pack_cl(g, &n4, recs);
    if (!params) {
        neuronte_gpu_say(why, len, "out of memory");
        goto fail;
    }
    m->bufs[B_PARAMS] = cl->clCreateBuffer(m->ctx, ro, n4 * 4 * sizeof(float), params, &err);
    m->bufs[B_TILES] = cl->clCreateBuffer(m->ctx, ro, (size_t)g->ntiles * T_INTS * sizeof(int32_t), recs, &err);
    m->bufs[B_MASKS] = cl->clCreateBuffer(m->ctx, ro, (size_t)g->ntiles * NMASK * sizeof(uint32_t), g->pk.masks, &err);
    m->bufs[B_SCALARS] =
        cl->clCreateBuffer(m->ctx, CL_MEM_READ_ONLY, (size_t)g->max_batch * sizeof(uint32_t), NULL, &err);
    m->bufs[B_OUT] =
        cl->clCreateBuffer(m->ctx, CL_MEM_WRITE_ONLY, (size_t)g->max_batch * plane * sizeof(float), NULL, &err);
    free(params);
    params = NULL;
    for (i = 0; i < NBUF; i++)
        if (!m->bufs[i]) {
            neuronte_gpu_say(why, len, "clCreateBuffer failed (%d)", (int)err);
            goto fail;
        }
    cvals[0] = g->cell_w;
    cvals[1] = (cl_int)plane;
    cvals[2] = g->gpb;
    if (cl->clSetKernelArg(m->k, A_PARAMS, sizeof(cl_mem), &m->bufs[B_PARAMS]) ||
        cl->clSetKernelArg(m->k, A_TILES, sizeof(cl_mem), &m->bufs[B_TILES]) ||
        cl->clSetKernelArg(m->k, A_MASKS, sizeof(cl_mem), &m->bufs[B_MASKS]) ||
        cl->clSetKernelArg(m->k, A_SCALARS, sizeof(cl_mem), &m->bufs[B_SCALARS]) ||
        cl->clSetKernelArg(m->k, A_OUT, sizeof(cl_mem), &m->bufs[B_OUT]) ||
        cl->clSetKernelArg(m->k, A_CELL_W, sizeof(cl_int), &cvals[0]) ||
        cl->clSetKernelArg(m->k, A_PLANE, sizeof(cl_int), &cvals[1]) ||
        cl->clSetKernelArg(m->k, A_GPB, sizeof(cl_int), &cvals[2]) ||
        cl->clSetKernelArg(m->k, A_CS, (size_t)g->gpb * nq * 16, NULL) ||
        cl->clSetKernelArg(m->k, A_SV, (size_t)(g->block > m->ns ? g->block : m->ns) * sizeof(float), NULL)) {
        neuronte_gpu_say(why, len, "clSetKernelArg failed");
        goto fail;
    }
    neuronte_gpu_say(g->versions, sizeof g->versions,
                     "opencl %s (%s); device %s; driver %s; %s; %u compute units, max clock %u MHz; fp32 denormals %s, "
                     "correctly rounded divide/sqrt %s; local memory %llu KB (%s); kernel: %llu private bytes, "
                     "max work-group %zu, preferred multiple %zu; %s kernel; work-group %d, %d glyphs per group; "
                     "build %s%s",
                     pver, plat_name, dver, drv, cver, cu, mhz, fp & CL_FP_DENORM ? "kept" : "flushed",
                     fp & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT ? "yes" : "no", (unsigned long long)(lmem / 1024),
                     lmem_type == CL_LOCAL ? "dedicated" : "global memory", (unsigned long long)kpriv, kmax, kmult,
                     m->ns ? "tile" : "per-sample", g->block, g->gpb, opts, m->host_divide ? "; divides on the host" : "");
    g->ops = &ops;
    g->impl = m;
    return 0;
fail:
    free(params);
    free(src_file);
    impl_close(m);
    return -1;
}
