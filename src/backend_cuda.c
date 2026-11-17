/* neuronte-gpu CUDA backend, through the CUDA driver library loaded at run
 * time. No CUDA toolkit is needed to build or run: the kernels are embedded
 * as PTX, which the driver compiles for the GPU it finds (and caches). */
#define _GNU_SOURCE
#include "internal.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The few driver API types and constants used here (values from cuda.h). */
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef unsigned long long CUdeviceptr;

#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN 97
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES 8
#define CU_FUNC_ATTRIBUTE_NUM_REGS 4
#define CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES 3
#define CU_JIT_WALL_TIME 2
#define CU_JIT_INFO_LOG_BUFFER 3
#define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES 4
#define CU_JIT_ERROR_LOG_BUFFER 5
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6
#define CU_STREAM_NON_BLOCKING 1
#define CU_EVENT_DEFAULT 0

struct cu {
    void *dl;
    CUresult (*Init)(unsigned);
    CUresult (*DriverGetVersion)(int *);
    CUresult (*DeviceGetCount)(int *);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    CUresult (*DeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*CtxCreate)(CUcontext *, unsigned, CUdevice);
    CUresult (*CtxDestroy)(CUcontext);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*ModuleLoadDataEx)(CUmodule *, const void *, unsigned, int *, void **);
    CUresult (*ModuleUnload)(CUmodule);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*FuncGetAttribute)(int *, int, CUfunction);
    CUresult (*FuncSetAttribute)(CUfunction, int, int);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemAllocHost)(void **, size_t);
    CUresult (*MemFreeHost)(void *);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*MemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*MemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
    CUresult (*StreamCreate)(CUstream *, unsigned);
    CUresult (*StreamDestroy)(CUstream);
    CUresult (*StreamSynchronize)(CUstream);
    CUresult (*EventCreate)(CUevent *, unsigned);
    CUresult (*EventDestroy)(CUevent);
    CUresult (*EventRecord)(CUevent, CUstream);
    CUresult (*EventElapsedTime)(float *, CUevent, CUevent);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                             CUstream, void **, void **);
    CUresult (*GetErrorString)(CUresult, const char **);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*CtxSynchronize)(void);
};

struct impl {
    struct cu cu;
    neuronte_gpu *g;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUstream stream;
    CUevent ev0, ev1;
    CUdeviceptr d_params, d_tiles, d_masks, d_scalars, d_out;
    uint32_t *h_scalars; /* pinned */
    float *h_out;        /* pinned */
    unsigned shared_bytes;
    /* file handles */
    CUmodule *mods; /* one per kernel of the plan */
    CUfunction *fns, reduce;
    CUdeviceptr d_segs, d_blocks, d_pairs, d_samp, d_list;
    size_t blocks_cap, pairs_cap; /* records the device buffers hold */
    unsigned char *h_up;          /* pinned: a call's tables on their way to the device */
    size_t up_cap;
};

static const char *err(struct cu *cu, CUresult rc)
{
    const char *s = NULL;
    if (cu->GetErrorString && cu->GetErrorString(rc, &s) == 0 && s)
        return s;
    return "unknown CUDA error";
}

static int load_api(struct cu *cu, char *why, size_t len)
{
    static const char *names[] = {"libcuda.so.1", "libcuda.so"};
    size_t i;
    for (i = 0; i < sizeof names / sizeof *names && !cu->dl; i++)
        cu->dl = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!cu->dl) {
        neuronte_gpu_say(why, len, "no CUDA driver library (libcuda.so.1)");
        return -1;
    }
#define SYM(field, name)                                                       \
    do {                                                                       \
        *(void **)&cu->field = dlsym(cu->dl, name);                            \
        if (!cu->field) {                                                      \
            neuronte_gpu_say(why, len, "CUDA driver lacks %s", name);          \
            return -1;                                                         \
        }                                                                      \
    } while (0)
    SYM(Init, "cuInit");
    SYM(DriverGetVersion, "cuDriverGetVersion");
    SYM(DeviceGetCount, "cuDeviceGetCount");
    SYM(DeviceGet, "cuDeviceGet");
    SYM(DeviceGetName, "cuDeviceGetName");
    SYM(DeviceGetAttribute, "cuDeviceGetAttribute");
    SYM(CtxCreate, "cuCtxCreate_v2");
    SYM(CtxDestroy, "cuCtxDestroy_v2");
    SYM(CtxSetCurrent, "cuCtxSetCurrent");
    SYM(ModuleLoadDataEx, "cuModuleLoadDataEx");
    SYM(ModuleUnload, "cuModuleUnload");
    SYM(ModuleGetFunction, "cuModuleGetFunction");
    SYM(FuncGetAttribute, "cuFuncGetAttribute");
    SYM(FuncSetAttribute, "cuFuncSetAttribute");
    SYM(MemAlloc, "cuMemAlloc_v2");
    SYM(MemFree, "cuMemFree_v2");
    SYM(MemAllocHost, "cuMemAllocHost_v2");
    SYM(MemFreeHost, "cuMemFreeHost");
    SYM(MemcpyHtoD, "cuMemcpyHtoD_v2");
    SYM(MemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    SYM(MemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    SYM(StreamCreate, "cuStreamCreate");
    SYM(StreamDestroy, "cuStreamDestroy_v2");
    SYM(StreamSynchronize, "cuStreamSynchronize");
    SYM(EventCreate, "cuEventCreate");
    SYM(EventDestroy, "cuEventDestroy_v2");
    SYM(EventRecord, "cuEventRecord");
    SYM(EventElapsedTime, "cuEventElapsedTime");
    SYM(LaunchKernel, "cuLaunchKernel");
    SYM(GetErrorString, "cuGetErrorString");
    SYM(MemGetInfo, "cuMemGetInfo_v2");
    SYM(CtxSynchronize, "cuCtxSynchronize");
#undef SYM
    return 0;
}

/* One context per device for the whole process: every handle on a device shares it, so a second
 * handle costs its own buffers and kernels, not a second context. */
static struct {
    int device; /* ordinal */
    CUcontext ctx;
    int refs;
} shared_ctx[8];
static pthread_mutex_t shared_lock = PTHREAD_MUTEX_INITIALIZER;

static CUresult ctx_acquire(struct cu *cu, int ordinal, CUdevice dev, CUcontext *out, int *created)
{
    CUresult rc = 0;
    int i, spare = -1;
    pthread_mutex_lock(&shared_lock);
    for (i = 0; i < 8; i++) {
        if (shared_ctx[i].refs && shared_ctx[i].device == ordinal)
            break;
        if (!shared_ctx[i].refs && spare < 0)
            spare = i;
    }
    *created = 0;
    if (i == 8) {
        i = spare;
        if (i < 0) {
            rc = 2; /* CUDA_ERROR_OUT_OF_MEMORY */
        } else if ((rc = cu->CtxCreate(&shared_ctx[i].ctx, 0, dev)) == 0) {
            shared_ctx[i].device = ordinal;
            *created = 1;
        }
    } else {
        rc = cu->CtxSetCurrent(shared_ctx[i].ctx);
    }
    if (!rc) {
        shared_ctx[i].refs++;
        *out = shared_ctx[i].ctx;
    }
    pthread_mutex_unlock(&shared_lock);
    return rc;
}

static void ctx_release(struct cu *cu, CUcontext ctx)
{
    int i;
    pthread_mutex_lock(&shared_lock);
    for (i = 0; i < 8; i++)
        if (shared_ctx[i].refs && shared_ctx[i].ctx == ctx && --shared_ctx[i].refs == 0)
            cu->CtxDestroy(ctx);
    pthread_mutex_unlock(&shared_lock);
}

static void impl_close(void *p)
{
    struct impl *m = p;
    struct cu *cu;
    if (!m)
        return;
    cu = &m->cu;
    if (m->ctx) {
        cu->CtxSetCurrent(m->ctx);
        if (m->h_scalars)
            cu->MemFreeHost(m->h_scalars);
        if (m->h_out)
            cu->MemFreeHost(m->h_out);
        if (m->d_params)
            cu->MemFree(m->d_params);
        if (m->d_tiles)
            cu->MemFree(m->d_tiles);
        if (m->d_masks)
            cu->MemFree(m->d_masks);
        if (m->d_scalars)
            cu->MemFree(m->d_scalars);
        if (m->d_out)
            cu->MemFree(m->d_out);
        if (m->d_segs)
            cu->MemFree(m->d_segs);
        if (m->d_blocks)
            cu->MemFree(m->d_blocks);
        if (m->d_pairs)
            cu->MemFree(m->d_pairs);
        if (m->d_samp)
            cu->MemFree(m->d_samp);
        if (m->d_list)
            cu->MemFree(m->d_list);
        if (m->h_up)
            cu->MemFreeHost(m->h_up);
        for (int i = 0; m->mods && i < m->g->plan.nkern; i++)
            if (m->mods[i])
                cu->ModuleUnload(m->mods[i]);
        if (m->ev0)
            cu->EventDestroy(m->ev0);
        if (m->ev1)
            cu->EventDestroy(m->ev1);
        if (m->stream)
            cu->StreamDestroy(m->stream);
        if (m->mod)
            cu->ModuleUnload(m->mod);
        ctx_release(cu, m->ctx);
    }
    if (cu->dl)
        dlclose(cu->dl);
    free(m->mods);
    free(m->fns);
    free(m);
}

static int impl_run(void *p, const uint32_t *scalars, int count, float *out, struct neuronte_gpu_timing *t)
{
    struct impl *m = p;
    struct cu *cu = &m->cu;
    neuronte_gpu *g = m->g;
    size_t plane = (size_t)g->cell_w * g->cell_h, bytes = (size_t)count * plane * sizeof(float);
    unsigned groups = (unsigned)((count + g->gpb - 1) / g->gpb);
    unsigned nz = (unsigned)neuronte_gpu_splits(g, g->ntiles * (int)groups);
    int gpb = g->gpb, cw = g->cell_w, ch = g->cell_h, s = g->samples, per_sample = nz > 1;
    void *args[] = {&m->d_params, &m->d_tiles, &m->d_masks, &m->d_scalars, &m->d_out, &count, &gpb, &cw, &ch, &s,
                    &per_sample};
    uint64_t t0, t1, t2;
    float ms = 0;
    CUresult rc;

    if (count <= 0)
        return 0;
    t0 = neuronte_gpu_now();
    if (cu->CtxSetCurrent(m->ctx))
        return -1;
    memcpy(m->h_scalars, scalars, (size_t)count * sizeof *scalars);
    rc = cu->MemcpyHtoDAsync(m->d_scalars, m->h_scalars, (size_t)count * sizeof *scalars, m->stream);
    t1 = neuronte_gpu_now();
    if (!rc)
        rc = cu->EventRecord(m->ev0, m->stream);
    if (!rc)
        rc = cu->LaunchKernel(m->fn, (unsigned)g->ntiles, groups, nz, (unsigned)g->block, 1, 1, m->shared_bytes,
                              m->stream, args, NULL);
    if (!rc)
        rc = cu->EventRecord(m->ev1, m->stream);
    if (!rc)
        rc = cu->MemcpyDtoHAsync(m->h_out, m->d_out, bytes, m->stream);
    if (!rc)
        rc = cu->StreamSynchronize(m->stream);
    t2 = neuronte_gpu_now();
    if (rc) {
        fprintf(stderr, "neuronte-gpu: CUDA: %s\n", err(cu, rc));
        return -1;
    }
    memcpy(out, m->h_out, bytes);
    cu->EventElapsedTime(&ms, m->ev0, m->ev1);
    t->upload += t1 - t0;
    t->wait += t2 - t1;
    t->run += (uint64_t)((double)ms * 1e6);
    t->download += neuronte_gpu_now() - t2;
    t->launches++;
    return 0;
}

static const struct backend_ops ops = {impl_close, impl_run, NULL};

int neuronte_gpu_cuda_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len)
{
    struct impl *m = calloc(1, sizeof *m);
    struct cu *cu;
    const struct embedded_ptx *ptx = NULL;
    int i, count = 0, ver = 0, pick = -1, major = 0, minor = 0, sms = 0, optin = 0, regs = 0, local = 0;
    int fresh = 0;
    char name[256] = "", info[4096] = "", error[4096] = "";
    CUdevice dev = 0;
    CUresult rc;
    size_t plane = (size_t)g->cell_w * g->cell_h;

    if (!m)
        return -1;
    neuronte_gpu_set_block(g, opt, 128);
    m->g = g;
    cu = &m->cu;
    for (i = 0; i < neuronte_gpu_ptx_count; i++)
        if (neuronte_gpu_ptx[i].h == g->hidden)
            ptx = &neuronte_gpu_ptx[i];
    if (!ptx) {
        neuronte_gpu_say(why, len, "no CUDA kernel built for hidden width %d", g->hidden);
        goto fail;
    }
    if (load_api(cu, why, len))
        goto fail;
    if ((rc = cu->Init(0)) != 0) {
        neuronte_gpu_say(why, len, "cuInit: %s", err(cu, rc));
        goto fail;
    }
    cu->DriverGetVersion(&ver);
    if (cu->DeviceGetCount(&count) || count < 1) {
        neuronte_gpu_say(why, len, "no CUDA device");
        goto fail;
    }
    for (i = 0; i < count && pick < 0; i++) {
        if (cu->DeviceGet(&dev, i) || cu->DeviceGetName(name, sizeof name, dev))
            continue;
        if (!opt->device || strcasestr(name, opt->device))
            pick = i;
    }
    if (pick < 0) {
        neuronte_gpu_say(why, len, "no CUDA device matches \"%s\"", opt->device ? opt->device : "");
        goto fail;
    }
    cu->DeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cu->DeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cu->DeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    g->fill_blocks = 2 * (sms > 0 ? sms : 1);
    cu->DeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
    if ((rc = ctx_acquire(cu, pick, dev, &m->ctx, &fresh)) != 0) {
        neuronte_gpu_say(why, len, "cuCtxCreate: %s", err(cu, rc));
        goto fail;
    }
    {
        /* cuModuleLoadDataEx needs NUL-terminated PTX */
        char *text = malloc(ptx->len + 1);
        int keys[] = {CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER,
                      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
        void *vals[] = {info, (void *)(size_t)sizeof info, error, (void *)(size_t)sizeof error};
        if (!text)
            goto fail;
        memcpy(text, ptx->data, ptx->len);
        text[ptx->len] = 0;
        rc = cu->ModuleLoadDataEx(&m->mod, text, 4, keys, vals);
        free(text);
        if (rc) {
            neuronte_gpu_say(why, len, "loading the PTX: %s: %.300s", err(cu, rc), error);
            goto fail;
        }
    }
    if ((rc = cu->ModuleGetFunction(&m->fn, m->mod, g->exact ? "neuronte_cover_exact" : "neuronte_cover_fast"))) {
        neuronte_gpu_say(why, len, "kernel not found: %s", err(cu, rc));
        goto fail;
    }
    m->shared_bytes = (unsigned)g->pk.shared_floats * sizeof(float);
    if (m->shared_bytes > 48 * 1024) {
        if ((int)m->shared_bytes > optin ||
            cu->FuncSetAttribute(m->fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)m->shared_bytes)) {
            neuronte_gpu_say(why, len, "needs %u bytes of shared memory per block, device allows %d",
                             m->shared_bytes, optin);
            goto fail;
        }
    }
    cu->FuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, m->fn);
    cu->FuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, m->fn);
    if ((rc = cu->StreamCreate(&m->stream, CU_STREAM_NON_BLOCKING)) || (rc = cu->EventCreate(&m->ev0, 0)) ||
        (rc = cu->EventCreate(&m->ev1, 0)) ||
        (rc = cu->MemAlloc(&m->d_params, g->pk.nparams * sizeof(float))) ||
        (rc = cu->MemAlloc(&m->d_tiles, sizeof g->pk.tiles)) || (rc = cu->MemAlloc(&m->d_masks, sizeof g->pk.masks)) ||
        (rc = cu->MemAlloc(&m->d_scalars, (size_t)g->max_batch * sizeof(uint32_t))) ||
        (rc = cu->MemAlloc(&m->d_out, (size_t)g->max_batch * plane * sizeof(float))) ||
        (rc = cu->MemAllocHost((void **)&m->h_scalars, (size_t)g->max_batch * sizeof(uint32_t))) ||
        (rc = cu->MemAllocHost((void **)&m->h_out, (size_t)g->max_batch * plane * sizeof(float))) ||
        (rc = cu->MemcpyHtoD(m->d_params, g->pk.params, g->pk.nparams * sizeof(float))) ||
        (rc = cu->MemcpyHtoD(m->d_tiles, g->pk.tiles, sizeof g->pk.tiles)) ||
        (rc = cu->MemcpyHtoD(m->d_masks, g->pk.masks, sizeof g->pk.masks)) ||
        (rc = cu->CtxSynchronize())) { /* the copies have landed before any launch on the stream */
        neuronte_gpu_say(why, len, "allocating or uploading: %s", err(cu, rc));
        goto fail;
    }
    snprintf(g->device, sizeof g->device, "%s", name);
    snprintf(g->versions, sizeof g->versions,
             "cuda driver api %d.%d; compute capability %d.%d; %d SMs; %d registers, %d local bytes per thread; "
             "%u shared bytes per block; ptx for H=%d",
             ver / 1000, ver % 1000 / 10, major, minor, sms, regs, local, m->shared_bytes, ptx->h);
    g->device_type = "gpu";
    g->ops = &ops;
    g->impl = m;
    return 0;
fail:
    impl_close(m);
    return -1;
}

/* ---- file handles ---- */

static CUresult load_module(struct cu *cu, const struct embedded_ptx *ptx, CUmodule *mod, char *error, size_t elen)
{
    char *text = malloc(ptx->len + 1);
    int keys[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *vals[] = {error, (void *)elen};
    CUresult rc;
    if (!text)
        return 2; /* CUDA_ERROR_OUT_OF_MEMORY */
    memcpy(text, ptx->data, ptx->len);
    text[ptx->len] = 0;
    rc = cu->ModuleLoadDataEx(mod, text, 2, keys, vals);
    free(text);
    return rc;
}

static CUresult dev_alloc(struct cu *cu, CUdeviceptr *d, size_t n, size_t *sum)
{
    *sum += n;
    return cu->MemAlloc(d, n ? n : 4);
}

/* A device buffer of at least `need` records of `size` bytes, grown when a call needs more. */
static CUresult fit(struct impl *m, CUdeviceptr *d, size_t *cap, size_t need, size_t size)
{
    CUresult rc;
    size_t want;
    if (need <= *cap)
        return 0;
    want = *cap ? *cap : 1024;
    while (want < need)
        want *= 2;
    if (*d) {
        m->cu.MemFree(*d);
        m->g->device_bytes -= *cap * size;
        *d = 0;
        *cap = 0;
    }
    rc = m->cu.MemAlloc(d, want * size);
    if (!rc) {
        *cap = want;
        m->g->device_bytes += want * size;
    }
    return rc;
}

static int impl_run_plan(void *p, int count, float *out, struct neuronte_gpu_timing *t)
{
    struct impl *m = p;
    struct cu *cu = &m->cu;
    neuronte_gpu *g = m->g;
    struct v2plan *pl = &g->plan;
    size_t plane = (size_t)g->cell_w * g->cell_h, bytes = (size_t)count * plane * sizeof(float);
    int cw = g->cell_w, ch = g->cell_h, s = g->samples, k, block0;
    void *args[] = {&m->d_params, &m->d_segs, &m->d_blocks, &m->d_pairs, &m->d_masks, &m->d_out, &m->d_samp,
                    &block0,      &cw,        &ch,          &s};
    uint64_t t0, t1, t2;
    float ms = 0;
    CUresult rc;

    t0 = neuronte_gpu_now();
    if (cu->CtxSetCurrent(m->ctx))
        return -1;
    rc = fit(m, &m->d_pairs, &m->pairs_cap, (size_t)pl->npairs, 2 * sizeof(uint32_t));
    if (!rc)
        rc = fit(m, &m->d_blocks, &m->blocks_cap, (size_t)pl->nblocks, B_INTS * sizeof(int32_t));
    if (!rc) {
        /* The call's tables go through the stream, from pinned memory, so the kernels queued behind
         * them see them: a plain copy is not ordered against a launch on this stream, and a small
         * one may still be on its way when it returns. */
        struct {
            CUdeviceptr dst;
            const void *src;
            size_t bytes;
        } up[5] = {
            {m->d_pairs, pl->pairs, (size_t)pl->npairs * 2 * sizeof(uint32_t)},
            {m->d_blocks, pl->blocks, (size_t)pl->nblocks * B_INTS * sizeof(int32_t)},
            {m->d_list, pl->reduce, (size_t)pl->nreduce * sizeof(uint32_t)},
            {m->d_segs + (size_t)pl->nstatic_segs * S_INTS * sizeof(int32_t),
             pl->segs + (size_t)pl->nstatic_segs * S_INTS,
             pl->dynamic ? (size_t)(pl->nsegs - pl->nstatic_segs) * S_INTS * sizeof(int32_t) : 0},
            {m->d_params + pl->nstatic * sizeof(float), pl->params + pl->nstatic,
             pl->dynamic ? (pl->nparams - pl->nstatic) * sizeof(float) : 0},
        };
        size_t total = 0, at = 0;
        int u;
        for (u = 0; u < 5; u++)
            total += up[u].bytes;
        if (total > m->up_cap) {
            size_t want = m->up_cap ? m->up_cap : 65536;
            while (want < total)
                want *= 2;
            if (m->h_up)
                cu->MemFreeHost(m->h_up);
            m->h_up = NULL;
            g->staging_bytes -= m->up_cap;
            m->up_cap = 0;
            rc = cu->MemAllocHost((void **)&m->h_up, want);
            if (!rc) {
                m->up_cap = want;
                g->staging_bytes += want;
            }
        }
        for (u = 0; u < 5 && !rc; u++) {
            if (!up[u].bytes)
                continue;
            memcpy(m->h_up + at, up[u].src, up[u].bytes);
            rc = cu->MemcpyHtoDAsync(up[u].dst, m->h_up + at, up[u].bytes, m->stream);
            at += up[u].bytes;
        }
    }
    t1 = neuronte_gpu_now();
    if (!rc)
        rc = cu->EventRecord(m->ev0, m->stream);
    for (k = 0; k < pl->nkern && !rc; k++) {
        const struct v2kern *kn = &pl->kern[k];
        if (!kn->nblocks)
            continue;
        block0 = kn->block0;
        rc = cu->LaunchKernel(m->fns[k], (unsigned)kn->nblocks, 1, 1, (unsigned)kn->threads, 1, 1,
                              (unsigned)kn->shared_floats * sizeof(float), m->stream, args, NULL);
        t->launches++;
    }
    if (!rc && pl->nreduce) {
        int n = pl->nreduce;
        void *rargs[] = {&m->d_samp, &m->d_out, &m->d_list, &n, &cw, &ch, &s};
        size_t threads = (size_t)n * plane;
        rc = cu->LaunchKernel(m->reduce, (unsigned)((threads + 255) / 256), 1, 1, 256, 1, 1, 0, m->stream, rargs,
                              NULL);
        t->launches++;
    }
    if (!rc)
        rc = cu->EventRecord(m->ev1, m->stream);
    if (!rc)
        rc = cu->MemcpyDtoHAsync(m->h_out, m->d_out, bytes, m->stream);
    if (!rc)
        rc = cu->StreamSynchronize(m->stream);
    t2 = neuronte_gpu_now();
    if (rc) {
        fprintf(stderr, "neuronte-gpu: CUDA: %s\n", err(cu, rc));
        return -1;
    }
    memcpy(out, m->h_out, bytes);
    cu->EventElapsedTime(&ms, m->ev0, m->ev1);
    t->upload += t1 - t0;
    t->wait += t2 - t1;
    t->run += (uint64_t)((double)ms * 1e6);
    t->download += neuronte_gpu_now() - t2;
    return 0;
}

static const struct backend_ops ops2 = {impl_close, NULL, impl_run_plan};

int neuronte_gpu_cuda_open2(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len)
{
    struct impl *m = calloc(1, sizeof *m);
    struct cu *cu;
    struct v2plan *pl = &g->plan;
    int i, k, count = 0, ver = 0, pick = -1, major = 0, minor = 0, sms = 0, optin = 0;
    int fresh = 0;
    char name[256] = "", error[4096] = "", kinds[256] = "";
    CUdevice dev = 0;
    CUresult rc;
    size_t plane = (size_t)g->cell_w * g->cell_h, splane = plane * g->samples * g->samples, free0 = 0, free1 = 0,
           total = 0, bytes = 0;
    const char *tune;

    if (!m)
        return -1;
    m->g = g;
    cu = &m->cu;
    {
        /* narrow blocks of 128 threads; wide blocks of 4 samples in flight, a warp each */
        int nt = opt->block > 0 ? opt->block : 128, wt = opt->block > 0 ? (opt->block + LANES - 1) / LANES : 4;
        if ((tune = getenv("NEURONTE_GPU_WIDE_SLOTS")) && atoi(tune) > 0 && atoi(tune) <= 32)
            wt = atoi(tune);
        if (neuronte_gpu_v2_size(g, nt, wt * LANES, wt, 1, 12288 /* 48 KB */, why, len))
            goto fail;
    }
    for (k = 0; k < pl->nkern; k++) {
        const struct v2kern *kn = &pl->kern[k];
        int want = kn->wide ? 1000 + kn->n : kn->n, found = 0;
        for (i = 0; i < neuronte_gpu_ptx2_count; i++)
            found |= neuronte_gpu_ptx2[i].h == want;
        if (!found) {
            if (kn->wide)
                neuronte_gpu_say(why, len, "no CUDA kernel built for hidden widths up to %d", kn->n * LANES);
            else
                neuronte_gpu_say(why, len, "no CUDA kernel built for hidden width %d", kn->n);
            goto fail;
        }
    }
    if (load_api(cu, why, len))
        goto fail;
    if ((rc = cu->Init(0)) != 0) {
        neuronte_gpu_say(why, len, "cuInit: %s", err(cu, rc));
        goto fail;
    }
    cu->DriverGetVersion(&ver);
    if (cu->DeviceGetCount(&count) || count < 1) {
        neuronte_gpu_say(why, len, "no CUDA device");
        goto fail;
    }
    for (i = 0; i < count && pick < 0; i++) {
        if (cu->DeviceGet(&dev, i) || cu->DeviceGetName(name, sizeof name, dev))
            continue;
        if (!opt->device || strcasestr(name, opt->device))
            pick = i;
    }
    if (pick < 0) {
        neuronte_gpu_say(why, len, "no CUDA device matches \"%s\"", opt->device ? opt->device : "");
        goto fail;
    }
    cu->DeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cu->DeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cu->DeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cu->DeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
    if (sms < 1)
        sms = 1;
    /* The narrow kernel's blocks hold their weights in shared memory, so few fit a multiprocessor;
     * the wide kernel's hold little, so a multiprocessor takes as many as its threads allow. */
    g->fill_blocks = 2 * sms;
    g->fill_wide = 8 * sms;
    if ((tune = getenv("NEURONTE_GPU_FILL")) && atoi(tune) > 0)
        g->fill_blocks = atoi(tune);
    if ((tune = getenv("NEURONTE_GPU_FILL_WIDE")) && atoi(tune) > 0)
        g->fill_wide = atoi(tune);
    if ((rc = ctx_acquire(cu, pick, dev, &m->ctx, &fresh)) != 0) {
        neuronte_gpu_say(why, len, "cuCtxCreate: %s", err(cu, rc));
        goto fail;
    }
    cu->MemGetInfo(&free0, &total);
    g->context_fresh = fresh;
    m->mods = calloc((size_t)pl->nkern + 1, sizeof *m->mods);
    m->fns = calloc((size_t)pl->nkern + 1, sizeof *m->fns);
    if (!m->mods || !m->fns)
        goto fail;
    for (k = 0; k < pl->nkern; k++) {
        const struct v2kern *kn = &pl->kern[k];
        int want = kn->wide ? 1000 + kn->n : kn->n;
        const struct embedded_ptx *ptx = NULL;
        size_t used = strlen(kinds);
        for (i = 0; i < neuronte_gpu_ptx2_count; i++)
            if (neuronte_gpu_ptx2[i].h == want)
                ptx = &neuronte_gpu_ptx2[i];
        if ((rc = load_module(cu, ptx, &m->mods[k], error, sizeof error)) != 0) {
            neuronte_gpu_say(why, len, "loading the PTX: %s: %.300s", err(cu, rc), error);
            goto fail;
        }
        if ((rc = cu->ModuleGetFunction(&m->fns[k], m->mods[k],
                                        g->exact ? "neuronte_cover2_exact" : "neuronte_cover2_fast")) ||
            (!m->reduce && (rc = cu->ModuleGetFunction(&m->reduce, m->mods[k], "neuronte_reduce")))) {
            neuronte_gpu_say(why, len, "kernel not found: %s", err(cu, rc));
            goto fail;
        }
        if ((size_t)kn->shared_floats * sizeof(float) > 48 * 1024 &&
            ((int)((size_t)kn->shared_floats * sizeof(float)) > optin ||
             cu->FuncSetAttribute(m->fns[k], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  (int)((size_t)kn->shared_floats * sizeof(float))))) {
            neuronte_gpu_say(why, len, "needs %zu bytes of shared memory per block, device allows %d",
                             (size_t)kn->shared_floats * sizeof(float), optin);
            goto fail;
        }
        snprintf(kinds + used, sizeof kinds - used, "%s%s%d", used ? " " : "", kn->wide ? "wide" : "H", kn->n);
    }
#define ALLOC(ptr, n) dev_alloc(cu, &(ptr), (n), &bytes)
    if ((rc = cu->StreamCreate(&m->stream, CU_STREAM_NON_BLOCKING)) || (rc = cu->EventCreate(&m->ev0, 0)) ||
        (rc = cu->EventCreate(&m->ev1, 0)) || (rc = ALLOC(m->d_params, pl->params_cap * sizeof(float))) ||
        (rc = ALLOC(m->d_segs, (size_t)pl->segs_cap * S_INTS * sizeof(int32_t))) ||
        (rc = ALLOC(m->d_masks, (size_t)pl->nmasks * sizeof(uint32_t))) ||
        (rc = ALLOC(m->d_list, (size_t)g->max_batch * sizeof(uint32_t))) ||
        (rc = ALLOC(m->d_out, (size_t)g->max_batch * plane * sizeof(float))) ||
        (pl->need_samples && (rc = ALLOC(m->d_samp, (size_t)g->max_batch * splane * sizeof(float)))) ||
        (rc = cu->MemAllocHost((void **)&m->h_out, (size_t)g->max_batch * plane * sizeof(float))) ||
        (rc = cu->MemcpyHtoD(m->d_params, pl->params, pl->nstatic * sizeof(float))) ||
        (pl->nstatic_segs &&
         (rc = cu->MemcpyHtoD(m->d_segs, pl->segs, (size_t)pl->nstatic_segs * S_INTS * sizeof(int32_t)))) ||
        (pl->nmasks && (rc = cu->MemcpyHtoD(m->d_masks, pl->masks, (size_t)pl->nmasks * sizeof(uint32_t)))) ||
        (rc = cu->CtxSynchronize())) { /* the copies have landed before any launch on the stream */
        neuronte_gpu_say(why, len, "allocating or uploading: %s", err(cu, rc));
        goto fail;
    }
#undef ALLOC
    g->device_bytes = bytes;
    g->staging_bytes = (size_t)g->max_batch * plane * sizeof(float);
    cu->MemGetInfo(&free1, &total);
    g->measured_bytes = free0 > free1 ? free0 - free1 : 0;
    snprintf(g->device, sizeof g->device, "%s", name);
    snprintf(g->versions, sizeof g->versions,
             "cuda driver api %d.%d; compute capability %d.%d; %d SMs; file handle: %d nets, %d segments, kernels %s",
             ver / 1000, ver % 1000 / 10, major, minor, sms, g->nmembers, pl->nstatic_segs, kinds);
    g->device_type = "gpu";
    g->ops = &ops2;
    g->impl = m;
    return 0;
fail:
    impl_close(m);
    return -1;
}
