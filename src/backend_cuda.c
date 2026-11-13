/* neuronte-gpu CUDA backend, through the CUDA driver library loaded at run
 * time. No CUDA toolkit is needed to build or run: the kernels are embedded
 * as PTX, which the driver compiles for the GPU it finds (and caches). */
#define _GNU_SOURCE
#include "internal.h"

#include <dlfcn.h>
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
#undef SYM
    return 0;
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
        if (m->ev0)
            cu->EventDestroy(m->ev0);
        if (m->ev1)
            cu->EventDestroy(m->ev1);
        if (m->stream)
            cu->StreamDestroy(m->stream);
        if (m->mod)
            cu->ModuleUnload(m->mod);
        cu->CtxDestroy(m->ctx);
    }
    if (cu->dl)
        dlclose(cu->dl);
    free(m);
}

static int impl_run(void *p, const uint32_t *scalars, int count, float *out, struct neuronte_gpu_timing *t)
{
    struct impl *m = p;
    struct cu *cu = &m->cu;
    neuronte_gpu *g = m->g;
    size_t plane = (size_t)g->cell_w * g->cell_h, bytes = (size_t)count * plane * sizeof(float);
    unsigned groups = (unsigned)((count + g->gpb - 1) / g->gpb);
    int gpb = g->gpb, cw = g->cell_w, ch = g->cell_h, s = g->samples;
    void *args[] = {&m->d_params, &m->d_tiles, &m->d_masks, &m->d_scalars, &m->d_out, &count, &gpb, &cw, &ch, &s};
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
        rc = cu->LaunchKernel(m->fn, (unsigned)g->ntiles, groups, 1, (unsigned)g->block, 1, 1, m->shared_bytes,
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

static const struct backend_ops ops = {impl_close, impl_run};

int neuronte_gpu_cuda_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len)
{
    struct impl *m = calloc(1, sizeof *m);
    struct cu *cu;
    const struct embedded_ptx *ptx = NULL;
    int i, count = 0, ver = 0, pick = -1, major = 0, minor = 0, sms = 0, optin = 0, regs = 0, local = 0;
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
    cu->DeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
    if ((rc = cu->CtxCreate(&m->ctx, 0, dev)) != 0) {
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
        (rc = cu->MemcpyHtoD(m->d_masks, g->pk.masks, sizeof g->pk.masks))) {
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
