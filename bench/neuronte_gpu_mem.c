/* neuronte_gpu_mem: device memory of weight files held open together.
 *
 *   neuronte_gpu_mem --cell WxH [--backend cuda|vulkan|opengl] [--device NAME] [--max-batch N] [--draw] FILE.nrt ...
 *
 * Opens the files one after another on one device and keeps them open. After
 * each, prints what the handle allocated and, on NVIDIA devices, the device
 * memory the driver attributes to this process. --draw makes one call per
 * handle first (a space), so buffers that grow with use are counted. One
 * JSON object per file, then a total.
 */
#define _GNU_SOURCE
#include "neuronte_gpu.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct nvml_proc {
    unsigned pid;
    unsigned long long used;
    unsigned gpu_instance, compute_instance;
};

static int (*nv_handle)(unsigned, void **);
static int (*nv_procs)(void *, unsigned *, struct nvml_proc *);
static unsigned nv_count;

static void nvml_load(void)
{
    void *dl = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    int (*init)(void), (*count)(unsigned *);
    if (!dl)
        return;
    *(void **)&init = dlsym(dl, "nvmlInit_v2");
    *(void **)&count = dlsym(dl, "nvmlDeviceGetCount_v2");
    *(void **)&nv_handle = dlsym(dl, "nvmlDeviceGetHandleByIndex_v2");
    *(void **)&nv_procs = dlsym(dl, "nvmlDeviceGetComputeRunningProcesses_v3");
    if (!init || !count || !nv_handle || !nv_procs || init() || count(&nv_count))
        nv_count = 0;
}

/* Device memory the driver attributes to this process, in bytes; 0 when it does not say. */
static unsigned long long process_bytes(void)
{
    unsigned long long sum = 0;
    for (unsigned d = 0; d < nv_count; d++) {
        void *dev;
        struct nvml_proc pr[64];
        unsigned n = 64;
        if (nv_handle(d, &dev) || nv_procs(dev, &n, pr))
            continue;
        for (unsigned i = 0; i < n && i < 64; i++)
            if (pr[i].pid == (unsigned)getpid() && pr[i].used != ~0ull)
                sum += pr[i].used;
    }
    return sum;
}

int main(int argc, char **argv)
{
    struct neuronte_gpu_options opt = {0};
    const char *backend = "cuda";
    neuronte_gpu *held[256];
    int n = 0, i, draw = 0;
    size_t buffers = 0, weights = 0, staging = 0;

    opt.max_batch = 256;
    for (i = 1; i < argc && !strncmp(argv[i], "--", 2); i++) {
        if (!strcmp(argv[i], "--cell") && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &opt.cell_w, &opt.cell_h) != 2)
                return 2;
        } else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) opt.device = argv[++i];
        else if (!strcmp(argv[i], "--max-batch") && i + 1 < argc) opt.max_batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--draw")) draw = 1;
        else return 2;
    }
    if (opt.cell_w < 1 || i >= argc)
        return 2;
    opt.backend = !strcmp(backend, "vulkan") ? NEURONTE_GPU_VULKAN
                  : !strcmp(backend, "opengl") ? NEURONTE_GPU_OPENGL : NEURONTE_GPU_CUDA;
    opt.flags = NEURONTE_GPU_FILE;
    nvml_load();
    printf("{\"before\":{\"process_device_bytes\":%llu}}\n", process_bytes());
    for (; i < argc && n < 256; i++) {
        struct neuronte_gpu_memory m = {0};
        char reason[512] = "";
        const char *name = strrchr(argv[i], '/');
        opt.nrt_path = argv[i];
        if (neuronte_gpu_open(&opt, &held[n], reason, sizeof reason)) {
            printf("{\"file\":\"%s\",\"error\":\"%s\"}\n", name ? name + 1 : argv[i], reason);
            continue;
        }
        if (draw) {
            uint32_t sp = 0x20;
            float *out = malloc((size_t)opt.cell_w * opt.cell_h * sizeof *out);
            if (!out || neuronte_gpu_coverage(held[n], &sp, 1, out, NULL))
                return 1;
            free(out);
        }
        neuronte_gpu_memory(held[n], &m);
        buffers += m.buffers;
        weights += m.weights;
        staging += m.staging;
        printf("{\"file\":\"%s\",\"backend\":\"%s\",\"device\":\"%s\",\"nets\":%d,\"widest\":%d,\"buffers\":%zu,"
               "\"weights\":%zu,\"staging\":%zu,\"driver_measured\":%zu,\"shared_context\":%d,"
               "\"process_device_bytes\":%llu}\n",
               name ? name + 1 : argv[i], neuronte_gpu_backend_name(neuronte_gpu_backend(held[n])),
               neuronte_gpu_device(held[n]), neuronte_gpu_tiles(held[n]), neuronte_gpu_hidden(held[n]), m.buffers,
               m.weights, m.staging, m.measured, m.shared_context, process_bytes());
        fflush(stdout);
        n++;
    }
    printf("{\"total\":{\"files\":%d,\"cell\":[%d,%d],\"max_batch\":%d,\"buffers\":%zu,\"weights\":%zu,"
           "\"staging\":%zu,\"process_device_bytes\":%llu}}\n",
           n, opt.cell_w, opt.cell_h, opt.max_batch, buffers, weights, staging, process_bytes());
    for (i = 0; i < n; i++)
        neuronte_gpu_close(held[i]);
    return 0;
}
