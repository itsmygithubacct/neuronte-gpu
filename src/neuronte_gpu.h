/* neuronte-gpu: evaluate Neuronte distance networks on a GPU.
 *
 * A handle covers one model set (one network, or a set of spatial tiles that
 * each draw a rectangle of the cell) at one cell size. The model set is
 * described by a layout file (tools/layout.py writes one from a kilix-bench
 * experiment manifest) or by a single .nrt file.
 *
 * Backends: CUDA (NVIDIA, through the driver library loaded at run time) and
 * Vulkan compute (any GPU with a Vulkan driver). When neither can be used,
 * the handle can fall back to the reference CPU library with the same tiles.
 * The active backend and device are always reported; fallback results are
 * CPU results.
 *
 * In exact mode (the default) the GPU uses the CPU library's operation order
 * with separately rounded multiplies and adds, so its coverage matches the
 * CPU library's bit for bit wherever both round IEEE single precision the
 * same way. NEURONTE_GPU_FAST allows fused multiply-adds instead.
 */
#ifndef NEURONTE_GPU_H
#define NEURONTE_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct neuronte_gpu neuronte_gpu;

enum neuronte_gpu_backend {
    NEURONTE_GPU_AUTO = 0,         /* request: CUDA, then Vulkan */
    NEURONTE_GPU_CUDA = 1,         /* inference on an NVIDIA GPU through CUDA */
    NEURONTE_GPU_VULKAN = 2,       /* inference through Vulkan compute */
    NEURONTE_GPU_CPU_FALLBACK = 3, /* no usable GPU: the reference CPU library */
    NEURONTE_GPU_CPU = 4,          /* the CPU path was requested explicitly */
};

enum neuronte_gpu_flags {
    NEURONTE_GPU_ALLOW_FALLBACK = 1,   /* fall back to the CPU instead of failing open */
    NEURONTE_GPU_FAST = 2,             /* allow fused multiply-adds (not bit-exact) */
    NEURONTE_GPU_ALLOW_CPU_DEVICE = 4, /* accept a Vulkan device that is a CPU (e.g. lavapipe);
                                        * it is reported as such, never as a GPU */
    NEURONTE_GPU_SHARED_WEIGHTS = 8,   /* Vulkan: copy layers 2-4 to shared memory instead of
                                        * reading them from the parameter buffer (CUDA always
                                        * uses shared memory) */
};

struct neuronte_gpu_options {
    const char *layout_path; /* a layout file, or NULL with nrt_path */
    const char *model_dir;   /* directory of the .nrt files a layout names; NULL = the layout's own */
    const char *nrt_path;    /* one network over the whole cell, when layout_path is NULL */
    int cell_w, cell_h;      /* with nrt_path */
    int samples;             /* per axis; 0 = the library's choice for the cell width */
    int backend;             /* enum neuronte_gpu_backend: AUTO, CUDA, VULKAN or CPU */
    const char *device;      /* substring of the device name to select; NULL = the first GPU */
    int flags;
    int max_batch;           /* glyphs per launch; 0 = 1024 */
    int block;               /* threads per block / workgroup; 0 = backend default */
};

/* Per-call stage times in nanoseconds. GPU stages are zero on the CPU path. */
struct neuronte_gpu_timing {
    uint64_t upload;   /* scalars to the device */
    uint64_t run;      /* the kernel, as the device measured it (CUDA events, Vulkan timestamps) */
    uint64_t wait;     /* submit to completion, as the host saw it */
    uint64_t download; /* coverage back to the caller's buffer */
    uint64_t cpu_eval; /* reference CPU evaluation: the CPU paths, or glyphs the GPU path cannot draw */
    uint64_t total;
    int launches;
    int cpu_glyphs;    /* glyphs evaluated by the CPU library on a GPU handle */
};

int neuronte_gpu_open(const struct neuronte_gpu_options *opt, neuronte_gpu **out, char *reason,
                      size_t reason_len);
void neuronte_gpu_close(neuronte_gpu *g);

int neuronte_gpu_backend(const neuronte_gpu *g);
const char *neuronte_gpu_backend_name(int backend);
const char *neuronte_gpu_fallback_reason(const neuronte_gpu *g);
/* Device name, and the driver/API versions, of the active backend; "" on the CPU. */
const char *neuronte_gpu_device(const neuronte_gpu *g);
const char *neuronte_gpu_versions(const neuronte_gpu *g);
/* "gpu", "integrated-gpu", "cpu" (a Vulkan CPU device) or "" */
const char *neuronte_gpu_device_type(const neuronte_gpu *g);
int neuronte_gpu_cell(const neuronte_gpu *g, int *w, int *h);
int neuronte_gpu_samples(const neuronte_gpu *g);
int neuronte_gpu_hidden(const neuronte_gpu *g);
int neuronte_gpu_tiles(const neuronte_gpu *g);
int neuronte_gpu_exact(const neuronte_gpu *g);

/* Coverage for `count` scalars into out[count * cell_w * cell_h], row-major
 * per glyph: the values neuronte_coverage gives for each tile, placed at the
 * tile's position. */
int neuronte_gpu_coverage(neuronte_gpu *g, const uint32_t *scalars, int count, float *out,
                          struct neuronte_gpu_timing *timing);

#ifdef __cplusplus
}
#endif

#endif
