/* neuronte-gpu: evaluate Neuronte distance networks on a GPU.
 *
 * A handle covers one model set (one network, or a set of spatial tiles that
 * each draw a rectangle of the cell) at one cell size. The model set is
 * described by a layout file (tools/layout.py writes one from a kilix-bench
 * experiment manifest) or by a single .nrt file.
 *
 * Backends: CUDA (NVIDIA, through the driver library loaded at run time),
 * Vulkan compute (any GPU with a Vulkan driver) and OpenCL (any GPU with an
 * OpenCL 1.2 driver; written for Arm Mali). When none can be used,
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
    NEURONTE_GPU_AUTO = 0,         /* request: CUDA, then Vulkan, then OpenCL; where Arm's kernel
                                    * driver is present (/dev/mali0), OpenCL before Vulkan */
    NEURONTE_GPU_CUDA = 1,         /* inference on an NVIDIA GPU through CUDA */
    NEURONTE_GPU_VULKAN = 2,       /* inference through Vulkan compute */
    NEURONTE_GPU_CPU_FALLBACK = 3, /* no usable GPU: the reference CPU library */
    NEURONTE_GPU_CPU = 4,          /* the CPU path was requested explicitly */
    NEURONTE_GPU_OPENCL = 5,       /* inference through OpenCL */
    NEURONTE_GPU_OPENGL = 6,       /* inference in an OpenGL 4.3 compute shader; only when asked for by
                                    * name, and always as a file handle (see below) */
};

enum neuronte_gpu_flags {
    NEURONTE_GPU_ALLOW_FALLBACK = 1,   /* fall back to the CPU instead of failing open */
    NEURONTE_GPU_FAST = 2,             /* allow fused multiply-adds (not bit-exact) */
    NEURONTE_GPU_ALLOW_CPU_DEVICE = 4, /* accept a Vulkan or OpenCL device that is a CPU (e.g. lavapipe, PoCL);
                                        * it is reported as such, never as a GPU */
    NEURONTE_GPU_SHARED_WEIGHTS = 8,   /* Vulkan: copy layers 2-4 to shared memory instead of
                                        * reading them from the parameter buffer (CUDA always
                                        * uses shared memory) */
    NEURONTE_GPU_FILE = 16,            /* open nrt_path as a file handle even when it is one narrow
                                        * net (any other file always is one, see below) */
    NEURONTE_GPU_GL_CURRENT = 32,      /* OpenGL: use the context that is current on the calling thread,
                                        * at open and at every call and at close, instead of making one.
                                        * For a program that draws with OpenGL itself. The handle changes
                                        * the shader storage buffer bindings 0 to 7 and the buffer bound to
                                        * GL_SHADER_STORAGE_BUFFER, and restores the program in use. */
};

struct neuronte_gpu_options {
    const char *layout_path; /* a layout file, or NULL with nrt_path */
    const char *model_dir;   /* directory of the .nrt files a layout names; NULL = the layout's own */
    const char *nrt_path;    /* one network over the whole cell, when layout_path is NULL */
    int cell_w, cell_h;      /* with nrt_path */
    int samples;             /* per axis; 0 = the library's choice for the cell width */
    int backend;             /* enum neuronte_gpu_backend: AUTO, CUDA, VULKAN, OPENCL or CPU */
    const char *device;      /* substring of the device name to select; NULL = the first GPU */
    int flags;
    int max_batch;           /* glyphs per launch; 0 = 1024 */
    int block;               /* threads per block / workgroup; 0 = backend default */
};

/* Per-call stage times in nanoseconds. GPU stages are zero on the CPU path. */
struct neuronte_gpu_timing {
    uint64_t upload;   /* scalars to the device */
    uint64_t run;      /* the kernel, as the device measured it (CUDA events, Vulkan timestamps,
                        * OpenCL profiling) */
    uint64_t wait;     /* submit to completion, as the host saw it */
    uint64_t download; /* coverage back to the caller's buffer */
    uint64_t cpu_eval; /* reference CPU evaluation: the CPU paths, or glyphs the GPU path cannot draw */
    uint64_t total;
    int launches;
    int cpu_glyphs;    /* glyphs evaluated by the CPU library on a GPU handle */
    /* file handles */
    uint64_t plan;     /* routing the scalars and laying out the launch, on the host */
    int marks;         /* untrained marks, drawn by the library (coverage 0 or 1, no net) */
    int warped;        /* letters drawn at positions warped by the letter edge table */
};

/* The number of this binary interface: the options structure, the enums and
 * the calls below. A program that loads the library at run time compares it
 * with the one it was built for and does not use a library that differs. */
#define NEURONTE_GPU_ABI 1
int neuronte_gpu_abi(void);

int neuronte_gpu_open(const struct neuronte_gpu_options *opt, neuronte_gpu **out, char *reason,
                      size_t reason_len);
void neuronte_gpu_close(neuronte_gpu *g);

int neuronte_gpu_backend(const neuronte_gpu *g);
const char *neuronte_gpu_backend_name(int backend);
const char *neuronte_gpu_fallback_reason(const neuronte_gpu *g);
/* Device name, and the driver/API versions, of the active backend; "" on the CPU. */
const char *neuronte_gpu_device(const neuronte_gpu *g);
const char *neuronte_gpu_versions(const neuronte_gpu *g);
/* "gpu", "integrated-gpu", "accelerator" (OpenCL), "cpu" (a Vulkan or OpenCL CPU device) or "" */
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

/* ---- file handles ----
 *
 * nrt_path may name any file the reference library opens that holds a net: a
 * bundle (its router sends each scalar to one of its nets), a tiled net, a
 * drawing file (with or without a bundle, advances and a letter edge table),
 * or one distance net of any hidden width up to 1,024. One narrow net (hidden
 * width up to 64) keeps the single-net path above unless NEURONTE_GPU_FILE is
 * given. neuronte_gpu_coverage() then returns, for each scalar, what
 * neuronte_coverage() returns on that file at the handle's cell size and
 * samples. Colour nets and the untrained mark are drawn by the library on
 * the host and counted in the timing (cpu_glyphs, marks). CUDA and Vulkan
 * draw file handles; OpenCL does not yet.
 *
 * The calls below mirror the library's and return -1 on a handle that is
 * not a file handle. */
int neuronte_gpu_route(const neuronte_gpu *g, uint32_t scalar);      /* neuronte_route */
int neuronte_gpu_untrained(const neuronte_gpu *g, uint32_t scalar);  /* neuronte_untrained */
int neuronte_gpu_set_untrained(neuronte_gpu *g, int mode);           /* neuronte_set_untrained */
int neuronte_gpu_set_snap(neuronte_gpu *g, int below_px);            /* neuronte_set_snap */

int neuronte_gpu_set_binary(neuronte_gpu *g, int on);                /* neuronte_set_binary */

/* neuronte_draw_text() and neuronte_draw_text_stroke() for a file handle
 * opened on a drawing file at cell em x 2 em: the same pixels, with the
 * line's letters drawn by the GPU in one launch the first time each is
 * seen and kept by the handle afterwards. stroke_px -1 draws as
 * neuronte_draw_text() does with `bold`; 0 and up as
 * neuronte_draw_text_stroke(). Returns how far the pen moved, or one of the
 * library's error codes. `timing`, when not NULL, receives the launch's
 * stage times (zero when every letter was kept already). */
int neuronte_gpu_draw_text(neuronte_gpu *g, uint32_t *pixels, int stride, int width, int height, int x, int y,
                           int bold, int stroke_px, const char *text, size_t len, uint32_t color,
                           struct neuronte_gpu_timing *timing);

/* OpenGL file handles: draw without bringing the pixels back. The coverage
 * of the `count` glyphs (at most max_batch) is left in the handle's output
 * buffer, count * cell_w * cell_h floats, glyph after glyph, rows from the
 * top: a shader storage buffer whose name neuronte_gpu_gl_output returns,
 * for the caller's own shaders to read (for instance into a glyph atlas).
 * Glyphs the library draws on the host are copied into it. The buffer holds
 * a call's glyphs until the next call. Returns 0, or -1 (also on any other
 * backend). */
int neuronte_gpu_coverage_device(neuronte_gpu *g, const uint32_t *scalars, int count,
                                 struct neuronte_gpu_timing *timing);
unsigned neuronte_gpu_gl_output(const neuronte_gpu *g);

/* Device memory, in bytes. buffers: what this handle allocated on the device
 * (weights and tables, the batch's pairs and blocks, the output and, where
 * used, the sample buffer). weights: the part of buffers that is weights and
 * position tables. staging: host memory pinned or mapped for transfers.
 * measured: what the driver says the device lost between this handle
 * entering its context and the end of open (kernels and buffers, in the
 * driver's own granularity); 0 when the driver does not say. Handles on one
 * CUDA device share one context; shared_context is 1 for a handle that
 * joined a context another handle had made. What a context itself takes is
 * the process's device memory, as the driver's tools report it, less the
 * handles' own. */
struct neuronte_gpu_memory {
    size_t buffers, weights, staging, measured;
    int shared_context;
};
int neuronte_gpu_memory(const neuronte_gpu *g, struct neuronte_gpu_memory *m);

#ifdef __cplusplus
}
#endif

#endif
