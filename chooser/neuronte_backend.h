/* neuronte_backend: draw coverage with whichever of the CPU library and a
 * GPU is fastest on this machine, as Neuronte's setup measured it.
 *
 * A program that draws cells opens its weights file here in place of
 * neuronte_open_file() and asks for coverage in lists. The answer is byte
 * for byte what neuronte_coverage() gives, wherever it was computed: a GPU
 * backend that differed from the CPU by one byte when setup compared them
 * is never used. Without a GPU, without the GPU library, without setup's
 * backend.conf, on battery, or when anything about the GPU goes wrong, the
 * CPU library draws; no call fails for a reason the CPU library would not
 * fail for.
 *
 * The order for each scalar of a call: the cells file setup made for the
 * weights file, when there is one (pre-drawn, no computing at all); then,
 * for what it does not hold, the GPU when the call has min_glyphs or more
 * such scalars and a GPU is in use, else the CPU.
 *
 * A handle is for one thread at a time, like a neuronte handle. Nothing
 * here waits for a GPU: it is opened in the background at the first call
 * that wants it, the CPU draws until it is there, and a call that is under
 * way when it arrives hands it the glyphs that are left.
 *
 * Overrides: NEURONTE_BACKEND=cpu in the environment (this program), and
 * backend=cpu in setup.conf (this machine, or this user). There is no way
 * to force a GPU that setup did not choose.
 *
 * C11 and POSIX; links the reference library, libdl and libpthread, and
 * loads libneuronte_gpu only when backend.conf names it. */
#ifndef NEURONTE_BACKEND_H
#define NEURONTE_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct neuronte_backend neuronte_backend;

struct neuronte_backend_info {
    int gpu;          /* 1 when calls of min_glyphs or more go to a GPU now */
    int min_glyphs;   /* 0 when no GPU is in use */
    char backend[32]; /* "cpu", "gpu-cuda", "gpu-vulkan" */
    char device[128]; /* the GPU's name once it has been opened, else what setup measured; "" on the CPU */
    char reason[256]; /* why, in words: setup's reason, or "forced by NEURONTE_BACKEND",
                       * "no backend.conf", "warming", "on battery", "device lost", ... */
    /* glyphs answered so far by each (zero from neuronte_backend_choice) */
    uint64_t from_cells, from_cpu, from_gpu;
    int ready;        /* 1 when the GPU is open at this moment, so a call for it draws nothing on the CPU */
};

/* Open a weights file for cells of cell_w x cell_h pixels. Returns 0 and the
 * handle, or the reference library's error for a file it cannot open or a
 * size it does not take. Never fails because of a GPU. */
int neuronte_backend_open(const char *nrt_path, int cell_w, int cell_h, neuronte_backend **out);

/* Coverage of `count` scalars: out[count * cell_w * cell_h] floats, row-major
 * per glyph, each glyph what neuronte_coverage(n, scalar, cell_w, cell_h, 0, ...)
 * writes. Returns 0, or the reference library's error. */
int neuronte_backend_coverage(neuronte_backend *b, const uint32_t *scalars, int count, float *out);

/* For a caller that draws a job a few glyphs at a time so as not to wait
 * (a terminal's main thread): the calls that follow are parts of one job of
 * `count` glyphs, and are judged for the GPU as a call of that size would
 * be. Each call takes its glyphs off the job; 0 forgets it. Draws nothing
 * and returns at once. */
void neuronte_backend_expect(neuronte_backend *b, int count);

/* What the handle is doing now. */
void neuronte_backend_info(const neuronte_backend *b, struct neuronte_backend_info *info);

void neuronte_backend_close(neuronte_backend *b);

/* The choice alone, from backend.conf, the overrides and the power supply,
 * opening nothing and touching no GPU. Returns info->gpu. */
int neuronte_backend_choice(struct neuronte_backend_info *info);

#ifdef __cplusplus
}
#endif

#endif
