/* neuronte-gpu internals shared by the core and the backends. */
#ifndef NEURONTE_GPU_INTERNAL_H
#define NEURONTE_GPU_INTERNAL_H

#include "neuronte_gpu.h"
#include "neuronte.h"

#define OCT 6
#define POS (2 + 4 * OCT) /* 26 position features */
#define CODE 64           /* 21 scalar bits and 43 parity bits */
#define NMASK (CODE - 21)
#define IN (POS + CODE)
#define MAX_H 1024
#define MAX_TILES 64

/* Per-tile record in the tile table, ints. Must match the kernels. */
enum { T_X, T_Y, T_W, T_H, T_MASK, T_PARAM, T_XTAB, T_YTAB, T_MASKS, T_INTS };

/* One distance network, as stored in a version 1 .nrt file. */
struct net {
    int h;
    float tau;
    uint32_t masks[NMASK];
    float *w; /* every float of the file, in file order */
    const float *w1, *b1, *w2, *b2, *w3, *b3, *w4;
    float b4;
};

struct gtile {
    int mask; /* -1 passes the scalar through */
    int x, y, w, h;
    char file[1024];
    struct net net;
    neuronte *cpu;
};

/* Everything a kernel reads, packed for upload. Offsets are in elements. */
struct packed {
    float *params;
    size_t nparams;
    int32_t tiles[MAX_TILES * T_INTS];
    uint32_t masks[MAX_TILES * NMASK];
    int shared_floats; /* shared memory a block needs, in floats */
};

struct backend_ops {
    void (*close)(void *impl);
    /* count <= max_batch glyphs; writes count * plane floats to out */
    int (*run)(void *impl, const uint32_t *scalars, int count, float *out, struct neuronte_gpu_timing *t);
    /* a file handle: runs the plan in g->v2 (see below) and writes count * plane floats to out */
    int (*run_plan)(void *impl, int count, float *out, struct neuronte_gpu_timing *t);
};

/* ---- file handles: bundles, tiled nets, drawing files, any hidden width ----
 *
 * A file at one cell size is a table of segments. A segment is a rectangle of
 * the cell's sample grid drawn by one distance net: the net's weights, the
 * rectangle, its two layer-1 position tables and its coverage slope. A plain
 * net is one segment over the whole grid; a tiled net is one per tile. A
 * letter whose samples are warped (a drawing file's letter edge table) gets
 * segments of its own for the call, with tables made for its sample positions.
 *
 * A call turns its scalars into pairs (glyph, code), grouped by segment, and
 * into blocks: a block is a run of pairs of one segment and a slice of their
 * pixels or samples. Blocks are grouped by kernel (one per hidden width up
 * to 64, one per accumulator count above), one launch per kernel present.
 * Where tiles split a pixel, or the kernel is the wide one, samples go to a
 * sample buffer and a last pass adds each pixel's samples in the library's
 * order. Records below must match the kernels. */
enum { S_PARAM, S_MASKS, S_X0, S_Y0, S_W, S_H, S_XTAB, S_YTAB, S_SLOPE, S_HID, S_INTS };
enum { B_SEG, B_PAIR0, B_NPAIRS, B_LO, B_HI, B_MODE, B_INTS };
/* block modes */
enum {
    BM_PIXEL = 0,       /* items are pixels of an aligned segment; a thread takes a pixel's samples in turn */
    BM_PIXEL_SPLIT = 1, /* the same, a thread per sample and an ordered sum through shared memory */
    BM_SAMPLE = 2,      /* items are samples, written to the sample buffer */
    BM_DIRECT = 3,      /* items are samples of a 1-sample grid, written to the output */
};
#define NARROW_MAX 64 /* hidden widths with a kernel of their own */
#define LANES 32      /* threads that share one sample in the wide kernel */

struct v2kern {
    int wide; /* 0: one kernel for hidden width n; 1: the wide kernel with n accumulators per thread */
    int n;
    int shared_floats; /* per block */
    int threads;       /* per block */
    int slots;         /* wide: samples a block takes per pass */
    int hmax;          /* the widest net it draws */
    int gp_max;        /* the most pairs a block of this kernel takes */
    int block0, nblocks; /* this call's blocks, a run of the block table */
};

struct v2plan {
    /* static, made at open */
    float *params;
    size_t nparams, nstatic, params_cap; /* floats: used, the static part, capacity */
    int32_t *segs;
    int nsegs, nstatic_segs, segs_cap;
    uint32_t *masks;
    int nmasks; /* uints */
    struct v2kern *kern;
    int nkern;
    int need_samples; /* some glyph can need the sample buffer */
    /* per call */
    uint32_t *pairs; /* 2 per pair: glyph index in the batch, code */
    int npairs, pairs_cap;
    int32_t *blocks;
    int nblocks, blocks_cap;
    uint32_t *reduce; /* glyphs whose pixels come from the sample buffer */
    int nreduce;
    int dynamic; /* segments and params past the static part were made for this call */
};

struct v2;

struct neuronte_gpu {
    int backend;
    char reason[512];
    char device[256];
    char versions[1024];
    const char *device_type;
    char name[256];
    int cell_w, cell_h, samples, hidden, ntiles, exact, gpb, block, max_batch;
    int max_tile_px; /* pixels in the largest tile */
    int fill_blocks; /* blocks a launch needs to occupy the device; set by the backend */
    int has_span;
    uint32_t span_first, span_last;
    struct gtile tiles[MAX_TILES];
    struct packed pk;
    const struct backend_ops *ops;
    void *impl;
    float *cpu_buf;
    int nmembers;       /* a file handle's nets, as the library numbers them */
    struct v2 *v2;      /* a file handle (bundle, tiled, drawing file, wide net), else NULL */
    struct v2plan plan; /* what its backend reads */
    int fill_wide;      /* blocks of the wide kernel a launch needs to occupy the device */
    size_t device_bytes; /* bytes this handle allocated on the device */
    size_t context_bytes; /* device memory the context took before any allocation of ours, 0 if unknown */
    size_t staging_bytes; /* host memory pinned or mapped for transfers */
    int context_fresh;    /* this handle created the device context it uses (otherwise it joined one) */
    size_t measured_bytes; /* what the driver says the device lost between context creation and the end of open */
};

/* Backends. On failure they return nonzero, write a reason and leave
 * g->impl NULL; on success they set g->ops, g->impl, g->device, g->versions
 * and g->device_type. */
int neuronte_gpu_cuda_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len);
int neuronte_gpu_vulkan_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len);
int neuronte_gpu_opencl_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len);

int neuronte_gpu_cuda_open2(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len);
int neuronte_gpu_vulkan_open2(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len);
int neuronte_gpu_gl_open2(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len);

/* neuronte_gpu_v2.c */
int neuronte_gpu_v2_wanted(const char *path, int flags);
int neuronte_gpu_v2_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *reason, size_t reason_len);
void neuronte_gpu_v2_close(neuronte_gpu *g);
int neuronte_gpu_v2_size(neuronte_gpu *g, int narrow_threads, int wide_threads, int wide_slots, int cuda,
                         size_t shared_max, char *why, size_t wl);
int neuronte_gpu_v2_coverage(neuronte_gpu *g, const uint32_t *scalars, int count, float *out,
                             struct neuronte_gpu_timing *timing);
int neuronte_gpu_parse_net(const unsigned char *p, size_t len, struct net *n, char *why, size_t wl);
unsigned char *neuronte_gpu_read_file(const char *path, size_t *len);

/* Embedded kernels (generated at build time). */
struct embedded_ptx {
    int h;
    const unsigned char *data;
    unsigned len;
};
extern const struct embedded_ptx neuronte_gpu_ptx[];
extern const int neuronte_gpu_ptx_count;
/* File-handle kernels: h is the hidden width, or 1000 + accumulators per thread for the wide kernel. */
extern const struct embedded_ptx neuronte_gpu_ptx2[];
extern const int neuronte_gpu_ptx2_count;

struct embedded_spirv {
    int exact;
    const uint32_t *code;
    unsigned words;
};
extern const struct embedded_spirv neuronte_gpu_spirv[];
extern const int neuronte_gpu_spirv_count;

/* GLSL text of the file-handle shaders, NUL-terminated, without a #version line. */
extern const char neuronte_gpu_glsl_cover2[];
extern const char neuronte_gpu_glsl_reduce[];

/* OpenCL C source of kernels/cover.cl, NUL-terminated ("" when not embedded). */
extern const char neuronte_gpu_opencl_source[];

/* blocks along z for a launch of `blocks` blocks: more when the launch is too small to fill the device */
int neuronte_gpu_splits(const neuronte_gpu *g, int blocks);
/* sets block (opt->block, else dflt), glyphs per block and the shared memory a block needs */
void neuronte_gpu_set_block(neuronte_gpu *g, const struct neuronte_gpu_options *opt, int dflt);
void neuronte_gpu_say(char *buf, size_t len, const char *fmt, ...);
uint64_t neuronte_gpu_now(void);

#endif
