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
};

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
};

/* Backends. On failure they return nonzero, write a reason and leave
 * g->impl NULL; on success they set g->ops, g->impl, g->device, g->versions
 * and g->device_type. */
int neuronte_gpu_cuda_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len);
int neuronte_gpu_vulkan_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len);
int neuronte_gpu_opencl_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len);

/* Embedded kernels (generated at build time). */
struct embedded_ptx {
    int h;
    const unsigned char *data;
    unsigned len;
};
extern const struct embedded_ptx neuronte_gpu_ptx[];
extern const int neuronte_gpu_ptx_count;

struct embedded_spirv {
    int exact;
    const uint32_t *code;
    unsigned words;
};
extern const struct embedded_spirv neuronte_gpu_spirv[];
extern const int neuronte_gpu_spirv_count;

/* OpenCL C source of kernels/cover.cl, NUL-terminated ("" when not embedded). */
extern const char neuronte_gpu_opencl_source[];

/* blocks along z for a launch of `blocks` blocks: more when the launch is too small to fill the device */
int neuronte_gpu_splits(const neuronte_gpu *g, int blocks);
/* sets block (opt->block, else dflt), glyphs per block and the shared memory a block needs */
void neuronte_gpu_set_block(neuronte_gpu *g, const struct neuronte_gpu_options *opt, int dflt);
void neuronte_gpu_say(char *buf, size_t len, const char *fmt, ...);
uint64_t neuronte_gpu_now(void);

#endif
