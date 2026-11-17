/* neuronte-gpu core: model sets, layer-1 tables, packing, CPU path and
 * dispatch to a backend. See neuronte_gpu.h. */
#define _GNU_SOURCE
#include "internal.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define HEADER (8 + 5 * 4 + 4 + NMASK * 4)
#define BUNDLE_HEAD 28
#define BRAILLE_FIRST 0x2800u
#define DEFAULT_BATCH 1024

void neuronte_gpu_say(char *buf, size_t len, const char *fmt, ...)
{
    va_list ap;
    if (!buf || !len)
        return;
    va_start(ap, fmt);
    vsnprintf(buf, len, fmt, ap);
    va_end(ap);
}

uint64_t neuronte_gpu_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

const char *neuronte_gpu_backend_name(int backend)
{
    switch (backend) {
    case NEURONTE_GPU_CUDA: return "gpu-cuda";
    case NEURONTE_GPU_VULKAN: return "gpu-vulkan";
    case NEURONTE_GPU_OPENCL: return "gpu-opencl";
    case NEURONTE_GPU_OPENGL: return "gpu-opengl";
    case NEURONTE_GPU_CPU_FALLBACK: return "cpu-fallback";
    case NEURONTE_GPU_CPU: return "cpu";
    default: return "unknown";
    }
}

/* ---- weights files ---- */

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static float rdf(const unsigned char *p)
{
    uint32_t u = rd32(p);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static size_t net_floats(size_t h)
{
    return h * IN + h + 2 * (h * h + h) + h + 1;
}

unsigned char *neuronte_gpu_read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    unsigned char *buf = NULL;
    long n;
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) == 0 && (n = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0 &&
        (buf = malloc((size_t)n)) && fread(buf, 1, (size_t)n, f) == (size_t)n) {
        *len = (size_t)n;
    } else {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    return buf;
}

int neuronte_gpu_parse_net(const unsigned char *p, size_t len, struct net *n, char *why, size_t wl)
{
    size_t i, count;
    uint32_t h;
    if (len < HEADER || memcmp(p, "NEURONTE", 8) != 0) {
        neuronte_gpu_say(why, wl, "not a Neuronte image");
        return -1;
    }
    if (rd32(p + 8) != 1) {
        neuronte_gpu_say(why, wl, "not a version 1 distance network (version %u)", rd32(p + 8));
        return -1;
    }
    h = rd32(p + 20);
    if (rd32(p + 12) != OCT || rd32(p + 16) != CODE || rd32(p + 24) != 3 || h < 1 || h > MAX_H ||
        len != HEADER + 4 * net_floats(h)) {
        neuronte_gpu_say(why, wl, "unsupported distance network shape or length");
        return -1;
    }
    n->h = (int)h;
    n->tau = rdf(p + 28);
    if (!isfinite(n->tau) || n->tau <= 0) {
        neuronte_gpu_say(why, wl, "invalid tau");
        return -1;
    }
    for (i = 0; i < NMASK; i++) {
        n->masks[i] = rd32(p + 32 + 4 * i);
        if (n->masks[i] == 0 || n->masks[i] >= 1u << 21) {
            neuronte_gpu_say(why, wl, "invalid parity mask");
            return -1;
        }
    }
    count = net_floats(h);
    n->w = malloc(count * sizeof *n->w);
    if (!n->w) {
        neuronte_gpu_say(why, wl, "out of memory");
        return -1;
    }
    for (i = 0; i < count; i++) {
        n->w[i] = rdf(p + HEADER + 4 * i);
        if (!isfinite(n->w[i])) {
            neuronte_gpu_say(why, wl, "non-finite weight");
            free(n->w);
            n->w = NULL;
            return -1;
        }
    }
    n->w1 = n->w;
    n->b1 = n->w1 + (size_t)h * IN;
    n->w2 = n->b1 + h;
    n->b2 = n->w2 + (size_t)h * h;
    n->w3 = n->b2 + h;
    n->b3 = n->w3 + (size_t)h * h;
    n->w4 = n->b3 + h;
    n->b4 = n->w4[h];
    return 0;
}

/* The version 1 member of a version 2 bundle whose range table sends the
 * whole span [first, last] to one member. */
static int bundle_member(const unsigned char *p, size_t len, uint32_t first, uint32_t last, size_t *off_out,
                         size_t *len_out, char *why, size_t wl)
{
    uint32_t k, r, g, i, hit = UINT32_MAX, hits = 0;
    size_t off;
    if (len < BUNDLE_HEAD || memcmp(p, "NEURONTE", 8) != 0 || rd32(p + 8) != 2) {
        neuronte_gpu_say(why, wl, "not a version 2 bundle");
        return -1;
    }
    k = rd32(p + 12);
    r = rd32(p + 16);
    g = rd32(p + 24);
    if (k == 0 || k > 64 || len < BUNDLE_HEAD + 4 * (size_t)k + 12 * (size_t)r)
        goto bad;
    off = BUNDLE_HEAD + 4 * (size_t)k;
    for (i = 0; i < r; i++) {
        const unsigned char *q = p + off + 12 * (size_t)i;
        if (rd32(q) <= first && last <= rd32(q + 4)) {
            hit = rd32(q + 8);
            hits++;
        }
    }
    off += 12 * (size_t)r;
    if (g)
        off += 4 * ((size_t)NMASK + (size_t)g * CODE + g + (size_t)k * g + k);
    if (hits != 1 || hit >= k) {
        neuronte_gpu_say(why, wl, "the bundle's range table does not route %#x..%#x to one member", first, last);
        return -1;
    }
    for (i = 0; i < hit; i++)
        off += rd32(p + BUNDLE_HEAD + 4 * (size_t)i);
    *off_out = off;
    *len_out = rd32(p + BUNDLE_HEAD + 4 * (size_t)hit);
    if (off + *len_out > len)
        goto bad;
    return 0;
bad:
    neuronte_gpu_say(why, wl, "malformed bundle");
    return -1;
}

static int load_tile(neuronte_gpu *g, struct gtile *t, char *why, size_t wl)
{
    size_t len, off = 0, sub;
    unsigned char *data = neuronte_gpu_read_file(t->file, &len);
    int rc;
    if (!data) {
        neuronte_gpu_say(why, wl, "cannot read %s: %s", t->file, strerror(errno));
        return -1;
    }
    sub = len;
    if (len >= 12 && memcmp(data, "NEURONTE", 8) == 0 && rd32(data + 8) == 2) {
        if (!g->has_span) {
            free(data);
            neuronte_gpu_say(why, wl, "%s is a bundle: the layout needs a span to route", t->file);
            return -1;
        }
        if (bundle_member(data, len, g->span_first, g->span_last, &off, &sub, why, wl)) {
            free(data);
            return -1;
        }
    }
    rc = neuronte_gpu_parse_net(data + off, sub, &t->net, why, wl);
    free(data);
    if (rc)
        return -1;
    if (neuronte_open_file(t->file, &t->cpu) != NEURONTE_OK) {
        neuronte_gpu_say(why, wl, "the reference library cannot open %s", t->file);
        return -1;
    }
    return 0;
}

/* ---- layouts ---- */

static void dir_of(const char *path, char *dir, size_t len)
{
    const char *slash = strrchr(path, '/');
    if (!slash)
        snprintf(dir, len, ".");
    else
        snprintf(dir, len, "%.*s", (int)(slash - path), path);
}

static int read_layout(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t wl)
{
    FILE *f = fopen(opt->layout_path, "r");
    char line[2048], dir[1024];
    int lineno = 0, version = 0;
    if (!f) {
        neuronte_gpu_say(why, wl, "cannot open %s: %s", opt->layout_path, strerror(errno));
        return -1;
    }
    if (opt->model_dir)
        snprintf(dir, sizeof dir, "%s", opt->model_dir);
    else
        dir_of(opt->layout_path, dir, sizeof dir);
    while (fgets(line, sizeof line, f)) {
        char key[32] = "", file[1024];
        unsigned long a, b;
        int mask, x, y, w, h;
        lineno++;
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0] || line[0] == '#')
            continue;
        if (sscanf(line, "%31s", key) != 1)
            continue;
        if (!strcmp(key, "neuronte-gpu-layout")) {
            version = atoi(line + strlen(key));
        } else if (!strcmp(key, "name")) {
            if (snprintf(g->name, sizeof g->name, "%s", line + 5) >= (int)sizeof g->name)
                goto bad;
        } else if (!strcmp(key, "cell")) {
            if (sscanf(line, "cell %d %d", &g->cell_w, &g->cell_h) != 2)
                goto bad;
        } else if (!strcmp(key, "samples")) {
            if (sscanf(line, "samples %d", &g->samples) != 1)
                goto bad;
        } else if (!strcmp(key, "span")) {
            if (sscanf(line, "span %li %li", (long *)&a, (long *)&b) != 2 || a > b || b > 0x10FFFF)
                goto bad;
            g->has_span = 1;
            g->span_first = (uint32_t)a;
            g->span_last = (uint32_t)b;
        } else if (!strcmp(key, "tile")) {
            struct gtile *t;
            if (g->ntiles >= MAX_TILES ||
                sscanf(line, "tile %d %d %d %d %d %1023s", &mask, &x, &y, &w, &h, file) != 6)
                goto bad;
            t = &g->tiles[g->ntiles++];
            t->mask = mask;
            t->x = x;
            t->y = y;
            t->w = w;
            t->h = h;
            if ((file[0] == '/' ? snprintf(t->file, sizeof t->file, "%s", file)
                                : snprintf(t->file, sizeof t->file, "%s/%s", dir, file)) >= (int)sizeof t->file)
                goto bad;
        } else {
            /* unknown keys are ignored, so later writers can add fields */
        }
    }
    fclose(f);
    if (version != 1) {
        neuronte_gpu_say(why, wl, "%s is not a version 1 neuronte-gpu layout", opt->layout_path);
        return -1;
    }
    return 0;
bad:
    fclose(f);
    neuronte_gpu_say(why, wl, "%s:%d: malformed line", opt->layout_path, lineno);
    return -1;
}

static int check_tiles(neuronte_gpu *g, char *why, size_t wl)
{
    int i, x, y;
    unsigned char *own;
    if (g->cell_w <= 0 || g->cell_h <= 0 || g->cell_w > NEURONTE_MAX_CELL || g->cell_h > NEURONTE_MAX_CELL ||
        g->samples < 0 || g->samples > NEURONTE_MAX_SAMPLES || g->ntiles < 1) {
        neuronte_gpu_say(why, wl, "invalid cell, samples or tile count");
        return -1;
    }
    own = calloc((size_t)g->cell_w * g->cell_h, 1);
    if (!own)
        return -1;
    for (i = 0; i < g->ntiles; i++) {
        struct gtile *t = &g->tiles[i];
        if (t->w <= 0 || t->h <= 0 || t->x < 0 || t->y < 0 || t->x + t->w > g->cell_w ||
            t->y + t->h > g->cell_h || t->mask < -1 || t->mask > 0xFF)
            goto bad;
        for (y = t->y; y < t->y + t->h; y++)
            for (x = t->x; x < t->x + t->w; x++) {
                if (own[(size_t)y * g->cell_w + x])
                    goto bad;
                own[(size_t)y * g->cell_w + x] = 1;
            }
    }
    for (x = 0; x < g->cell_w * g->cell_h; x++)
        if (!own[x])
            goto bad;
    free(own);
    return 0;
bad:
    free(own);
    neuronte_gpu_say(why, wl, "the tiles do not cover the cell exactly once");
    return -1;
}

/* ---- layer 1 tables and packing ---- */

/* One axis's columns of layer 1 at coordinate t, computed exactly as the
 * reference library's axis_part (same operations, same order). */
static void axis_part(const struct net *n, int axis, double t, float *out)
{
    const double two_pi = 6.283185307179586476925286766559;
    float v = axis == 0 ? (float)(2.0 * t - 1.0) : (float)(t - 1.0);
    int k, o, h = n->h;
    for (o = 0; o < h; o++)
        out[o] = v * n->w1[(size_t)o * IN + axis];
    for (k = 0; k < OCT; k++) {
        double w = two_pi * (double)(1 << k);
        float sv = (float)sin(w * t), cv = (float)cos(w * t);
        int fs = 2 + 4 * k + axis, fc = 4 + 4 * k + axis;
        for (o = 0; o < h; o++)
            out[o] += sv * n->w1[(size_t)o * IN + fs] + cv * n->w1[(size_t)o * IN + fc];
    }
}

static size_t up4(size_t v)
{
    return (v + 3) & ~(size_t)3;
}

static int pack(neuronte_gpu *g, char *why, size_t wl)
{
    int t, h = g->hidden, s = g->samples;
    size_t total = 0, off, i, j;
    for (t = 0; t < g->ntiles; t++) {
        const struct gtile *tl = &g->tiles[t];
        total += up4((size_t)(CODE + 1) * h + 2 * (size_t)h * h + 3 * (size_t)h + 2);
        total += up4((size_t)tl->w * s * h) + up4((size_t)tl->h * s * h);
    }
    if (total > (size_t)INT32_MAX) {
        neuronte_gpu_say(why, wl, "model set too large");
        return -1;
    }
    g->pk.params = calloc(total, sizeof(float));
    if (!g->pk.params) {
        neuronte_gpu_say(why, wl, "out of memory");
        return -1;
    }
    g->pk.nparams = total;
    off = 0;
    for (t = 0; t < g->ntiles; t++) {
        const struct gtile *tl = &g->tiles[t];
        const struct net *n = &tl->net;
        int32_t *rec = g->pk.tiles + t * T_INTS;
        float *p = g->pk.params + off;
        int sw = tl->w * s, sh = tl->h * s;
        rec[T_X] = tl->x;
        rec[T_Y] = tl->y;
        rec[T_W] = tl->w;
        rec[T_H] = tl->h;
        rec[T_MASK] = tl->mask;
        rec[T_PARAM] = (int32_t)off;
        rec[T_MASKS] = t * NMASK;
        memcpy(g->pk.masks + t * NMASK, n->masks, sizeof n->masks);
        /* code columns of layer 1, [input][output]; then b1 */
        for (i = 0; i < CODE; i++)
            for (j = 0; j < (size_t)h; j++)
                p[i * h + j] = n->w1[j * IN + POS + i];
        memcpy(p + (size_t)CODE * h, n->b1, (size_t)h * sizeof *p);
        p += (size_t)(CODE + 1) * h;
        /* w2, b2, w3, b3, w4 in file order ([out][in]), then b4 and the slope:
         * n->tau * (float)cell_w * (float)ss for this tile, as the CPU computes it */
        memcpy(p, n->w2, (2 * (size_t)h * h + 3 * (size_t)h) * sizeof *p);
        p[2 * (size_t)h * h + 3 * (size_t)h] = n->b4;
        p[2 * (size_t)h * h + 3 * (size_t)h + 1] = n->tau * (float)tl->w * (float)s;
        off += up4((size_t)(CODE + 1) * h + 2 * (size_t)h * h + 3 * (size_t)h + 2);
        rec[T_XTAB] = (int32_t)off;
        for (i = 0; i < (size_t)sw; i++)
            axis_part(n, 0, (i + 0.5) / sw, g->pk.params + off + i * h);
        off += up4((size_t)sw * h);
        rec[T_YTAB] = (int32_t)off;
        for (j = 0; j < (size_t)sh; j++)
            axis_part(n, 1, (j + 0.5) / sh * 2.0, g->pk.params + off + j * h);
        off += up4((size_t)sh * h);
    }
    return 0;
}

void neuronte_gpu_set_block(neuronte_gpu *g, const struct neuronte_gpu_options *opt, int dflt)
{
    int t, pmin = INT_MAX;
    size_t h = (size_t)g->hidden;

    g->block = opt->block > 0 ? opt->block : dflt;
    if (g->block < g->samples * g->samples) /* a block takes at least one pixel's samples per pass */
        g->block = g->samples * g->samples;
    /* glyphs per block: enough work per block for the smallest tile */
    g->max_tile_px = 0;
    for (t = 0; t < g->ntiles; t++) {
        if (g->tiles[t].w * g->tiles[t].h < pmin)
            pmin = g->tiles[t].w * g->tiles[t].h;
        if (g->tiles[t].w * g->tiles[t].h > g->max_tile_px)
            g->max_tile_px = g->tiles[t].w * g->tiles[t].h;
    }
    g->gpb = (2 * g->block + pmin - 1) / pmin;
    if (g->gpb < 1)
        g->gpb = 1;
    if (g->gpb > 64)
        g->gpb = 64;
    g->pk.shared_floats = (int)(up4(2 * h * h + 3 * h + 2) + (size_t)g->gpb * h + (size_t)g->block);
}

int neuronte_gpu_splits(const neuronte_gpu *g, int blocks)
{
    int nz, cap;
    if (blocks <= 0 || blocks >= g->fill_blocks)
        return 1;
    nz = (g->fill_blocks + blocks - 1) / blocks;
    cap = (g->gpb * g->max_tile_px * g->samples * g->samples + 31) / 32; /* at least 32 samples per block */
    if (nz > cap)
        nz = cap;
    return nz < 1 ? 1 : nz;
}

/* ---- CPU path ---- */

static uint32_t tile_scalar(const struct gtile *t, uint32_t s)
{
    return t->mask < 0 ? s : BRAILLE_FIRST | ((s - BRAILLE_FIRST) & (uint32_t)t->mask);
}

static int cpu_glyph(neuronte_gpu *g, uint32_t s, float *dst)
{
    int t, y;
    for (t = 0; t < g->ntiles; t++) {
        const struct gtile *tl = &g->tiles[t];
        if (neuronte_coverage(tl->cpu, tile_scalar(tl, s), tl->w, tl->h, g->samples, g->cpu_buf) != NEURONTE_OK)
            return -1;
        for (y = 0; y < tl->h; y++)
            memcpy(dst + (size_t)(tl->y + y) * g->cell_w + tl->x, g->cpu_buf + (size_t)y * tl->w,
                   (size_t)tl->w * sizeof *dst);
    }
    return 0;
}

/* ---- public API ---- */

static void free_tiles(neuronte_gpu *g)
{
    int i;
    for (i = 0; i < g->ntiles; i++) {
        free(g->tiles[i].net.w);
        if (g->tiles[i].cpu)
            neuronte_close(g->tiles[i].cpu);
    }
}

void neuronte_gpu_close(neuronte_gpu *g)
{
    if (!g)
        return;
    if (g->ops && g->impl)
        g->ops->close(g->impl);
    neuronte_gpu_v2_close(g);
    free_tiles(g);
    free(g->pk.params);
    free(g->cpu_buf);
    free(g);
}

int neuronte_gpu_open(const struct neuronte_gpu_options *opt, neuronte_gpu **out, char *reason,
                      size_t reason_len)
{
    neuronte_gpu *g;
    char why[512] = "";
    int i, wmax = 0, hmax = 0, want = opt ? opt->backend : 0;

    if (!opt || !out || (!opt->layout_path && !opt->nrt_path)) {
        neuronte_gpu_say(reason, reason_len, "a layout or an .nrt file is required");
        return -1;
    }
    *out = NULL;
    g = calloc(1, sizeof *g);
    if (!g)
        return -1;
    g->exact = !(opt->flags & NEURONTE_GPU_FAST);
    g->max_batch = opt->max_batch > 0 ? opt->max_batch : DEFAULT_BATCH;
    g->device_type = "";
    if (!opt->layout_path &&
        neuronte_gpu_v2_wanted(opt->nrt_path, opt->backend == NEURONTE_GPU_OPENGL ? opt->flags | NEURONTE_GPU_FILE : opt->flags)) {
        if (neuronte_gpu_v2_open(g, opt, reason, reason_len))
            goto fail;
        *out = g;
        return 0;
    }
    if (opt->layout_path) {
        if (read_layout(g, opt, reason, reason_len))
            goto fail;
    } else {
        struct gtile *t = &g->tiles[0];
        g->cell_w = opt->cell_w;
        g->cell_h = opt->cell_h;
        g->ntiles = 1;
        t->mask = -1;
        t->w = opt->cell_w;
        t->h = opt->cell_h;
        snprintf(t->file, sizeof t->file, "%s", opt->nrt_path);
        snprintf(g->name, sizeof g->name, "%s", opt->nrt_path);
    }
    if (opt->samples)
        g->samples = opt->samples;
    if (g->samples == 0)
        g->samples = g->cell_w < 24 ? 2 : 1; /* the library's choice, from the full cell width */
    if (check_tiles(g, reason, reason_len))
        goto fail;
    for (i = 0; i < g->ntiles; i++) {
        if (load_tile(g, &g->tiles[i], reason, reason_len))
            goto fail;
        if (g->tiles[i].w > wmax)
            wmax = g->tiles[i].w;
        if (g->tiles[i].h > hmax)
            hmax = g->tiles[i].h;
    }
    g->cpu_buf = malloc((size_t)wmax * hmax * sizeof *g->cpu_buf);
    if (!g->cpu_buf)
        goto fail;
    g->hidden = g->tiles[0].net.h;
    for (i = 1; i < g->ntiles; i++)
        if (g->tiles[i].net.h != g->hidden)
            g->hidden = 0;

    if (want == NEURONTE_GPU_CPU) {
        g->backend = NEURONTE_GPU_CPU;
        *out = g;
        return 0;
    }
    if (!g->hidden)
        neuronte_gpu_say(why, sizeof why, "the tiles have different hidden widths");
    else if ((int64_t)wmax * hmax * g->samples * g->samples * 64 > INT_MAX) /* 64 glyphs per block at most */
        neuronte_gpu_say(why, sizeof why, "a %dx%d tile with %d samples per axis is too large for the GPU kernel",
                         wmax, hmax, g->samples);
    else if (pack(g, why, sizeof why) == 0) {
        char w1[512] = "", w2[512] = "", w3[512] = "";
        int rc = -1;
        /* Arm's kernel driver (kbase, /dev/mali0) comes with Arm's user-space drivers, whose
           OpenCL runs this workload faster than their Vulkan (see README), so AUTO tries OpenCL
           before Vulkan there. */
        const int opencl_first = want == NEURONTE_GPU_AUTO && access("/dev/mali0", F_OK) == 0;
        if (want == NEURONTE_GPU_AUTO || want == NEURONTE_GPU_CUDA) {
            rc = neuronte_gpu_cuda_open(g, opt, w1, sizeof w1);
            if (rc == 0)
                g->backend = NEURONTE_GPU_CUDA;
        }
        if (rc && opencl_first) {
            rc = neuronte_gpu_opencl_open(g, opt, w3, sizeof w3);
            if (rc == 0)
                g->backend = NEURONTE_GPU_OPENCL;
        }
        if (rc && (want == NEURONTE_GPU_AUTO || want == NEURONTE_GPU_VULKAN)) {
            rc = neuronte_gpu_vulkan_open(g, opt, w2, sizeof w2);
            if (rc == 0)
                g->backend = NEURONTE_GPU_VULKAN;
        }
        if (rc && !opencl_first && (want == NEURONTE_GPU_AUTO || want == NEURONTE_GPU_OPENCL)) {
            rc = neuronte_gpu_opencl_open(g, opt, w3, sizeof w3);
            if (rc == 0)
                g->backend = NEURONTE_GPU_OPENCL;
        }
        if (rc == 0) {
            *out = g;
            return 0;
        }
        neuronte_gpu_say(why, sizeof why, "%s%s%s%s%s%s%s", w1[0] ? "cuda: " : "", w1,
                         w1[0] && (w2[0] || w3[0]) ? "; " : "", w2[0] ? "vulkan: " : "", w2,
                         w2[0] && w3[0] ? "; opencl: " : w3[0] ? "opencl: " : "", w3);
    }
    if (!(opt->flags & NEURONTE_GPU_ALLOW_FALLBACK)) {
        neuronte_gpu_say(reason, reason_len, "no GPU backend: %s", why);
        goto fail;
    }
    g->backend = NEURONTE_GPU_CPU_FALLBACK;
    snprintf(g->reason, sizeof g->reason, "%s", why);
    *out = g;
    return 0;
fail:
    neuronte_gpu_close(g);
    return -1;
}

int neuronte_gpu_backend(const neuronte_gpu *g) { return g ? g->backend : 0; }
const char *neuronte_gpu_fallback_reason(const neuronte_gpu *g) { return g ? g->reason : ""; }
const char *neuronte_gpu_device(const neuronte_gpu *g) { return g ? g->device : ""; }
const char *neuronte_gpu_versions(const neuronte_gpu *g) { return g ? g->versions : ""; }
const char *neuronte_gpu_device_type(const neuronte_gpu *g) { return g ? g->device_type : ""; }
int neuronte_gpu_samples(const neuronte_gpu *g) { return g ? g->samples : 0; }
int neuronte_gpu_hidden(const neuronte_gpu *g) { return g ? g->hidden : 0; }
int neuronte_gpu_tiles(const neuronte_gpu *g) { return !g ? 0 : g->v2 ? g->nmembers : g->ntiles; }
int neuronte_gpu_exact(const neuronte_gpu *g) { return g ? g->exact : 0; }

int neuronte_gpu_memory(const neuronte_gpu *g, struct neuronte_gpu_memory *m)
{
    if (!g || !m)
        return -1;
    m->buffers = g->device_bytes;
    m->staging = g->staging_bytes;
    m->measured = g->measured_bytes;
    m->shared_context = g->impl && !g->context_fresh && g->backend == NEURONTE_GPU_CUDA;
    m->weights = g->v2 ? g->plan.nstatic * sizeof(float) : g->pk.nparams * sizeof(float);
    return 0;
}

int neuronte_gpu_cell(const neuronte_gpu *g, int *w, int *h)
{
    if (!g)
        return -1;
    if (w)
        *w = g->cell_w;
    if (h)
        *h = g->cell_h;
    return 0;
}

static uint32_t normalize(uint32_t s)
{
    return (s > 0x10FFFF || (s >= 0xD800 && s <= 0xDFFF)) ? NEURONTE_REPLACEMENT : s;
}

int neuronte_gpu_coverage(neuronte_gpu *g, const uint32_t *scalars, int count, float *out,
                          struct neuronte_gpu_timing *timing)
{
    struct neuronte_gpu_timing t = {0};
    size_t plane;
    uint64_t t0 = neuronte_gpu_now(), c0;
    int i;
    if (!g || !scalars || !out || count < 0)
        return -1;
    if (g->v2)
        return neuronte_gpu_v2_coverage(g, scalars, count, out, timing);
    plane = (size_t)g->cell_w * g->cell_h;
    if (g->backend == NEURONTE_GPU_CPU || g->backend == NEURONTE_GPU_CPU_FALLBACK) {
        for (i = 0; i < count; i++)
            if (cpu_glyph(g, scalars[i], out + (size_t)i * plane))
                return -1;
        t.cpu_eval = neuronte_gpu_now() - t0;
        t.cpu_glyphs = count;
    } else {
        for (i = 0; i < count; i += g->max_batch) {
            int n = count - i < g->max_batch ? count - i : g->max_batch;
            struct neuronte_gpu_timing part = {0};
            if (g->ops->run(g->impl, scalars + i, n, out + (size_t)i * plane, &part))
                return -1;
            t.upload += part.upload;
            t.run += part.run;
            t.wait += part.wait;
            t.download += part.download;
            t.launches += part.launches;
        }
        /* A bundle's member was resolved for the layout's span; anything outside
         * it is drawn by the reference library, which routes it itself. */
        if (g->has_span) {
            c0 = neuronte_gpu_now();
            for (i = 0; i < count; i++) {
                uint32_t s = normalize(scalars[i]);
                if (s < g->span_first || s > g->span_last) {
                    if (cpu_glyph(g, scalars[i], out + (size_t)i * plane))
                        return -1;
                    t.cpu_glyphs++;
                }
            }
            t.cpu_eval = neuronte_gpu_now() - c0;
        }
    }
    t.total = neuronte_gpu_now() - t0;
    if (timing)
        *timing = t;
    return 0;
}
