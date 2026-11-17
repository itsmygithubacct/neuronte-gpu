/* neuronte-gpu file handles: what Neuronte's weight files hold beyond one
 * narrow net. Bundles with their router, tiled nets, drawing files with a
 * letter edge table, and distance nets of any hidden width.
 *
 * The file is opened by the reference library too. The library routes each
 * scalar (range table, gate, fallback) and says which scalars a bundle was
 * not trained on, so those answers are the library's own. This file reads
 * the weights from the same bytes and lays the work out for a GPU: see
 * "file handles" in internal.h. Glyphs the GPU path does not draw (colour
 * nets, the untrained mark) are drawn by the library and counted.
 *
 * The position arithmetic below (tile_axis, tile_axis_at, the snapping
 * knots, axis_part) is the library's, operation for operation, so the
 * tables the GPU reads are the tables the library would compute. */
#define _GNU_SOURCE
#include "internal.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUNDLE_HEAD 28
#define DRAWING_HEAD 16
#define TILED_HEAD 32
#define LETTERS_HEAD 20
#define MAX_TILE_AXIS 16
#define SHARED_FLOATS_MAX 12288 /* 48 KB, every CUDA device's default limit per block */
#define DYNAMIC_GLYPHS 256      /* warped letters a call can carry before it is split */

enum { M_PLAIN, M_TILED, M_OTHER };

struct vnet {
    struct net net;
    int kern;      /* index into plan.kern */
    int32_t param; /* offset of its weights in plan.params */
    int32_t masks; /* offset of its parity masks in plan.masks */
};

struct vtile {
    uint32_t net, mask, base, extract; /* net indexes v2.nets */
};

struct vmember {
    int kind;
    int net;        /* M_PLAIN */
    uint32_t first; /* M_TILED */
    int cols, rows;
    double xe[MAX_TILE_AXIS + 1], ye[MAX_TILE_AXIS + 1];
    struct vtile tile[MAX_TILE_AXIS * MAX_TILE_AXIS];
    int seg0, nseg; /* its segments on the even grid */
    int reduce;     /* those go through the sample buffer */
    size_t dyn_floats; /* table floats a warped glyph of this member can need */
};

/* The host's side of a segment. */
struct vseg {
    int member, tile; /* tile -1: a plain net */
    int kern;
    int to_samples;
    int gp; /* pairs per block */
};

struct edge {
    float pos;
    int rising;
};

struct letter {
    uint32_t scalar;
    int nx, ny;
    struct edge *x, *y;
};

struct warp_pair {
    double p, t;
};

struct triple {
    int seg;
    uint32_t glyph, code;
};

struct v2 {
    neuronte *cpu;
    struct vnet *nets;
    int nnets;
    struct vmember *members;
    int nmembers;
    struct vseg *seg; /* parallel to plan.segs */
    struct letter *letters;
    int nletters;
    struct edge *pool;
    int snap_file, snap_below, untrained;
    int sw, sh;
    size_t dyn_floats; /* capacity of the per-call part of params */
    /* per call */
    struct triple *triples;
    int ntriples, triples_cap;
    int *count, *start; /* per segment */
    int *cpu_list, ncpu;
    double *gx, *gy;
    /* text: coverage kept per scalar, an open-addressed table */
    uint32_t *kept_key; /* scalar + 1; 0 is empty */
    float *kept;
    int kept_cap, kept_n, binary;
    uint32_t *line, *fresh;
    int line_cap;
};

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint32_t rd16(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8;
}

static float rdf(const unsigned char *p)
{
    uint32_t u = rd32(p);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static uint32_t normalize(uint32_t s)
{
    return (s > 0x10FFFF || (s >= 0xD800 && s <= 0xDFFF)) ? NEURONTE_REPLACEMENT : s;
}

static size_t up4(size_t v)
{
    return (v + 3) & ~(size_t)3;
}

/* ---- reading the file ---- */

static int add_net(struct v2 *v, const unsigned char *p, size_t len, char *why, size_t wl)
{
    struct vnet *grown = realloc(v->nets, ((size_t)v->nnets + 1) * sizeof *grown);
    if (!grown)
        return -1;
    v->nets = grown;
    memset(&v->nets[v->nnets], 0, sizeof *grown);
    if (neuronte_gpu_parse_net(p, len, &v->nets[v->nnets].net, why, wl))
        return -1;
    return v->nnets++;
}

static struct vmember *add_member(struct v2 *v)
{
    struct vmember *grown = realloc(v->members, ((size_t)v->nmembers + 1) * sizeof *grown);
    if (!grown)
        return NULL;
    v->members = grown;
    memset(&v->members[v->nmembers], 0, sizeof *grown);
    return &v->members[v->nmembers++];
}

/* One member that draws: a distance net, a tiled net, or something the GPU
 * path leaves to the library (a colour net). The library has already checked
 * the image, so lengths and counts here are in range. */
static int read_member(struct v2 *v, const unsigned char *p, size_t len, char *why, size_t wl)
{
    struct vmember *m = add_member(v);
    uint32_t ver = rd32(p + 8);
    if (!m)
        return -1;
    if (ver == 1) {
        int n = add_net(v, p, len, why, wl);
        if (n < 0)
            return -1;
        m = &v->members[v->nmembers - 1];
        m->kind = M_PLAIN;
        m->net = n;
    } else if (ver == 6) {
        uint32_t c = rd32(p + 12), r = rd32(p + 16), k = rd32(p + 20), i;
        size_t prefix = TILED_HEAD + 4 * ((size_t)c + 1 + r + 1) + 16 * (size_t)c * r + 4 * (size_t)k, off = prefix;
        const unsigned char *q = p + TILED_HEAD;
        int base = v->nnets;
        m->kind = M_TILED;
        m->cols = (int)c;
        m->rows = (int)r;
        m->first = rd32(p + 24);
        for (i = 0; i <= c; i++, q += 4)
            m->xe[i] = rdf(q);
        for (i = 0; i <= r; i++, q += 4)
            m->ye[i] = rdf(q);
        for (i = 0; i < c * r; i++, q += 16) {
            m->tile[i].net = (uint32_t)base + rd32(q);
            m->tile[i].mask = rd32(q + 4);
            m->tile[i].base = rd32(q + 8);
            m->tile[i].extract = rd32(q + 12);
        }
        for (i = 0; i < k; i++) {
            size_t ml = rd32(p + prefix - 4 * (size_t)k + 4 * (size_t)i);
            if (add_net(v, p + off, ml, why, wl) < 0)
                return -1;
            off += ml;
        }
    } else {
        m->kind = M_OTHER;
    }
    return 0;
}

static int read_bundle(struct v2 *v, const unsigned char *p, size_t len, char *why, size_t wl)
{
    uint32_t k = rd32(p + 12), r = rd32(p + 16), g = rd32(p + 24), i;
    size_t off = BUNDLE_HEAD + 4 * (size_t)k + 12 * (size_t)r;
    if (g)
        off += 4 * ((size_t)NMASK + (size_t)g * CODE + g + (size_t)k * g + k);
    for (i = 0; i < k; i++) {
        size_t ml = rd32(p + BUNDLE_HEAD + 4 * (size_t)i);
        if (off + ml > len || read_member(v, p + off, ml, why, wl))
            return -1;
        off += ml;
    }
    return 0;
}

static int read_letters(struct v2 *v, const unsigned char *p, size_t len)
{
    uint32_t count = rd32(p + 16), e;
    size_t at, edges = 0, used = 0;
    for (e = 0, at = LETTERS_HEAD; e < count; e++) {
        uint32_t nx = rd16(p + at + 4), ny = rd16(p + at + 6);
        at += 8 + 8 * (size_t)(nx + ny);
        edges += (size_t)nx + ny;
    }
    if (at != len)
        return -1;
    v->letters = calloc(count, sizeof *v->letters);
    v->pool = calloc(edges ? edges : 1, sizeof *v->pool);
    if (!v->letters || !v->pool)
        return -1;
    v->nletters = (int)count;
    v->snap_file = (int)rd32(p + 12);
    for (e = 0, at = LETTERS_HEAD; e < count; e++) {
        struct letter *l = &v->letters[e];
        int t;
        l->scalar = rd32(p + at);
        l->nx = (int)rd16(p + at + 4);
        l->ny = (int)rd16(p + at + 6);
        at += 8;
        l->x = v->pool + used;
        for (t = 0; t < l->nx; t++, at += 8, used++) {
            v->pool[used].pos = rdf(p + at);
            v->pool[used].rising = (int)rd32(p + at + 4);
        }
        l->y = v->pool + used;
        for (t = 0; t < l->ny; t++, at += 8, used++) {
            v->pool[used].pos = rdf(p + at);
            v->pool[used].rising = (int)rd32(p + at + 4);
        }
    }
    return 0;
}

/* Members in the order the library numbers its nets. */
static int read_image(struct v2 *v, const unsigned char *p, size_t len, char *why, size_t wl)
{
    uint32_t ver = rd32(p + 8), i;
    if (ver == 2)
        return read_bundle(v, p, len, why, wl);
    if (ver != 5)
        return read_member(v, p, len, why, wl);
    for (i = 0; i < rd32(p + 12); i++) {
        size_t off = rd32(p + DRAWING_HEAD + 8 * (size_t)i), ml = rd32(p + DRAWING_HEAD + 8 * (size_t)i + 4);
        const unsigned char *m = p + off;
        int rc = 0;
        switch (rd32(m + 8)) {
        case 2:
            rc = read_bundle(v, m, ml, why, wl);
            break;
        case 4:
            break; /* advances: the library measures */
        case 7:
            rc = read_letters(v, m, ml);
            break;
        default:
            rc = read_member(v, m, ml, why, wl);
        }
        if (rc)
            return -1;
    }
    return 0;
}

/* ---- the library's position arithmetic ---- */

/* One axis's columns of layer 1 at coordinate t, exactly as the reference
 * library's axis_part. */
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

static void tile_axis(const double *edge, int tiles, int n, double scale, int *first, double *pos)
{
    int k, i;
    first[0] = 0;
    for (k = 1; k < tiles; k++) {
        double at = ceil(edge[k] / scale * (double)n - 0.5);
        first[k] = at < 0.0 ? 0 : at > (double)n ? n : (int)at;
        if (first[k] < first[k - 1])
            first[k] = first[k - 1];
    }
    first[tiles] = n;
    for (k = 0; k < tiles; k++) {
        double off = edge[k] / scale * (double)n;
        double span = (edge[k + 1] - edge[k]) / scale * (double)n;
        for (i = first[k]; i < first[k + 1]; i++)
            pos[i] = ((double)i + 0.5 - off) / span * scale;
    }
}

static void tile_axis_at(const double *edge, int tiles, int n, double scale, const double *g, int *first,
                         double *pos)
{
    int k, i = 0;
    first[0] = 0;
    for (k = 1; k < tiles; k++) {
        while (i < n && g[i] < edge[k])
            i++;
        first[k] = i;
    }
    first[tiles] = n;
    for (k = 0; k < tiles; k++)
        for (i = first[k]; i < first[k + 1]; i++)
            pos[i] = (g[i] - edge[k]) / (edge[k + 1] - edge[k]) * scale;
}

static uint32_t tile_code(const struct vmember *m, const struct vtile *tl, uint32_t s)
{
    uint32_t b = s - m->first, v = 0, bit = 1, k;
    if (!tl->extract)
        return tl->base | (b & tl->mask);
    for (k = tl->mask; k; k &= k - 1, bit <<= 1)
        if (b & (k & -k))
            v |= bit;
    return tl->base + v;
}

static int cmp_edge(const void *a, const void *b)
{
    const struct edge *x = a, *y = b;
    if (x->pos < y->pos)
        return -1;
    if (x->pos > y->pos)
        return 1;
    return x->rising - y->rising;
}

static int cmp_pair(const void *a, const void *b)
{
    const struct warp_pair *x = a, *y = b;
    if (x->p < y->p)
        return -1;
    if (x->p > y->p)
        return 1;
    if (x->t < y->t)
        return -1;
    if (x->t > y->t)
        return 1;
    return 0;
}

static double round_half(double v, double centre)
{
    double f = floor(v);
    if (v - f == 0.5)
        return v > centre ? f : f + 1.0;
    return floor(v + 0.5);
}

static double interp_knots(double u, const double *xp, const double *fp, int n)
{
    int i;
    if (n <= 1 || u <= xp[0])
        return fp[0];
    if (u >= xp[n - 1])
        return fp[n - 1];
    for (i = 1; i < n; i++) {
        if (u <= xp[i]) {
            double span = xp[i] - xp[i - 1];
            if (span == 0.0)
                return fp[i - 1];
            return fp[i - 1] + (u - xp[i - 1]) * (fp[i] - fp[i - 1]) / span;
        }
    }
    return fp[n - 1];
}

/* The warped sample positions of one axis of a letter: the library's
 * build_knots and axis_samples for one layer of edges. */
static int axis_samples(const struct letter *rec, int axis, int n_px, int ss, double scale, double extent,
                        double *out)
{
    const struct edge *src = axis == 0 ? rec->x : rec->y;
    int n = axis == 0 ? rec->nx : rec->ny, np = 0, nk, e, i;
    struct warp_pair *pairs = malloc(((size_t)n + 1) * sizeof *pairs);
    struct warp_pair *kn = malloc(((size_t)n + 2) * sizeof *kn);
    struct edge *ed = malloc(((size_t)n + 1) * sizeof *ed);
    double *orig = malloc(((size_t)n + 2) * sizeof *orig), *snapped = malloc(((size_t)n + 2) * sizeof *snapped);
    int rc = -1;

    if (!pairs || !kn || !ed || !orig || !snapped)
        goto done;
    memcpy(ed, src, (size_t)n * sizeof *ed);
    if (n)
        qsort(ed, (size_t)n, sizeof *ed, cmp_edge);
    for (e = 0; e < n;) {
        double p = (double)ed[e].pos * scale;
        if (ed[e].rising && e + 1 < n && !ed[e + 1].rising) {
            double q = (double)ed[e + 1].pos * scale;
            double w = ceil(q - p - 0.5);
            double left;
            if (w < 1.0)
                w = 1.0;
            left = round_half((p + q) / 2.0 - w / 2.0, extent / 2.0 - w / 2.0);
            pairs[np].p = p;
            pairs[np++].t = left;
            pairs[np].p = q;
            pairs[np++].t = left + w;
            e += 2;
        } else {
            pairs[np].p = p;
            pairs[np++].t = round_half(p, extent / 2.0);
            e++;
        }
    }
    if (np > 1)
        qsort(pairs, (size_t)np, sizeof *pairs, cmp_pair);
    kn[0].p = 0;
    kn[0].t = 0;
    nk = 1;
    for (i = 0; i < np; i++) {
        double p = pairs[i].p, t = pairs[i].t, lp = kn[nk - 1].p, lt = kn[nk - 1].t;
        if (p - lp < 0.5 || p >= extent)
            continue;
        if (t < lt + 1.0)
            t = lt + 1.0;
        if (t >= extent)
            break;
        kn[nk].p = p;
        kn[nk].t = t;
        nk++;
    }
    kn[nk].p = extent;
    kn[nk].t = extent;
    nk++;
    for (i = 0; i < nk; i++) {
        orig[i] = kn[i].p;
        snapped[i] = kn[i].t;
    }
    n = n_px * ss;
    for (i = 0; i < n; i++) {
        double u = ((double)i + 0.5) / (double)ss;
        out[i] = interp_knots(u, snapped, orig, nk) / scale;
    }
    rc = 0;
done:
    free(pairs);
    free(kn);
    free(ed);
    free(orig);
    free(snapped);
    return rc;
}

static const struct letter *find_letter(const struct v2 *v, uint32_t s)
{
    int lo = 0, hi = v->nletters;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (v->letters[mid].scalar < s)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < v->nletters && v->letters[lo].scalar == s ? &v->letters[lo] : NULL;
}

/* ---- segments ---- */

static int grow_params(struct v2plan *pl, size_t need)
{
    float *grown;
    size_t cap;
    if (pl->nparams + need <= pl->params_cap)
        return 0;
    cap = pl->params_cap ? pl->params_cap * 2 : 1 << 16;
    while (cap < pl->nparams + need)
        cap *= 2;
    if (cap > (size_t)INT32_MAX)
        return -1;
    grown = realloc(pl->params, cap * sizeof *grown);
    if (!grown)
        return -1;
    memset(grown + pl->params_cap, 0, (cap - pl->params_cap) * sizeof *grown);
    pl->params = grown;
    pl->params_cap = cap;
    return 0;
}

static int grow_segs(neuronte_gpu *g)
{
    struct v2plan *pl = &g->plan;
    int32_t *a;
    struct vseg *b;
    int cap;
    if (pl->nsegs < pl->segs_cap)
        return 0;
    cap = pl->segs_cap ? pl->segs_cap * 2 : 64;
    a = realloc(pl->segs, (size_t)cap * S_INTS * sizeof *a);
    if (!a)
        return -1;
    pl->segs = a;
    b = realloc(g->v2->seg, (size_t)cap * sizeof *b);
    if (!b)
        return -1;
    g->v2->seg = b;
    pl->segs_cap = cap;
    return 0;
}

/* Pairs of one segment a block takes, by the work a pair is. */
static int seg_gp(const neuronte_gpu *g, const struct v2kern *k, const struct vseg *sg, const int32_t *rec)
{
    int ss = g->samples * g->samples, items, gp, cap;
    if (k->wide) {
        items = rec[S_W] * rec[S_H];
        gp = (4 * k->slots + items - 1) / items;
        cap = 8;
    } else {
        /* about two passes of the block's threads over pixels */
        items = sg->to_samples ? rec[S_W] * rec[S_H] : (rec[S_W] / g->samples) * (rec[S_H] / g->samples);
        gp = ((sg->to_samples ? ss : 1) * 2 * k->threads + items - 1) / items;
        cap = 64;
    }
    return gp < 1 ? 1 : gp > cap ? cap : gp;
}

/* Append one segment and its tables. With fixed set, the per-call part of
 * params may not grow: -2 when it is full. */
static int add_seg(neuronte_gpu *g, int member, int tile, int net, int x0, int y0, int w, int h, const double *xpos,
                   const double *ypos, float slope, int fixed)
{
    struct v2 *v = g->v2;
    struct v2plan *pl = &g->plan;
    const struct vnet *vn = &v->nets[net];
    size_t hid = (size_t)vn->net.h, need = up4((size_t)w * hid) + up4((size_t)h * hid) + 4, off;
    int32_t *rec;
    int i;

    if (fixed ? pl->nparams + need > pl->params_cap || pl->nsegs >= pl->segs_cap
              : grow_params(pl, need) || grow_segs(g))
        return fixed ? -2 : -1;
    rec = pl->segs + (size_t)pl->nsegs * S_INTS;
    off = pl->nparams;
    rec[S_PARAM] = vn->param;
    rec[S_MASKS] = vn->masks;
    rec[S_X0] = x0;
    rec[S_Y0] = y0;
    rec[S_W] = w;
    rec[S_H] = h;
    rec[S_HID] = vn->net.h;
    rec[S_XTAB] = (int32_t)off;
    for (i = 0; i < w; i++)
        axis_part(&vn->net, 0, xpos[i], pl->params + off + (size_t)i * hid);
    off += up4((size_t)w * hid);
    rec[S_YTAB] = (int32_t)off;
    for (i = 0; i < h; i++)
        axis_part(&vn->net, 1, ypos[i], pl->params + off + (size_t)i * hid);
    off += up4((size_t)h * hid);
    rec[S_SLOPE] = (int32_t)off;
    pl->params[off] = slope;
    pl->nparams = off + 4;
    v->seg[pl->nsegs].member = member;
    v->seg[pl->nsegs].tile = tile;
    v->seg[pl->nsegs].kern = vn->kern;
    v->seg[pl->nsegs].to_samples = 0;
    v->seg[pl->nsegs].gp = 1;
    pl->nsegs++;
    return 0;
}

/* The segments of one member: on the even grid (gx, gy NULL), or at the
 * warped positions of one letter. */
static int build_member(neuronte_gpu *g, int mi, const double *gx, const double *gy, int fixed)
{
    struct v2 *v = g->v2;
    const struct vmember *m = &v->members[mi];
    int sw = v->sw, sh = v->sh, s = g->samples, i, rc = 0;
    double *xpos = v->gx + sw, *ypos = v->gy + sh; /* scratch after the caller's positions */

    if (m->kind == M_PLAIN) {
        const struct net *n = &v->nets[m->net].net;
        if (!gx) {
            for (i = 0; i < sw; i++)
                xpos[i] = (i + 0.5) / sw;
            for (i = 0; i < sh; i++)
                ypos[i] = (i + 0.5) / sh * 2.0;
        }
        return add_seg(g, mi, -1, m->net, 0, 0, sw, sh, gx ? gx : xpos, gx ? gy : ypos,
                       n->tau * (float)g->cell_w * (float)s, fixed);
    } else {
        int cx[MAX_TILE_AXIS + 1], ry[MAX_TILE_AXIS + 1], k, r, c;
        if (gx) {
            tile_axis_at(m->xe, m->cols, sw, 1.0, gx, cx, xpos);
            tile_axis_at(m->ye, m->rows, sh, 2.0, gy, ry, ypos);
        } else {
            tile_axis(m->xe, m->cols, sw, 1.0, cx, xpos);
            tile_axis(m->ye, m->rows, sh, 2.0, ry, ypos);
        }
        for (k = 0; k < m->cols * m->rows && !rc; k++) {
            const struct net *n = &v->nets[m->tile[k].net].net;
            float slope;
            r = k / m->cols;
            c = k % m->cols;
            if (cx[c + 1] == cx[c] || ry[r + 1] == ry[r])
                continue;
            /* the ramp is one sample of the whole cell wide; a tile's tau is for its own width */
            slope = (float)((m->xe[c + 1] - m->xe[c]) * (double)g->cell_w);
            rc = add_seg(g, mi, k, (int)m->tile[k].net, cx[c], ry[r], cx[c + 1] - cx[c], ry[r + 1] - ry[r],
                         xpos + cx[c], ypos + ry[r], n->tau * slope * (float)s, fixed);
        }
        return rc;
    }
}

static int kern_of(struct v2plan *pl, int wide, int n)
{
    struct v2kern *grown;
    int i;
    for (i = 0; i < pl->nkern; i++)
        if (pl->kern[i].wide == wide && pl->kern[i].n == n)
            return i;
    grown = realloc(pl->kern, ((size_t)pl->nkern + 1) * sizeof *grown);
    if (!grown)
        return -1;
    pl->kern = grown;
    memset(&grown[pl->nkern], 0, sizeof *grown);
    grown[pl->nkern].wide = wide;
    grown[pl->nkern].n = n;
    return pl->nkern++;
}

/* Weights of every net, the static segments, and what each kernel needs. */
static int pack(neuronte_gpu *g, char *why, size_t wl)
{
    struct v2 *v = g->v2;
    struct v2plan *pl = &g->plan;
    int i, s = g->samples;
    size_t j, o;

    pl->masks = malloc((size_t)(v->nnets ? v->nnets : 1) * NMASK * sizeof *pl->masks);
    v->gx = malloc(2 * (size_t)v->sw * sizeof *v->gx);
    v->gy = malloc(2 * (size_t)v->sh * sizeof *v->gy);
    if (!pl->masks || !v->gx || !v->gy)
        goto nomem;
    for (i = 0; i < v->nnets; i++) {
        struct vnet *vn = &v->nets[i];
        const struct net *n = &vn->net;
        size_t h = (size_t)n->h, off = pl->nparams, hp, need;
        float *p;
        int wide = n->h > NARROW_MAX, r = (n->h + LANES - 1) / LANES;
        vn->kern = kern_of(pl, wide, wide ? r : n->h);
        if (vn->kern < 0)
            goto nomem;
        hp = wide ? (size_t)r * LANES : h;
        need = up4((CODE + 1) * h + 2 * h * hp + 2 * hp + h + 1);
        if (grow_params(pl, need))
            goto nomem;
        p = pl->params + off;
        vn->param = (int32_t)off;
        vn->masks = i * NMASK;
        memcpy(pl->masks + (size_t)i * NMASK, n->masks, sizeof n->masks);
        /* code columns of layer 1, [input][output]; then b1 */
        for (j = 0; j < CODE; j++)
            for (o = 0; o < h; o++)
                p[j * h + o] = n->w1[o * IN + POS + j];
        memcpy(p + (size_t)CODE * h, n->b1, h * sizeof *p);
        p += (CODE + 1) * h;
        if (!wide) {
            /* w2, b2, w3, b3, w4 in file order ([out][in]), then b4 */
            memcpy(p, n->w2, (2 * h * h + 3 * h) * sizeof *p);
            p[2 * h * h + 3 * h] = n->b4;
        } else {
            /* layers 2 and 3 by input: row i holds the weights from input i to every output, padded
             * to a whole number of outputs per lane; biases padded alike; then w4 and b4 */
            for (j = 0; j < h; j++)
                for (o = 0; o < h; o++) {
                    p[j * hp + o] = n->w2[o * h + j];
                    p[h * hp + hp + j * hp + o] = n->w3[o * h + j];
                }
            memcpy(p + h * hp, n->b2, h * sizeof *p);
            memcpy(p + 2 * h * hp + hp, n->b3, h * sizeof *p);
            memcpy(p + 2 * h * hp + 2 * hp, n->w4, h * sizeof *p);
            p[2 * h * hp + 2 * hp + h] = n->b4;
        }
        pl->nparams = off + need;
    }
    for (i = 0; i < v->nmembers; i++) {
        struct vmember *m = &v->members[i];
        int k, any = 0;
        m->seg0 = pl->nsegs;
        if (m->kind != M_OTHER && build_member(g, i, NULL, NULL, 0))
            goto nomem;
        m->nseg = pl->nsegs - m->seg0;
        for (k = m->seg0; k < pl->nsegs; k++) {
            const int32_t *rec = pl->segs + (size_t)k * S_INTS;
            if (s > 1 && (pl->kern[v->seg[k].kern].wide || rec[S_X0] % s || rec[S_Y0] % s || rec[S_W] % s ||
                          rec[S_H] % s))
                any = 1;
            m->dyn_floats += up4((size_t)v->sw * rec[S_HID]) + up4((size_t)v->sh * rec[S_HID]) + 4;
        }
        m->reduce = any;
        for (k = m->seg0; k < pl->nsegs; k++)
            v->seg[k].to_samples = any;
        if (any)
            pl->need_samples = 1;
    }
    pl->nstatic = pl->nparams;
    pl->nstatic_segs = pl->nsegs;
    /* Room for letters warped by the letter edge table, when this cell size snaps. */
    if (v->nletters && g->cell_w < v->snap_file) {
        size_t worst = 0;
        int per = 1, n = g->max_batch < DYNAMIC_GLYPHS ? g->max_batch : DYNAMIC_GLYPHS;
        int32_t *a;
        struct vseg *b;
        for (i = 0; i < v->nmembers; i++) {
            if (v->members[i].dyn_floats > worst)
                worst = v->members[i].dyn_floats;
            if (v->members[i].kind == M_TILED && v->members[i].cols * v->members[i].rows > per)
                per = v->members[i].cols * v->members[i].rows;
        }
        v->dyn_floats = worst * (size_t)n;
        if (pl->nstatic + v->dyn_floats > (size_t)INT32_MAX) {
            neuronte_gpu_say(why, wl, "file too large for the GPU at this cell size");
            return -1;
        }
        if (grow_params(pl, v->dyn_floats))
            goto nomem;
        pl->params_cap = pl->nstatic + v->dyn_floats;
        pl->segs_cap = pl->nstatic_segs + n * per;
        a = realloc(pl->segs, (size_t)pl->segs_cap * S_INTS * sizeof *a);
        if (!a)
            goto nomem;
        pl->segs = a;
        b = realloc(v->seg, (size_t)pl->segs_cap * sizeof *b);
        if (!b)
            goto nomem;
        v->seg = b;
        if (s > 1)
            pl->need_samples = 1;
    } else {
        pl->params_cap = pl->nstatic; /* nothing is appended after open */
        pl->segs_cap = pl->nstatic_segs;
    }
    pl->nmasks = v->nnets * NMASK;
    return 0;
nomem:
    neuronte_gpu_say(why, wl, "out of memory");
    return -1;
}

/* A backend's block sizes: threads per block of the narrow kernels and of the wide ones, and the
 * samples a block of a wide kernel takes per pass. Sets how many pairs a block of each segment
 * takes and the shared memory a block of each kernel needs. cuda: the CUDA kernels' use of shared
 * memory (layers 2 to 4 or the hidden vectors in flight, besides the character parts); otherwise
 * only the character parts are shared. */
int neuronte_gpu_v2_size(neuronte_gpu *g, int narrow_threads, int wide_threads, int wide_slots, int cuda,
                         size_t shared_max, char *why, size_t wl)
{
    struct v2 *v = g->v2;
    struct v2plan *pl = &g->plan;
    int i, s = g->samples;

    if (narrow_threads < s * s)
        narrow_threads = s * s;
    for (i = 0; i < pl->nkern; i++) {
        struct v2kern *k = &pl->kern[i];
        size_t h, fixed, room;
        int gpmax = 1, sgi, hmax = 0;
        k->threads = k->wide ? wide_threads : narrow_threads;
        k->slots = k->wide ? wide_slots : k->threads;
        for (sgi = 0; sgi < pl->nstatic_segs; sgi++)
            if (v->seg[sgi].kern == i) {
                const int32_t *rec = pl->segs + (size_t)sgi * S_INTS;
                int gp = seg_gp(g, k, &v->seg[sgi], rec);
                if (gp > gpmax)
                    gpmax = gp;
            }
        for (sgi = 0; sgi < v->nnets; sgi++)
            if (v->nets[sgi].kern == i && v->nets[sgi].net.h > hmax)
                hmax = v->nets[sgi].net.h;
        h = k->wide ? (size_t)k->n * LANES : (size_t)k->n;
        k->hmax = hmax;
        /* CUDA narrow: layers 2 to 4, the character parts, a sample value per thread.
         * CUDA wide: the character parts, then two hidden vectors per sample in flight. */
        fixed = !cuda ? 0 : k->wide ? (size_t)k->slots * 2 * h : up4(2 * h * h + 3 * h + 1) + (size_t)k->threads;
        room = shared_max > fixed ? (shared_max - fixed) / (size_t)hmax : 0;
        if (room < 1) {
            neuronte_gpu_say(why, wl, "hidden width %d needs more shared memory per block than the device has",
                             hmax);
            return -1;
        }
        if ((size_t)gpmax > room)
            gpmax = (int)room;
        k->shared_floats = (int)(fixed + up4((size_t)gpmax * (size_t)hmax));
        for (sgi = 0; sgi < pl->nstatic_segs; sgi++)
            if (v->seg[sgi].kern == i) {
                int gp = seg_gp(g, k, &v->seg[sgi], pl->segs + (size_t)sgi * S_INTS);
                v->seg[sgi].gp = gp > gpmax ? gpmax : gp;
            }
        k->gp_max = gpmax;
    }
    return 0;
}

/* ---- planning a call ---- */

static int push_triple(struct v2 *v, int seg, uint32_t glyph, uint32_t code)
{
    if (v->ntriples == v->triples_cap) {
        int cap = v->triples_cap ? v->triples_cap * 2 : 1024;
        struct triple *grown = realloc(v->triples, (size_t)cap * sizeof *grown);
        if (!grown)
            return -1;
        v->triples = grown;
        v->triples_cap = cap;
    }
    v->triples[v->ntriples].seg = seg;
    v->triples[v->ntriples].glyph = glyph;
    v->triples[v->ntriples].code = code;
    v->ntriples++;
    return 0;
}

static int push_block(struct v2plan *pl, int seg, int pair0, int npairs, int lo, int hi, int mode)
{
    int32_t *b;
    if (pl->nblocks == pl->blocks_cap) {
        int cap = pl->blocks_cap ? pl->blocks_cap * 2 : 1024;
        int32_t *grown = realloc(pl->blocks, (size_t)cap * B_INTS * sizeof *grown);
        if (!grown)
            return -1;
        pl->blocks = grown;
        pl->blocks_cap = cap;
    }
    b = pl->blocks + (size_t)pl->nblocks * B_INTS;
    b[B_SEG] = seg;
    b[B_PAIR0] = pair0;
    b[B_NPAIRS] = npairs;
    b[B_LO] = lo;
    b[B_HI] = hi;
    b[B_MODE] = mode;
    pl->nblocks++;
    return 0;
}

/* Items of one pair of a segment, and the mode its blocks run in when a launch is full. */
static int seg_items(const neuronte_gpu *g, const struct v2kern *k, const struct vseg *sg, const int32_t *rec,
                     int *mode)
{
    int s = g->samples;
    if (k->wide) {
        *mode = s == 1 ? BM_DIRECT : BM_SAMPLE;
        return rec[S_W] * rec[S_H];
    }
    if (s > 1 && sg->to_samples) {
        *mode = BM_SAMPLE;
        return rec[S_W] * rec[S_H];
    }
    *mode = BM_PIXEL;
    return (rec[S_W] / s) * (rec[S_H] / s);
}

/* Plan the scalars from `first` on. Returns how many it took (fewer than
 * asked when the room for warped letters ran out), or -1. */
static int plan(neuronte_gpu *g, const uint32_t *scalars, int count, struct neuronte_gpu_timing *t)
{
    struct v2 *v = g->v2;
    struct v2plan *pl = &g->plan;
    int i, k, taken = count, s = g->samples, ss = s * s;
    const int snap = v->nletters && v->snap_below > 0 && g->cell_w < v->snap_below;

    pl->nsegs = pl->nstatic_segs;
    pl->nparams = pl->nstatic;
    pl->npairs = pl->nblocks = pl->nreduce = pl->dynamic = 0;
    v->ntriples = v->ncpu = 0;
    for (i = 0; i < count; i++) {
        uint32_t sc = normalize(scalars[i]);
        const struct vmember *m;
        const struct letter *rec;
        int mi, seg0, nseg, reduce, j;
        if (v->untrained && neuronte_untrained(v->cpu, sc) == 1) {
            v->cpu_list[v->ncpu++] = i;
            t->marks++;
            continue;
        }
        mi = neuronte_route(v->cpu, sc);
        if (mi < 0 || mi >= v->nmembers)
            return -1;
        m = &v->members[mi];
        if (m->kind == M_OTHER) {
            v->cpu_list[v->ncpu++] = i;
            t->cpu_glyphs++;
            continue;
        }
        rec = snap ? find_letter(v, sc) : NULL;
        if (rec) {
            int save_segs = pl->nsegs, rc;
            size_t save_params = pl->nparams;
            rc = axis_samples(rec, 0, g->cell_w, s, g->cell_w, g->cell_w, v->gx) ||
                 axis_samples(rec, 1, g->cell_h, s, (double)g->cell_h / 2.0, g->cell_h, v->gy);
            if (rc)
                return -1;
            rc = build_member(g, mi, v->gx, v->gy, 1);
            if (rc == -1)
                return -1;
            if (rc == -2) {
                pl->nsegs = save_segs;
                pl->nparams = save_params;
                if (save_segs == pl->nstatic_segs) { /* does not fit even alone */
                    v->cpu_list[v->ncpu++] = i;
                    t->cpu_glyphs++;
                    continue;
                }
                taken = i; /* the rest goes in the next launch */
                break;
            }
            seg0 = save_segs;
            nseg = pl->nsegs - save_segs;
            reduce = s > 1;
            for (j = seg0; j < seg0 + nseg; j++) {
                const struct v2kern *kn = &pl->kern[v->seg[j].kern];
                int gp = seg_gp(g, kn, &v->seg[j], pl->segs + (size_t)j * S_INTS);
                v->seg[j].to_samples = reduce;
                v->seg[j].gp = gp > kn->gp_max ? kn->gp_max : gp;
            }
            pl->dynamic = 1;
            t->warped++;
        } else {
            seg0 = m->seg0;
            nseg = m->nseg;
            reduce = m->reduce;
        }
        for (j = seg0; j < seg0 + nseg; j++) {
            const struct vseg *sg = &v->seg[j];
            if (push_triple(v, j, (uint32_t)i, sg->tile < 0 ? sc : tile_code(m, &m->tile[sg->tile], sc)))
                return -1;
        }
        if (reduce)
            pl->reduce[pl->nreduce++] = (uint32_t)i;
    }
    /* pairs, grouped by segment in the order the scalars came */
    if (pl->nsegs > 0) {
        memset(v->count, 0, (size_t)pl->nsegs * sizeof *v->count);
        for (i = 0; i < v->ntriples; i++)
            v->count[v->triples[i].seg]++;
        for (i = 0, k = 0; i < pl->nsegs; i++) {
            v->start[i] = k;
            k += v->count[i];
            v->count[i] = 0;
        }
    }
    if (v->ntriples > pl->pairs_cap) {
        uint32_t *grown = realloc(pl->pairs, 2 * (size_t)v->ntriples * sizeof *grown);
        if (!grown)
            return -1;
        pl->pairs = grown;
        pl->pairs_cap = v->ntriples;
    }
    for (i = 0; i < v->ntriples; i++) {
        const struct triple *tr = &v->triples[i];
        int at = v->start[tr->seg] + v->count[tr->seg]++;
        pl->pairs[2 * at] = tr->glyph;
        pl->pairs[2 * at + 1] = tr->code;
    }
    pl->npairs = v->ntriples;
    /* blocks, one run per kernel */
    for (k = 0; k < pl->nkern; k++) {
        struct v2kern *kn = &pl->kern[k];
        int fill = kn->wide ? g->fill_wide : g->fill_blocks, blocks0 = 0, want = 1, sgi;
        for (sgi = 0; sgi < pl->nsegs; sgi++)
            if (v->seg[sgi].kern == k && v->count[sgi])
                blocks0 += (v->count[sgi] + v->seg[sgi].gp - 1) / v->seg[sgi].gp;
        if (blocks0 > 0 && blocks0 < fill)
            want = (fill + blocks0 - 1) / blocks0;
        kn->block0 = pl->nblocks;
        kn->nblocks = 0;
        for (sgi = 0; sgi < pl->nsegs; sgi++) {
            const struct vseg *sg = &v->seg[sgi];
            const int32_t *rec = pl->segs + (size_t)sgi * S_INTS;
            int mode, ipg, a, n = v->count[sgi];
            if (sg->kern != k || !n)
                continue;
            ipg = seg_items(g, kn, sg, rec, &mode);
            for (a = 0; a < n; a += sg->gp) {
                int np = n - a < sg->gp ? n - a : sg->gp, items = np * ipg, nz = want, cap, per, lo, m2 = mode;
                /* a block keeps at least 32 samples (narrow) or one pass (wide) */
                if (kn->wide)
                    cap = (items + kn->slots - 1) / kn->slots;
                else
                    cap = (items * (mode == BM_PIXEL ? ss : 1) + 31) / 32;
                if (nz > cap)
                    nz = cap;
                if (nz < 1)
                    nz = 1;
                /* a small launch of a wider narrow net spreads a pixel's samples over threads */
                if (mode == BM_PIXEL && nz > 1 && kn->n >= 16)
                    m2 = BM_PIXEL_SPLIT;
                per = (items + nz - 1) / nz;
                for (lo = 0; lo < items; lo += per) {
                    if (push_block(pl, sgi, v->start[sgi] + a, np, lo, lo + per < items ? lo + per : items, m2))
                        return -1;
                    kn->nblocks++;
                }
            }
        }
    }
    return taken;
}

/* ---- public ---- */

int neuronte_gpu_v2_wanted(const char *path, int flags)
{
    unsigned char head[24];
    FILE *f;
    size_t got;
    uint32_t ver;
    if (flags & NEURONTE_GPU_FILE)
        return 1;
    f = fopen(path, "rb");
    if (!f)
        return 0;
    got = fread(head, 1, sizeof head, f);
    fclose(f);
    if (got < sizeof head || memcmp(head, "NEURONTE", 8) != 0)
        return 0;
    ver = rd32(head + 8);
    return ver != 1 || rd32(head + 20) > NARROW_MAX;
}

void neuronte_gpu_v2_close(neuronte_gpu *g)
{
    struct v2 *v = g->v2;
    struct v2plan *pl = &g->plan;
    int i;
    if (!v)
        return;
    for (i = 0; i < v->nnets; i++)
        free(v->nets[i].net.w);
    if (v->cpu)
        neuronte_close(v->cpu);
    free(v->nets);
    free(v->members);
    free(v->seg);
    free(v->letters);
    free(v->pool);
    free(v->triples);
    free(v->count);
    free(v->start);
    free(v->cpu_list);
    free(v->gx);
    free(v->gy);
    free(v->kept_key);
    free(v->kept);
    free(v->line);
    free(v->fresh);
    free(v);
    free(pl->params);
    free(pl->segs);
    free(pl->masks);
    free(pl->kern);
    free(pl->pairs);
    free(pl->blocks);
    free(pl->reduce);
    g->v2 = NULL;
}

int neuronte_gpu_v2_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *reason, size_t reason_len)
{
    struct v2 *v = calloc(1, sizeof *v);
    struct v2plan *pl = &g->plan;
    unsigned char *file = NULL;
    size_t len = 0;
    char why[512] = "", w1[512] = "", w2[512] = "";
    int want = opt->backend, rc = -1, i;

    if (!v)
        return -1;
    g->v2 = v;
    g->cell_w = opt->cell_w;
    g->cell_h = opt->cell_h;
    g->samples = opt->samples;
    snprintf(g->name, sizeof g->name, "%s", opt->nrt_path);
    if (g->cell_w <= 0 || g->cell_h <= 0 || g->cell_w > NEURONTE_MAX_CELL || g->cell_h > NEURONTE_MAX_CELL ||
        g->samples < 0 || g->samples > NEURONTE_MAX_SAMPLES) {
        neuronte_gpu_say(reason, reason_len, "invalid cell or samples");
        return -1;
    }
    if (g->samples == 0)
        g->samples = g->cell_w < 24 ? 2 : 1;
    v->sw = g->cell_w * g->samples;
    v->sh = g->cell_h * g->samples;
    file = neuronte_gpu_read_file(opt->nrt_path, &len);
    if (!file) {
        neuronte_gpu_say(reason, reason_len, "cannot read %s: %s", opt->nrt_path, strerror(errno));
        return -1;
    }
    if (neuronte_open(file, len, &v->cpu) != NEURONTE_OK || neuronte_nets(v->cpu) < 1) {
        neuronte_gpu_say(reason, reason_len, "the reference library cannot open %s, or it has no net", opt->nrt_path);
        free(file);
        return -1;
    }
    neuronte_set_samples(v->cpu, g->samples);
    if (read_image(v, file, len, reason, reason_len) || v->nmembers != neuronte_nets(v->cpu)) {
        if (reason && reason_len && !reason[0])
            neuronte_gpu_say(reason, reason_len, "cannot lay out %s", opt->nrt_path);
        free(file);
        return -1;
    }
    free(file);
    v->snap_below = v->snap_file;
    g->hidden = 0;
    for (i = 0; i < v->nnets; i++)
        if (v->nets[i].net.h > g->hidden)
            g->hidden = v->nets[i].net.h; /* the widest */
    g->nmembers = v->nmembers;
    v->cpu_list = malloc((size_t)g->max_batch * sizeof *v->cpu_list);
    pl->reduce = malloc((size_t)g->max_batch * sizeof *pl->reduce);
    if (!v->cpu_list || !pl->reduce)
        return -1;
    if (want == NEURONTE_GPU_CPU) {
        g->backend = NEURONTE_GPU_CPU;
        return 0;
    }
    if ((int64_t)v->sw * v->sh * 64 > INT_MAX)
        neuronte_gpu_say(why, sizeof why, "a %dx%d cell with %d samples per axis is too large for the GPU kernel",
                         g->cell_w, g->cell_h, g->samples);
    else if (pack(g, why, sizeof why) == 0) {
        v->count = calloc((size_t)(pl->segs_cap ? pl->segs_cap : 1), sizeof *v->count);
        v->start = calloc((size_t)(pl->segs_cap ? pl->segs_cap : 1), sizeof *v->start);
        if (!v->count || !v->start)
            return -1;
        if (want == NEURONTE_GPU_AUTO || want == NEURONTE_GPU_CUDA) {
            rc = neuronte_gpu_cuda_open2(g, opt, w1, sizeof w1);
            if (rc == 0)
                g->backend = NEURONTE_GPU_CUDA;
        }
        if (rc && (want == NEURONTE_GPU_AUTO || want == NEURONTE_GPU_VULKAN)) {
            rc = neuronte_gpu_vulkan_open2(g, opt, w2, sizeof w2);
            if (rc == 0)
                g->backend = NEURONTE_GPU_VULKAN;
        }
        if (rc && want == NEURONTE_GPU_OPENGL) {
            rc = neuronte_gpu_gl_open2(g, opt, w2, sizeof w2);
            if (rc == 0)
                g->backend = NEURONTE_GPU_OPENGL;
        }
        if (rc && want == NEURONTE_GPU_OPENCL)
            neuronte_gpu_say(w2, sizeof w2, "the OpenCL backend draws single nets and layouts only");
        if (rc == 0)
            return 0;
        neuronte_gpu_say(why, sizeof why, "%s%s%s%s", w1[0] ? "cuda: " : "", w1, w1[0] && w2[0] ? "; " : "", w2);
    }
    if (!(opt->flags & NEURONTE_GPU_ALLOW_FALLBACK)) {
        neuronte_gpu_say(reason, reason_len, "no GPU backend: %s", why);
        return -1;
    }
    g->backend = NEURONTE_GPU_CPU_FALLBACK;
    snprintf(g->reason, sizeof g->reason, "%s", why);
    return 0;
}

int neuronte_gpu_v2_coverage(neuronte_gpu *g, const uint32_t *scalars, int count, float *out,
                             struct neuronte_gpu_timing *timing)
{
    struct v2 *v = g->v2;
    struct neuronte_gpu_timing t = {0};
    size_t plane = (size_t)g->cell_w * g->cell_h;
    uint64_t t0 = neuronte_gpu_now(), c0;
    int i, j;

    if (g->backend == NEURONTE_GPU_CPU || g->backend == NEURONTE_GPU_CPU_FALLBACK) {
        for (i = 0; i < count; i++)
            if (neuronte_coverage(v->cpu, scalars[i], g->cell_w, g->cell_h, g->samples, out + (size_t)i * plane) !=
                NEURONTE_OK)
                return -1;
        t.cpu_eval = neuronte_gpu_now() - t0;
        t.cpu_glyphs = count;
    } else {
        for (i = 0; i < count;) {
            int n = count - i < g->max_batch ? count - i : g->max_batch;
            struct neuronte_gpu_timing part = {0};
            c0 = neuronte_gpu_now();
            n = plan(g, scalars + i, n, &part);
            if (n <= 0)
                return -1;
            t.plan += neuronte_gpu_now() - c0;
            if (g->plan.npairs && g->ops->run_plan(g->impl, n, out + (size_t)i * plane, &part))
                return -1;
            c0 = neuronte_gpu_now();
            for (j = 0; j < v->ncpu; j++)
                if (neuronte_coverage(v->cpu, scalars[i + v->cpu_list[j]], g->cell_w, g->cell_h, g->samples,
                                      out + (size_t)(i + v->cpu_list[j]) * plane) != NEURONTE_OK)
                    return -1;
            t.cpu_eval += neuronte_gpu_now() - c0;
            t.upload += part.upload;
            t.run += part.run;
            t.wait += part.wait;
            t.download += part.download;
            t.launches += part.launches;
            t.cpu_glyphs += part.cpu_glyphs;
            t.marks += part.marks;
            t.warped += part.warped;
            i += n;
        }
    }
    t.total = neuronte_gpu_now() - t0;
    if (timing)
        *timing = t;
    return 0;
}

int neuronte_gpu_set_untrained(neuronte_gpu *g, int mode)
{
    if (!g || !g->v2 || neuronte_set_untrained(g->v2->cpu, mode) != NEURONTE_OK)
        return -1;
    g->v2->untrained = mode;
    g->v2->kept_n = 0; /* kept glyphs were drawn under the old setting */
    if (g->v2->kept_key)
        memset(g->v2->kept_key, 0, (size_t)g->v2->kept_cap * sizeof *g->v2->kept_key);
    return 0;
}

int neuronte_gpu_untrained(const neuronte_gpu *g, uint32_t scalar)
{
    return g && g->v2 ? neuronte_untrained(g->v2->cpu, scalar) : -1;
}

int neuronte_gpu_route(const neuronte_gpu *g, uint32_t scalar)
{
    return g && g->v2 ? neuronte_route(g->v2->cpu, scalar) : -1;
}

int neuronte_gpu_set_snap(neuronte_gpu *g, int below_px)
{
    if (!g || !g->v2 || neuronte_set_snap(g->v2->cpu, below_px) != NEURONTE_OK)
        return -1;
    /* the library keeps one threshold for colour glyphs and letters; letters snap only with a table */
    g->v2->snap_below = below_px < 0 ? g->v2->snap_file : below_px;
    g->v2->kept_n = 0;
    if (g->v2->kept_key)
        memset(g->v2->kept_key, 0, (size_t)g->v2->kept_cap * sizeof *g->v2->kept_key);
    return 0;
}

const neuronte *neuronte_gpu_library(const neuronte_gpu *g)
{
    return g && g->v2 ? g->v2->cpu : NULL;
}

/* ---- a line of text ---- */

int neuronte_gpu_set_binary(neuronte_gpu *g, int on)
{
    if (!g || !g->v2 || neuronte_set_binary(g->v2->cpu, on) != NEURONTE_OK)
        return -1;
    g->v2->binary = on != 0;
    return 0;
}

static int kept_slot(const struct v2 *v, uint32_t s)
{
    uint32_t i = (s * 2654435761u) & (uint32_t)(v->kept_cap - 1);
    while (v->kept_key[i] && v->kept_key[i] != s + 1)
        i = (i + 1) & (uint32_t)(v->kept_cap - 1);
    return (int)i;
}

/* Room for `more` scalars in the table, which stays under half full. */
static int kept_room(neuronte_gpu *g, int more)
{
    struct v2 *v = g->v2;
    size_t plane = (size_t)g->cell_w * g->cell_h;
    int cap = v->kept_cap ? v->kept_cap : 256, i;
    uint32_t *keys, *old_keys = v->kept_key;
    float *cov, *old = v->kept;
    int old_cap = v->kept_cap;

    while (cap < 2 * (v->kept_n + more))
        cap *= 2;
    if (cap == v->kept_cap)
        return 0;
    keys = calloc((size_t)cap, sizeof *keys);
    cov = malloc((size_t)cap * plane * sizeof *cov);
    if (!keys || !cov) {
        free(keys);
        free(cov);
        return -1;
    }
    v->kept_key = keys;
    v->kept = cov;
    v->kept_cap = cap;
    for (i = 0; i < old_cap; i++)
        if (old_keys[i]) {
            int at = kept_slot(v, old_keys[i] - 1);
            keys[at] = old_keys[i];
            memcpy(cov + (size_t)at * plane, old + (size_t)i * plane, plane * sizeof *cov);
        }
    free(old_keys);
    free(old);
    return 0;
}

/* The library's blend and stroked, operation for operation. */
static uint32_t blend(uint32_t dst, uint32_t color, float a)
{
    float da = (float)(dst >> 24) / 255.0f, oa = a + da * (1.0f - a), ch[3];
    int i;
    if (oa <= 0.0f)
        return 0;
    for (i = 0; i < 3; i++) {
        int sh = 16 - 8 * i;
        float sc = (float)((color >> sh) & 0xFF), dc = (float)((dst >> sh) & 0xFF);
        ch[i] = (sc * a + dc * da * (1.0f - a)) / oa;
    }
    return (uint32_t)(oa * 255.0f + 0.5f) << 24 | (uint32_t)(ch[0] + 0.5f) << 16 | (uint32_t)(ch[1] + 0.5f) << 8 |
           (uint32_t)(ch[2] + 0.5f);
}

static float stroked(const float *row, int width, int x, int stroke)
{
    int lo = x > stroke ? x - stroke : 0;
    int hi = x < width ? x : width - 1;
    float c = 0.0f;
    int i;
    for (i = lo; i <= hi; i++)
        if (row[i] > c)
            c = row[i];
    return c;
}

int neuronte_gpu_draw_text(neuronte_gpu *g, uint32_t *pixels, int stride, int width, int height, int x, int y,
                           int bold, int stroke_px, const char *text, size_t len, uint32_t color,
                           struct neuronte_gpu_timing *timing)
{
    struct v2 *v;
    struct neuronte_gpu_timing t = {0};
    float alpha = (float)(color >> 24) / 255.0f;
    size_t plane, off = 0;
    int cw, ch, total, stroke, n = 0, nfresh = 0, i, px, py;
    int64_t pen;
    uint64_t t0 = neuronte_gpu_now();

    if (timing)
        *timing = t;
    if (!g || !g->v2 || !pixels || width < 0 || height < 0 || stride < width || g->cell_h != 2 * g->cell_w ||
        stroke_px < -1 || stroke_px > g->cell_w || (bold != 0 && bold != 1))
        return NEURONTE_ERR_ARG;
    v = g->v2;
    cw = g->cell_w;
    ch = g->cell_h;
    plane = (size_t)cw * ch;
    total = stroke_px < 0 ? neuronte_text_width(v->cpu, text, len, bold, cw)
                          : neuronte_text_width_stroke(v->cpu, text, len, stroke_px, cw);
    if (total < 0 || alpha <= 0.0f)
        return total;
    stroke = stroke_px < 0 ? bold : stroke_px;
    if ((size_t)v->line_cap < len + 1) {
        uint32_t *a = realloc(v->line, (len + 1) * sizeof *a), *b;
        if (!a)
            return NEURONTE_ERR_NOMEM;
        v->line = a;
        b = realloc(v->fresh, (len + 1) * sizeof *b);
        if (!b)
            return NEURONTE_ERR_NOMEM;
        v->fresh = b;
        v->line_cap = (int)len + 1;
    }
    /* The scalars drawn, and which of the visible ones are not kept yet. */
    if (kept_room(g, (int)len + 1))
        return NEURONTE_ERR_NOMEM;
    pen = x;
    while (off < len) {
        uint32_t s, as;
        int64_t adv;
        int ix, at;
        off += neuronte_utf8_decode(text + off, len - off, &s);
        as = neuronte_advance_scalar(v->cpu, s);
        v->line[n++] = as;
        ix = pen > INT_MAX ? INT_MAX : (int)pen;
        if (pen <= INT_MAX && (int64_t)ix < width && (int64_t)y < height && pen + (int64_t)cw + stroke > 0 &&
            (int64_t)y + ch > 0) {
            int mi = neuronte_route(v->cpu, as);
            if (mi < 0 || mi >= v->nmembers)
                return NEURONTE_ERR_ARG;
            /* as the library: text is not drawn from a colour net */
            if (v->members[mi].kind == M_OTHER && !(v->untrained && neuronte_untrained(v->cpu, as) == 1))
                return NEURONTE_ERR_ARG;
            at = kept_slot(v, as);
            if (!v->kept_key[at]) {
                v->kept_key[at] = as + 1;
                v->kept_n++;
                v->fresh[nfresh++] = as;
            }
        }
        adv = stroke_px < 0 ? neuronte_advance(v->cpu, s, bold, cw) : (int64_t)neuronte_advance(v->cpu, s, 0, cw) + stroke_px;
        if (adv > 0 && pen > INT64_MAX - adv)
            pen = INT64_MAX;
        else
            pen += adv;
    }
    if (nfresh) {
        /* one launch for the line's new letters, then each to its place in the table */
        float *cov = malloc((size_t)nfresh * plane * sizeof *cov);
        int rc = cov ? neuronte_gpu_v2_coverage(g, v->fresh, nfresh, cov, &t) : -1;
        for (i = 0; i < nfresh; i++) {
            int at = kept_slot(v, v->fresh[i]);
            if (rc) { /* forget what was not drawn */
                v->kept_key[at] = 0;
                v->kept_n--;
            } else {
                memcpy(v->kept + (size_t)at * plane, cov + (size_t)i * plane, plane * sizeof *cov);
            }
        }
        free(cov);
        if (rc) {
            /* removing keys can break probe chains: start the table again */
            memset(v->kept_key, 0, (size_t)v->kept_cap * sizeof *v->kept_key);
            v->kept_n = 0;
            return NEURONTE_ERR_NOMEM;
        }
    }
    pen = x;
    for (i = 0, off = 0; off < len; i++) {
        uint32_t s;
        int64_t adv;
        int ix;
        off += neuronte_utf8_decode(text + off, len - off, &s);
        ix = pen > INT_MAX ? INT_MAX : (int)pen;
        if (pen <= INT_MAX && (int64_t)ix < width && (int64_t)y < height && pen + (int64_t)cw + stroke > 0 &&
            (int64_t)y + ch > 0) {
            const float *cov = v->kept + (size_t)kept_slot(v, v->line[i]) * plane;
            for (py = y < 0 ? (int)(-(int64_t)y) : 0; py < ch && (int64_t)y + py < height; py++)
                for (px = ix < 0 ? (int)(-(int64_t)ix) : 0; px < cw + stroke && (int64_t)ix + px < width; px++) {
                    const float *row = cov + (size_t)py * cw;
                    float c = stroked(row, cw, px, stroke);
                    if (v->binary)
                        c = c > 0.5f ? 1.0f : 0.0f;
                    if (c * alpha > 0.0f) {
                        uint32_t *d = pixels + (size_t)((int64_t)y + py) * (size_t)stride + (size_t)((int64_t)ix + px);
                        *d = blend(*d, color, c * alpha);
                    }
                }
        }
        adv = stroke_px < 0 ? neuronte_advance(v->cpu, s, bold, cw) : (int64_t)neuronte_advance(v->cpu, s, 0, cw) + stroke_px;
        if (adv > 0 && pen > INT64_MAX - adv)
            pen = INT64_MAX;
        else
            pen += adv;
    }
    t.total = neuronte_gpu_now() - t0;
    if (timing)
        *timing = t;
    return total;
}
