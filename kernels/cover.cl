/* neuronte-gpu OpenCL kernels (OpenCL C 1.2): the algorithm of cover.comp, arranged for GPUs with
 * few registers per work-item and 4-wide vector arithmetic, such as Arm Mali.
 *
 * Both kernels share the layout of the Vulkan one. One work-group evaluates one tile (group x) for
 * a group of `gpb` glyphs (group y); groups along z share out the pixels of a launch too small to
 * fill the GPU. The character part of layer 1 of the group's glyphs is computed into local memory
 * first. The host chooses the kernel by hidden width and device (backend_opencl.c).
 *
 * cover, the per-sample kernel: each work-item takes one pixel at a time and evaluates its S x S
 * samples in order or, in small launches with a hidden width of 16 or more (`per_sample`), one
 * sample at a time, the samples of a pixel meeting in local memory to be added in order by one
 * work-item. A sample keeps only H values live per work-item. Layer 2 is evaluated one input at a
 * time: each layer-1 activation is formed when it is needed (two additions and a ReLU, from the
 * character part and the x and y tables) and added into all H layer-2 sums, which is also the
 * order the reference library uses. Layer 3 is evaluated four outputs at a time from the H layer-2
 * activations, and each output is added into layer 4 as soon as it is known.
 *
 * cover_tile, the tile kernel: the work-group evaluates NS samples at a time, layer by layer,
 * through local memory. Layers 2 and 3 are matrix products in which each work-item computes
 * blocks of 4 outputs x 4 samples, so one weight load and one activation load feed four vector
 * multiply-adds. It does more arithmetic per load than the per-sample kernel and keeps fewer
 * registers live, which pays at large hidden widths.
 *
 * Weights are stored [input][output] with rows padded to a multiple of four, so four outputs'
 * weights are one 16-byte load and the arithmetic is on 4-vectors, which Midgard executes natively.
 *
 * Every sum adds its terms in the reference library's order. (The library skips terms whose input
 * is zero; adding them can change only the sign of a zero sum, which the ReLU discards.) With
 * EXACT, every multiply and add is rounded separately (FP_CONTRACT OFF), so the coverage is the
 * library's bit for bit wherever the device's single-precision add and multiply are correctly
 * rounded and keep denormals.
 *
 * Compile-time definitions: H (hidden width), S (samples per axis), EXACT (0 or 1), HOST_DIVIDE
 * (1 when S*S is not a power of two: the kernel writes the sums and the host divides them, since
 * the device's division need not be correctly rounded and 1/(S*S) is then not exact); optionally
 * L3B (per-sample kernel: layer-3 output blocks per pass, default 1), NS (tile kernel: samples per
 * pass, default 64) and TO (tile kernel: output blocks per work-item, 1 or 2, default 1).
 */
#ifndef H
#error H (the hidden width) must be defined
#endif
#ifndef S
#error S (samples per axis) must be defined
#endif
#ifndef EXACT
#define EXACT 1
#endif
#ifndef HOST_DIVIDE
#define HOST_DIVIDE 0
#endif

#if EXACT
#pragma OPENCL FP_CONTRACT OFF
#define MAC(acc, a, b) ((acc) + (a) * (b))
#else
#define MAC(acc, a, b) mad((a), (b), (acc))
#endif
#define MAC4(acc, a, b) MAC((acc), (float4)(a), (b))

#define CODE 64
#define NBITS 21
#define BRAILLE_FIRST 0x2800u
#define NQ ((H + 3) / 4) /* float4s per row of H values */
#define SS (S * S)
#define SAMPLE_ITEMS_H 16 /* below this width a pixel's chain is short: pixels per work-item only */

/* Per-tile record in `tiles`, ints; offsets are in float4s. Must match backend_opencl.c. */
#define T_X 0
#define T_Y 1
#define T_W 2
#define T_H 3
#define T_MASK 4
#define T_PARAM 5
#define T_XTAB 6
#define T_YTAB 7
#define T_MASKS 8
#define T_INTS 9

/* A tile's weight block, in float4s: code columns [bit][output], b1, w2 [input][output], b2,
 * w3 [input][output], b3, w4, then (b4, slope, 0, 0). Rows are NQ float4s, zero-padded. */
#define O_B1 (CODE * NQ)
#define O_W2 (O_B1 + NQ)
#define O_B2 (O_W2 + H * NQ)
#define O_W3 (O_B2 + NQ)
#define O_B3 (O_W3 + H * NQ)
#define O_W4 (O_B3 + NQ)
#define O_END (O_W4 + NQ)

#define ZERO4 ((float4)(0.0f))
/* x > 0 ? x : 0 per component (0 for NaN), as the library's ReLU */
#define RELU4(x) select(ZERO4, (x), isgreater((x), ZERO4))
/* component j (a constant once the loops are unrolled) */
#define COMP(v, j) ((j) == 0 ? (v).x : (j) == 1 ? (v).y : (j) == 2 ? (v).z : (v).w)

#if HOST_DIVIDE
#define FINISH(acc) (acc)
#else
/* 1/(S*S) is exact when S*S is a power of two, so this is the library's acc / (S*S) */
#define FINISH(acc) ((acc) * (1.0f / (float)SS))
#endif

#define NQF (H / 4) /* full blocks of four inputs */
#define HT (H % 4)  /* inputs in the last, partial block */
/* Layer-3 output blocks per pass: independent accumulator chains, so that a work-item's
   multiply-adds do not wait on each other. Must divide NQ (the host chooses it). */
#ifndef L3B
#define L3B 1
#endif
#if NQ % L3B
#error L3B must divide NQ
#endif

/* layer 2: input value aj, with weight row wrow ([output] for that input), into all H sums */
#define L2_INPUT(aj, wrow)                                                                         \
    do {                                                                                           \
        const float a_ = (aj);                                                                     \
        __global const float4 *restrict r_ = (wrow);                                               \
        _Pragma("unroll") for (int k = 0; k < NQ; k++) s[k] = MAC4(s[k], a_, r_[k]);               \
    } while (0)

/* One sample's clamped value: layer 1 from the character part c and the axis rows xr, yr, then
 * layers 2 to 4 with the tile's weights w. */
static float sample_value(__global const float4 *restrict w, __local const float4 *restrict c,
                          __global const float4 *restrict xr, __global const float4 *restrict yr)
{
    float4 s[NQ];
#pragma unroll
    for (int k = 0; k < NQ; k++)
        s[k] = w[O_B2 + k];
    /* layer 2, one input at a time, into all H sums */
    for (int q4 = 0; q4 < NQF; q4++) {
        const float4 a = RELU4((c[q4] + xr[q4]) + yr[q4]);
        __global const float4 *restrict wq = w + O_W2 + 4 * q4 * NQ;
        L2_INPUT(a.x, wq);
        L2_INPUT(a.y, wq + NQ);
        L2_INPUT(a.z, wq + 2 * NQ);
        L2_INPUT(a.w, wq + 3 * NQ);
    }
#if HT
    {
        const float4 a = RELU4((c[NQF] + xr[NQF]) + yr[NQF]);
        __global const float4 *restrict wq = w + O_W2 + 4 * NQF * NQ;
        L2_INPUT(a.x, wq);
#if HT > 1
        L2_INPUT(a.y, wq + NQ);
#endif
#if HT > 2
        L2_INPUT(a.z, wq + 2 * NQ);
#endif
    }
#endif
#pragma unroll
    for (int k = 0; k < NQ; k++)
        s[k] = RELU4(s[k]);
    /* layers 3 and 4: L3B blocks of four layer-3 outputs per pass over the H layer-2 values,
       each output added into layer 4 in order once known */
    const float4 e = w[O_END];
    float f = e.x;
    for (int k0 = 0; k0 < NQ; k0 += L3B) {
        float4 acc[L3B];
        __global const float4 *restrict wk = w + O_W3 + k0;
#pragma unroll
        for (int u = 0; u < L3B; u++)
            acc[u] = w[O_B3 + k0 + u];
#pragma unroll
        for (int q = 0; q < H; q++) {
            const float bq = COMP(s[q >> 2], q & 3);
#pragma unroll
            for (int u = 0; u < L3B; u++)
                acc[u] = MAC4(acc[u], bq, wk[q * NQ + u]);
        }
#pragma unroll
        for (int u = 0; u < L3B; u++) {
            const float4 r = RELU4(acc[u]), w4 = w[O_W4 + k0 + u];
            f = MAC(f, w4.x, r.x);
#if HT == 0
            f = MAC(f, w4.y, r.y);
            f = MAC(f, w4.z, r.z);
            f = MAC(f, w4.w, r.w);
#else
            if (4 * (k0 + u) + 1 < H)
                f = MAC(f, w4.y, r.y);
            if (4 * (k0 + u) + 2 < H)
                f = MAC(f, w4.z, r.z);
            if (4 * (k0 + u) + 3 < H)
                f = MAC(f, w4.w, r.w);
#endif
        }
    }
    const float v = MAC(0.5f, e.y, f);
    return v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
}

/* The character part of layer 1 of the work-group's glyphs (from g0, at most gpb) into cs, four
   outputs per work-item and step. */
static void char_part(__global const float4 *restrict w, __global const uint *restrict masks, const int mk,
                      const int mask, __global const uint *restrict scalars, const int g0, const int nglyph,
                      const int gpb, const int tid, const int nt, __local float4 *restrict cs)
{
    for (int k = tid; k < gpb * NQ; k += nt) {
        const int gl = k / NQ, o4 = k - gl * NQ, glyph = g0 + gl;
        if (glyph >= nglyph)
            break;
        uint s = scalars[glyph];
        if (mask >= 0)
            s = BRAILLE_FIRST | ((s - BRAILLE_FIRST) & (uint)mask);
        if (s > 0x10FFFFu || (s >= 0xD800u && s <= 0xDFFFu))
            s = 0xFFFDu;
        float4 c = w[O_B1 + o4];
        for (int i = 0; i < CODE; i++) {
            const uint bit = i < NBITS ? (s >> i) & 1u : popcount(s & masks[mk + i - NBITS]) & 1u;
            const float4 wi = w[i * NQ + o4];
            c = c + (bit ? wi : -wi); /* (+-1) * w is exact */
        }
        cs[k] = c;
    }
}

/* The start of both kernels: the tile's record, the character parts, and this group's share of
   the pixels. Groups along z share out the pixels of one tile and glyph group, so a call with few
   glyphs still fills the GPU. */
#define PROLOGUE                                                                                   \
    const int tid = (int)get_local_id(0), nt = (int)get_local_size(0);                             \
    const int tb = (int)get_group_id(0) * T_INTS;                                                  \
    const int tx = tiles[tb + T_X], ty = tiles[tb + T_Y], tw = tiles[tb + T_W], th = tiles[tb + T_H]; \
    __global const float4 *restrict w = params + tiles[tb + T_PARAM];                              \
    __global const float4 *restrict xt = params + tiles[tb + T_XTAB];                              \
    __global const float4 *restrict yt = params + tiles[tb + T_YTAB];                              \
    const int g0 = (int)get_group_id(1) * gpb;                                                     \
    char_part(w, masks, tiles[tb + T_MASKS], tiles[tb + T_MASK], scalars, g0, nglyph, gpb, tid, nt, cs); \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                  \
    const int P = tw * th;                                                                         \
    const int pixels = gpb * P, nz = (int)get_num_groups(2), pchunk = (pixels + nz - 1) / nz;      \
    const int plo = (int)get_group_id(2) * pchunk;                                                 \
    const int phi = min(min(plo + pchunk, pixels), min(nglyph - g0, gpb) * P)

#define KERNEL_ARGS                                                                                \
    __global const float4 *restrict params, __global const int *restrict tiles,                   \
        __global const uint *restrict masks, __global const uint *restrict scalars,                \
        __global float *restrict outv, const int nglyph, const int cell_w, const int plane, const int gpb, \
        const int per_sample, __local float4 *restrict cs, __local float *restrict sv

/* Per-sample kernel: each work-item evaluates whole samples, keeping the H layer-2 sums in
   registers (sample_value). */
__kernel void cover(KERNEL_ARGS)
{
    PROLOGUE;
    if (H >= SAMPLE_ITEMS_H && per_sample) {
        /* One item is one sample: (glyph, pixel, sub-row, sub-column), in that order. Each pass of
           the work-group takes `per` consecutive items, a whole number of pixels; the samples'
           values meet in local memory and one work-item per pixel adds them in the library's
           order. */
        const int per = nt / SS * SS, hi = phi * SS;
        for (int base = plo * SS; base < hi; base += per) {
            const int it = base + tid;
            const int on = tid < per && it < hi;
            const int pix = it / SS, sub = it - pix * SS;
            const int gl = pix / P, pi = pix - gl * P;
            const int py = pi / tw, px = pi - py * tw;
            float v = 0.0f;
            if (on) {
                const int j = sub / S, i = sub - j * S;
                v = sample_value(w, cs + gl * NQ, xt + (px * S + i) * NQ, yt + (py * S + j) * NQ);
            }
            const int dst = (g0 + gl) * plane + (ty + py) * cell_w + tx + px;
#if SS == 1
            if (on)
                outv[dst] = FINISH(0.0f + v);
#else
            sv[tid] = v;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (on && sub == 0) {
                float acc = 0.0f;
                for (int k = 0; k < SS; k++)
                    acc = acc + sv[tid + k];
                outv[dst] = FINISH(acc);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
#endif
        }
    } else {
        for (int pix = plo + tid; pix < phi; pix += nt) {
            const int gl = pix / P, pi = pix - gl * P;
            const int py = pi / tw, px = pi - py * tw;
            float acc = 0.0f;
            for (int j = 0; j < S; j++)
                for (int i = 0; i < S; i++)
                    acc = acc + sample_value(w, cs + gl * NQ, xt + (px * S + i) * NQ, yt + (py * S + j) * NQ);
            outv[(g0 + gl) * plane + (ty + py) * cell_w + tx + px] = FINISH(acc);
        }
    }
}

/* ---- tile kernel ----
 * The work-group evaluates NS samples at a time, layer by layer, through local memory, as a
 * matrix product: each work-item computes blocks of 4 outputs x 4 samples (four 4-vectors of
 * sums, one per output, across the samples), so one 16-byte weight load and one 16-byte
 * activation load feed four vector multiply-adds and few registers stay live. Every sum is the
 * library's: the bias, then its terms in ascending input order, separately rounded. */
#ifndef NS
#define NS 64 /* samples per pass: a multiple of 4 and of S*S (the kernel is left out otherwise) */
#endif
#if NS % 4 == 0 && NS % SS == 0
#define HP (4 * NQ) /* rows of an activation buffer: H rounded up to a multiple of 4 */
#define N4 (NS / 4)

#ifndef TO
#define TO 1 /* blocks of 4 outputs per work-item and step: 1, or 2 when NQ is even */
#endif

/* One dense layer and its ReLU over NS samples: out[o][n] = relu(b[o] + sum over ascending q of
   in[q][n] * W[o][q]); wt is [input][output], NQ float4s per input. Buffers are [row][N4 float4s]. */
static void dense_tile(__local const float4 *restrict in, __local float4 *restrict out,
                       __global const float4 *restrict wt, __global const float4 *restrict b, const int tid,
                       const int nt)
{
#if TO == 2 && NQ % 2 == 0
    /* 8 outputs x 4 samples: one activation load and two weight loads feed eight multiply-adds */
    for (int p = tid; p < NQ / 2 * N4; p += nt) {
        const int k = 2 * (p / N4), n4 = p - (k / 2) * N4;
        const float4 bk = b[k], bl = b[k + 1];
        float4 a0 = (float4)(bk.x), a1 = (float4)(bk.y), a2 = (float4)(bk.z), a3 = (float4)(bk.w);
        float4 a4 = (float4)(bl.x), a5 = (float4)(bl.y), a6 = (float4)(bl.z), a7 = (float4)(bl.w);
        __local const float4 *restrict x = in + n4;
        __global const float4 *restrict wk = wt + k;
        for (int q = 0; q < H; q++) {
            const float4 v = x[q * N4], ww = wk[q * NQ], wv = wk[q * NQ + 1];
            a0 = MAC4(a0, ww.x, v);
            a1 = MAC4(a1, ww.y, v);
            a2 = MAC4(a2, ww.z, v);
            a3 = MAC4(a3, ww.w, v);
            a4 = MAC4(a4, wv.x, v);
            a5 = MAC4(a5, wv.y, v);
            a6 = MAC4(a6, wv.z, v);
            a7 = MAC4(a7, wv.w, v);
        }
        __local float4 *restrict y = out + 4 * k * N4 + n4;
        y[0] = RELU4(a0);
        y[N4] = RELU4(a1);
        y[2 * N4] = RELU4(a2);
        y[3 * N4] = RELU4(a3);
        y[4 * N4] = RELU4(a4);
        y[5 * N4] = RELU4(a5);
        y[6 * N4] = RELU4(a6);
        y[7 * N4] = RELU4(a7);
    }
#else
    for (int p = tid; p < NQ * N4; p += nt) {
        const int k = p / N4, n4 = p - k * N4;
        const float4 bk = b[k];
        float4 a0 = (float4)(bk.x), a1 = (float4)(bk.y), a2 = (float4)(bk.z), a3 = (float4)(bk.w);
        __local const float4 *restrict x = in + n4;
        __global const float4 *restrict wk = wt + k;
        for (int q = 0; q < H; q++) {
            const float4 v = x[q * N4], ww = wk[q * NQ];
            a0 = MAC4(a0, ww.x, v);
            a1 = MAC4(a1, ww.y, v);
            a2 = MAC4(a2, ww.z, v);
            a3 = MAC4(a3, ww.w, v);
        }
        __local float4 *restrict y = out + 4 * k * N4 + n4;
        y[0] = RELU4(a0);
        y[N4] = RELU4(a1);
        y[2 * N4] = RELU4(a2);
        y[3 * N4] = RELU4(a3);
    }
#endif
}

__kernel void cover_tile(KERNEL_ARGS)
{
    __local float4 act[HP * N4]; /* layer-1 activations, later layer 3's */
    __local float4 hid[HP * N4]; /* layer-2 activations */
    __local float *restrict acts = (__local float *)act;
    PROLOGUE;
    const int hi = phi * SS;
    const float4 e = w[O_END];
    __global const float *restrict w4 = (__global const float *)(w + O_W4);
    for (int base = plo * SS; base < hi; base += NS) {
        /* layer 1: one sample (glyph, pixel, sub-row, sub-column) per work-item and step */
        for (int n = tid; n < NS; n += nt) {
            const int it = base + n;
            if (it < hi) {
                const int pix = it / SS, sub = it - pix * SS;
                const int gl = pix / P, pi = pix - gl * P;
                const int py = pi / tw, px = pi - py * tw;
                const int j = sub / S, i = sub - j * S;
                __local const float4 *restrict c = cs + gl * NQ;
                __global const float4 *restrict xr = xt + (px * S + i) * NQ;
                __global const float4 *restrict yr = yt + (py * S + j) * NQ;
                for (int q4 = 0; q4 < NQ; q4++) {
                    const float4 a = RELU4((c[q4] + xr[q4]) + yr[q4]);
                    acts[(4 * q4) * NS + n] = a.x;
                    acts[(4 * q4 + 1) * NS + n] = a.y;
                    acts[(4 * q4 + 2) * NS + n] = a.z;
                    acts[(4 * q4 + 3) * NS + n] = a.w;
                }
            } else {
                for (int q = 0; q < HP; q++)
                    acts[q * NS + n] = 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        dense_tile(act, hid, w + O_W2, w + O_B2, tid, nt);
        barrier(CLK_LOCAL_MEM_FENCE);
        dense_tile(hid, act, w + O_W3, w + O_B3, tid, nt);
        barrier(CLK_LOCAL_MEM_FENCE);
        /* layer 4 and the clamp, one sample per work-item and step */
        for (int n = tid; n < NS; n += nt) {
            float f = e.x;
            for (int o = 0; o < H; o++)
                f = MAC(f, w4[o], acts[o * NS + n]);
            const float v = MAC(0.5f, e.y, f);
            sv[n] = v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        /* each pixel's samples added in order */
        for (int lp = tid; lp < NS / SS; lp += nt) {
            const int pix = base / SS + lp;
            if (pix < phi) {
                float acc = 0.0f;
                for (int k = 0; k < SS; k++)
                    acc = acc + sv[lp * SS + k];
                const int gl = pix / P, pi = pix - gl * P;
                const int py = pi / tw, px = pi - py * tw;
                outv[(g0 + gl) * plane + (ty + py) * cell_w + tx + px] = FINISH(acc);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
#endif
