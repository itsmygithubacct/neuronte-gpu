/* neuronte-gpu CUDA kernels for file handles: bundles, tiled nets, drawing
 * files and wide nets. See "file handles" in src/internal.h for the tables.
 *
 * Compiled to PTX once per hidden width H up to 64 (-DNEURONTE_H=H, the
 * narrow kernel) and once per accumulator count R (-DNEURONTE_R=R, the wide
 * kernel, for hidden widths 65 to 32 R).
 *
 * A launch is a run of the block table. A block draws a slice of the pixels
 * or samples of a few glyphs of one segment: a rectangle of the cell's
 * sample grid, one net's weights, two position tables and a slope. Which
 * glyphs, and the code each is drawn from, come from the pair table.
 *
 * Narrow: as kernels/cover.cu. The block copies layers 2 to 4 into shared
 * memory, computes its glyphs' character parts, and every thread takes whole
 * samples with the hidden vectors in registers.
 *
 * Wide: the weights stay in device memory, stored by input. Thirty-two
 * threads (one warp) share one sample: each holds R outputs' sums in
 * registers and adds input after input in ascending order, as the CPU
 * library's axpy form does; the hidden vectors are in shared memory. An
 * input ReLU set to zero is skipped by the whole warp, as the library skips
 * it. Layer 4 is one thread's ordered sum.
 *
 * Where a pixel's samples come from more than one segment, or from the wide
 * kernel, the samples go to a sample buffer and neuronte_reduce adds each
 * pixel's samples in the library's order.
 *
 * EXACT = 1 uses separately rounded multiplies and adds, so results match
 * the CPU library bit for bit. EXACT = 0 uses fused multiply-adds.
 */
#if defined(NEURONTE_H) == defined(NEURONTE_R)
#error "define NEURONTE_H (narrow) or NEURONTE_R (wide)"
#endif

#if defined(__clang__) && !defined(__NVCC__)
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define INLINE __attribute__((always_inline)) inline
#define TID() ((int)__nvvm_read_ptx_sreg_tid_x())
#define NTID() ((int)__nvvm_read_ptx_sreg_ntid_x())
#define CTAX() ((int)__nvvm_read_ptx_sreg_ctaid_x())
#define SYNC() __nvvm_bar_sync(0)
#define ADD(a, b) __nvvm_add_rn_f((a), (b))
#define MUL(a, b) __nvvm_mul_rn_f((a), (b))
#define FMA(a, b, c) __nvvm_fma_rn_f((a), (b), (c))
#define DIV(a, b) __nvvm_div_rn_f((a), (b))
#define POPC(x) __builtin_popcount(x)
#define ALIGN16 __attribute__((aligned(16)))
typedef unsigned long size_t;
#else
#define INLINE __forceinline__
#define TID() ((int)threadIdx.x)
#define NTID() ((int)blockDim.x)
#define CTAX() ((int)blockIdx.x)
#define SYNC() __syncthreads()
#define ADD(a, b) __fadd_rn((a), (b))
#define MUL(a, b) __fmul_rn((a), (b))
#define FMA(a, b, c) __fmaf_rn((a), (b), (c))
#define DIV(a, b) __fdiv_rn((a), (b))
#define POPC(x) __popc(x)
#define ALIGN16 __align__(16)
#endif

#define CODE 64
#define NBITS 21

/* Must match the host (src/internal.h). */
enum { S_PARAM, S_MASKS, S_X0, S_Y0, S_W, S_H, S_XTAB, S_YTAB, S_SLOPE, S_HID, S_INTS };
enum { B_SEG, B_PAIR0, B_NPAIRS, B_LO, B_HI, B_MODE, B_INTS };
enum { BM_PIXEL, BM_PIXEL_SPLIT, BM_SAMPLE, BM_DIRECT };

/* acc + a * b, rounded like the CPU (EXACT) or fused. */
template <int EXACT> static __device__ INLINE float mac(float acc, float a, float b)
{
    return EXACT ? ADD(acc, MUL(a, b)) : FMA(a, b, acc);
}

/* The character part of layer 1 for one output: bias plus the 64 code columns. */
static __device__ INLINE float char_part(const float *p, const unsigned *mk, unsigned s, int hid, int o)
{
    float c = p[CODE * hid + o]; /* b1 */
    for (int i = 0; i < CODE; i++) {
        int bit = i < NBITS ? (int)((s >> i) & 1u) : (POPC(s & mk[i - NBITS]) & 1);
        float w = p[i * hid + o];
        c = ADD(c, bit ? w : -w); /* (+-1) * w is exact */
    }
    return c;
}

#define ARGS                                                                                             \
    const float *__restrict__ params, const int *__restrict__ segs, const int *__restrict__ blocks,     \
        const unsigned *__restrict__ pairs, const unsigned *__restrict__ masks, float *__restrict__ out, \
        float *__restrict__ samp, int block0, int cell_w, int cell_h, int S
#define PASS params, segs, blocks, pairs, masks, out, samp, block0, cell_w, cell_h, S

#ifdef NEURONTE_H
/* ---------------------------------------------------------------- narrow */
#define H NEURONTE_H

/* Zero, but opaque to the compiler: keeps the 2 H^2 weight loads inside the sample loop
 * (see kernels/cover.cu). */
static __device__ INLINE int opaque_zero(int v)
{
    int z;
    asm("and.b32 %0, %1, 0;" : "=r"(z) : "r"(v));
    return z;
}

template <int EXACT>
static __device__ INLINE void dense(const float *w, const float *b, const float *in, float *out)
{
#pragma unroll
    for (int o = 0; o < H; o++) {
        float s = b[o];
#pragma unroll
        for (int i = 0; i < H; i++)
            s = mac<EXACT>(s, in[i], w[o * H + i]);
        out[o] = s > 0.0f ? s : 0.0f;
    }
}

template <int EXACT>
static __device__ INLINE float sample(const float *cg, const float *xr, const float *yr, const float *w, float b4,
                                      float slope)
{
    const float *w2 = w, *b2 = w + H * H, *w3 = w + H * H + H, *b3 = w + 2 * H * H + H;
    const float *w4 = w + 2 * H * H + 2 * H;
    float a[H], b[H];
#pragma unroll
    for (int o = 0; o < H; o++) {
        float x = ADD(ADD(cg[o], xr[o]), yr[o]);
        a[o] = x > 0.0f ? x : 0.0f;
    }
    dense<EXACT>(w2, b2, a, b);
    dense<EXACT>(w3, b3, b, a);
    float f = b4;
#pragma unroll
    for (int o = 0; o < H; o++)
        f = mac<EXACT>(f, w4[o], a[o]);
    float v = mac<EXACT>(0.5f, slope, f);
    return v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
}

template <int EXACT> static __device__ INLINE void cover(ARGS)
{
    extern __shared__ ALIGN16 float sm[];
    const int *bk = blocks + (size_t)(block0 + CTAX()) * B_INTS;
    const int *sg = segs + (size_t)bk[B_SEG] * S_INTS;
    const unsigned *pr = pairs + 2 * (size_t)bk[B_PAIR0];
    const int np = bk[B_NPAIRS], lo = bk[B_LO], hi = bk[B_HI], mode = bk[B_MODE];
    const float *p = params + sg[S_PARAM];
    const float *X = params + sg[S_XTAB], *Y = params + sg[S_YTAB];
    const unsigned *mk = masks + sg[S_MASKS];
    const int x0 = sg[S_X0], y0 = sg[S_Y0], w = sg[S_W], h = sg[S_H];
    const float slope = params[sg[S_SLOPE]];
    /* shared: w2 [H][H], b2, w3 [H][H], b3, w4, b4; then the pairs' character parts; then a
     * sample value per thread (BM_PIXEL_SPLIT) */
    const int nw = 2 * H * H + 3 * H + 1;
    float *C = sm + ((nw + 3) & ~3);

    for (int i = TID(); i < nw; i += NTID())
        sm[i] = p[(CODE + 1) * H + i];
    for (int k = TID(); k < np * H; k += NTID()) {
        int g = k / H, o = k - g * H;
        C[k] = char_part(p, mk, pr[2 * g + 1], H, o);
    }
    SYNC();

    const float b4 = sm[2 * H * H + 3 * H];
    const float ss = (float)(S * S);
    const int SS = S * S;
    const size_t plane = (size_t)cell_w * (size_t)cell_h;

    if (mode == BM_SAMPLE) {
        /* one item is one sample of the segment, row-major; its value goes to the sample buffer */
        const int Q = w * h, sw = cell_w * S;
        const size_t splane = (size_t)sw * (size_t)(cell_h * S);
        for (int it = lo + TID(); it < hi; it += NTID()) {
            const int g = it / Q, r = it - g * Q;
            const int sj = r / w, si = r - sj * w;
            samp[(size_t)pr[2 * g] * splane + (size_t)(y0 + sj) * sw + x0 + si] =
                sample<EXACT>(C + g * H, X + (size_t)si * H, Y + (size_t)sj * H,
                              (const float *)__builtin_assume_aligned(sm + opaque_zero(it), 16), b4, slope);
        }
        return;
    }
    /* items are pixels of a segment whose rectangle is whole pixels */
    const int tw = w / S, th = h / S, P = tw * th, px0 = x0 / S, py0 = y0 / S;
    if (H >= 16 && mode == BM_PIXEL_SPLIT) {
        /* One thread per sample: (pair, pixel, sub-row, sub-column), in that order. Each pass of the
         * block takes `per` consecutive samples, a whole number of pixels; the values meet in
         * shared memory and one thread per pixel adds them in the library's order. */
        const int per = NTID() / SS * SS, slo = lo * SS, shi = hi * SS;
        float *V = C + ((np * H + 3) & ~3);
        for (int base = slo; base < shi; base += per) {
            const int it = base + TID();
            const bool on = TID() < per && it < shi;
            const int pix = it / SS, sub = it - pix * SS;
            const int g = pix / P, pi = pix - g * P;
            const int py = pi / tw, px = pi - py * tw;
            float v = 0.0f;
            if (on) {
                const int j = sub / S, i = sub - j * S;
                v = sample<EXACT>(C + g * H, X + (size_t)(px * S + i) * H, Y + (size_t)(py * S + j) * H,
                                  (const float *)__builtin_assume_aligned(sm + opaque_zero(it), 16), b4, slope);
            }
            if (SS == 1) {
                if (on)
                    out[(size_t)pr[2 * g] * plane + (size_t)(py0 + py) * cell_w + px0 + px] = DIV(ADD(0.0f, v), ss);
            } else {
                V[TID()] = v;
                SYNC();
                if (on && sub == 0) {
                    float acc = 0.0f;
                    for (int k = 0; k < SS; k++)
                        acc = ADD(acc, V[TID() + k]);
                    out[(size_t)pr[2 * g] * plane + (size_t)(py0 + py) * cell_w + px0 + px] = DIV(acc, ss);
                }
                SYNC();
            }
        }
    } else {
        for (int pix = lo + TID(); pix < hi; pix += NTID()) {
            const int g = pix / P, pi = pix - g * P;
            const int py = pi / tw, px = pi - py * tw;
            float acc = 0.0f;
            for (int j = 0; j < S; j++)
                for (int i = 0; i < S; i++)
                    acc = ADD(acc, sample<EXACT>(C + g * H, X + (size_t)(px * S + i) * H, Y + (size_t)(py * S + j) * H,
                                                 (const float *)__builtin_assume_aligned(
                                                     sm + opaque_zero(pix * SS + j * S + i), 16),
                                                 b4, slope));
            out[(size_t)pr[2 * g] * plane + (size_t)(py0 + py) * cell_w + px0 + px] = DIV(acc, ss);
        }
    }
}

#else
/* ------------------------------------------------------------------ wide */
#define R NEURONTE_R
#define LANES 32
#define HP (R * LANES) /* outputs, padded to R per lane */

/* out = ReLU(W in + b) for this lane's R outputs. wt is by input: row i, HP wide, holds the weights
 * from input i. Inputs are added in ascending order; a zero input adds nothing (the CPU library
 * skips it too), and every lane of the warp sees the same input, so the skip does not diverge. */
template <int EXACT>
static __device__ INLINE void dense_w(const float *wt, const float *b, const float *in, float *outv, int hid,
                                      int lane)
{
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++)
        acc[r] = b[lane * R + r];
    for (int i = 0; i < hid; i++) {
        const float a = in[i];
        if (a != 0.0f) {
            const float *row = wt + (size_t)i * HP + lane * R;
#pragma unroll
            for (int r = 0; r < R; r++)
                acc[r] = mac<EXACT>(acc[r], a, row[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++)
        outv[lane * R + r] = acc[r] > 0.0f ? acc[r] : 0.0f;
}

template <int EXACT> static __device__ INLINE void cover(ARGS)
{
    extern __shared__ ALIGN16 float sm[];
    const int *bk = blocks + (size_t)(block0 + CTAX()) * B_INTS;
    const int *sg = segs + (size_t)bk[B_SEG] * S_INTS;
    const unsigned *pr = pairs + 2 * (size_t)bk[B_PAIR0];
    const int np = bk[B_NPAIRS], lo = bk[B_LO], hi = bk[B_HI], mode = bk[B_MODE];
    const int hid = sg[S_HID];
    const float *p = params + sg[S_PARAM];
    const float *X = params + sg[S_XTAB], *Y = params + sg[S_YTAB];
    const unsigned *mk = masks + sg[S_MASKS];
    const int x0 = sg[S_X0], y0 = sg[S_Y0], w = sg[S_W], h = sg[S_H];
    const float slope = params[sg[S_SLOPE]];
    const int lane = TID() & (LANES - 1), slot = TID() / LANES, M = NTID() / LANES;
    /* shared: the pairs' character parts; then two hidden vectors per sample in flight */
    float *C = sm;
    float *A = sm + ((np * hid + 3) & ~3) + slot * 2 * HP, *B = A + HP;
    const float *w2 = p + (CODE + 1) * hid, *b2 = w2 + (size_t)hid * HP, *w3 = b2 + HP;
    const float *b3 = w3 + (size_t)hid * HP, *w4 = b3 + HP;
    const float b4 = w4[hid];

    for (int k = TID(); k < np * hid; k += NTID()) {
        int g = k / hid, o = k - g * hid;
        C[k] = char_part(p, mk, pr[2 * g + 1], hid, o);
    }
    SYNC();

    /* one item is one sample of the segment, row-major; a pass takes M of them, one per warp */
    const int Q = w * h, sw = cell_w * S;
    const size_t plane = (size_t)cell_w * (size_t)cell_h, splane = (size_t)sw * (size_t)(cell_h * S);
    for (int base = lo; base < hi; base += M) {
        const int it = base + slot;
        const bool on = it < hi;
        const int g = it / Q, r = it - g * Q;
        const int sj = r / w, si = r - sj * w;
        if (on) {
            const float *cg = C + g * hid, *xr = X + (size_t)si * hid, *yr = Y + (size_t)sj * hid;
#pragma unroll
            for (int k = 0; k < R; k++) {
                const int o = lane * R + k;
                if (o < hid) {
                    float x = ADD(ADD(cg[o], xr[o]), yr[o]);
                    A[o] = x > 0.0f ? x : 0.0f;
                }
            }
        }
        SYNC();
        if (on)
            dense_w<EXACT>(w2, b2, A, B, hid, lane);
        SYNC();
        if (on)
            dense_w<EXACT>(w3, b3, B, A, hid, lane);
        SYNC();
        if (on && lane == 0) {
            float f = b4;
            for (int o = 0; o < hid; o++)
                f = mac<EXACT>(f, w4[o], A[o]);
            float v = mac<EXACT>(0.5f, slope, f);
            v = v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
            if (mode == BM_DIRECT)
                out[(size_t)pr[2 * g] * plane + (size_t)(y0 + sj) * cell_w + x0 + si] = ADD(0.0f, v);
            else
                samp[(size_t)pr[2 * g] * splane + (size_t)(y0 + sj) * sw + x0 + si] = v;
        }
        SYNC(); /* layer 4 reads A until here */
    }
}
#endif

extern "C" __global__ void neuronte_cover2_exact(ARGS)
{
    cover<1>(PASS);
}

extern "C" __global__ void neuronte_cover2_fast(ARGS)
{
    cover<0>(PASS);
}

/* Pixels of the listed glyphs from the sample buffer: each pixel's samples added row by row, as
 * the CPU library adds them, then divided by their number. One thread per pixel. */
extern "C" __global__ void neuronte_reduce(const float *__restrict__ samp, float *__restrict__ out,
                                           const unsigned *__restrict__ list, int count, int cell_w, int cell_h,
                                           int S)
{
    const size_t plane = (size_t)cell_w * (size_t)cell_h;
    const size_t idx = (size_t)CTAX() * (size_t)NTID() + (size_t)TID();
    if (idx >= (size_t)count * plane)
        return;
    const size_t gi = idx / plane, pi = idx - gi * plane;
    const unsigned g = list[gi];
    const int py = (int)(pi / (size_t)cell_w), px = (int)(pi - (size_t)py * cell_w);
    const int sw = cell_w * S;
    const float *s = samp + (size_t)g * ((size_t)sw * (size_t)(cell_h * S));
    float acc = 0.0f;
    for (int j = 0; j < S; j++)
        for (int i = 0; i < S; i++)
            acc = ADD(acc, s[(size_t)(py * S + j) * sw + px * S + i]);
    out[(size_t)g * plane + pi] = DIV(acc, (float)(S * S));
}
