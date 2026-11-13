/* neuronte-gpu CUDA kernel: coverage of a batch of glyphs for one model set.
 *
 * Compiled once per hidden width H (-DNEURONTE_H=H) to PTX, by clang's NVPTX
 * target (no CUDA toolkit needed) or by nvcc. The host loads the PTX with the
 * driver API, which compiles it for the GPU at hand.
 *
 * One block evaluates one tile for a group of `gpb` glyphs:
 *   1. copies the tile's layers 2 to 4 into shared memory;
 *   2. computes each glyph's character part of layer 1 (bias plus the 64
 *      code columns), H values per glyph, into shared memory;
 *   3. gives every thread one pixel at a time: for each of its S x S samples
 *      layer 1 is character part + x part + y part (the position parts are
 *      tables made on the host), then layers 2 to 4, the coverage ramp and the
 *      clamp, averaged over the samples and written to the glyph's raster.
 *
 * EXACT = 1 uses the CPU library's operation order with separately rounded
 * multiplies and adds (no contraction), so results match it bit for bit.
 * EXACT = 0 uses fused multiply-adds.
 */
#ifndef NEURONTE_H
#error "define NEURONTE_H"
#endif
#define H NEURONTE_H

#if defined(__clang__) && !defined(__NVCC__)
/* clang without CUDA headers */
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define INLINE __attribute__((always_inline)) inline
#define TID() ((int)__nvvm_read_ptx_sreg_tid_x())
#define NTID() ((int)__nvvm_read_ptx_sreg_ntid_x())
#define CTAX() ((int)__nvvm_read_ptx_sreg_ctaid_x())
#define CTAY() ((int)__nvvm_read_ptx_sreg_ctaid_y())
#define CTAZ() ((int)__nvvm_read_ptx_sreg_ctaid_z())
#define NCTAZ() ((int)__nvvm_read_ptx_sreg_nctaid_z())
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
#define CTAY() ((int)blockIdx.y)
#define CTAZ() ((int)blockIdx.z)
#define NCTAZ() ((int)gridDim.z)
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
#define BRAILLE_FIRST 0x2800u

/* Per-tile record in `tiles`, ints. Must match the host (neuronte_gpu.c). */
enum { T_X, T_Y, T_W, T_H, T_MASK, T_PARAM, T_XTAB, T_YTAB, T_MASKS, T_INTS };

/* Zero, but opaque to the compiler. The weights in shared memory do not change
 * between samples, so without this the compiler hoists all 2 H^2 of their loads
 * out of the sample loop and spills them to local memory. */
static __device__ INLINE int opaque_zero(int v)
{
    int z;
    asm("and.b32 %0, %1, 0;" : "=r"(z) : "r"(v));
    return z;
}

/* acc + a * b, rounded like the CPU (EXACT) or fused. */
template <int EXACT> static __device__ INLINE float mac(float acc, float a, float b)
{
    return EXACT ? ADD(acc, MUL(a, b)) : FMA(a, b, acc);
}

/* out = ReLU(W in + b); w is [out][in] in shared memory. Each output sums its
 * inputs in ascending order, as the CPU's axpy form does. */
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

/* One sample's clamped value: layers 1 (from the character part and the axis tables) to 4. */
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

/* A launch too small to fill the GPU (per_sample set by the host) gives each thread one sample, and
 * a pass of the block ends in an ordered reduction, so one glyph spreads over many threads; a full
 * launch gives each thread a pixel's samples in turn, which costs no barriers. Below this hidden
 * width a pixel's chain is short and only the second form is compiled. */
#define SAMPLE_ITEMS_H 16

template <int EXACT>
static __device__ INLINE void cover(const float *__restrict__ params, const int *__restrict__ tiles,
                                    const unsigned *__restrict__ masks, const unsigned *__restrict__ scalars,
                                    float *__restrict__ out, int nglyph, int gpb, int cell_w, int cell_h, int S,
                                    int per_sample)
{
    extern __shared__ ALIGN16 float sm[];
    const int *tl = tiles + CTAX() * T_INTS;
    const int tx = tl[T_X], ty = tl[T_Y], tw = tl[T_W], th = tl[T_H], mask = tl[T_MASK];
    const float *p = params + tl[T_PARAM];
    const float *X = params + tl[T_XTAB], *Y = params + tl[T_YTAB];
    const unsigned *mk = masks + tl[T_MASKS];
    const int g0 = CTAY() * gpb;
    /* shared: w2 [H][H], b2, w3 [H][H], b3, w4, b4, slope; then gpb character parts; then a
     * sample value per thread (per_sample) */
    const int nw = 2 * H * H + 3 * H + 2;
    float *C = sm + ((nw + 3) & ~3);

    for (int i = TID(); i < nw; i += NTID())
        sm[i] = p[(CODE + 1) * H + i];
    for (int k = TID(); k < gpb * H; k += NTID()) {
        int g = k / H, o = k - g * H, glyph = g0 + g;
        if (glyph >= nglyph)
            break;
        unsigned s = scalars[glyph];
        if (mask >= 0)
            s = BRAILLE_FIRST | ((s - BRAILLE_FIRST) & (unsigned)mask);
        if (s > 0x10FFFFu || (s >= 0xD800u && s <= 0xDFFFu)) /* as neuronte_coverage normalizes */
            s = 0xFFFDu;
        float c = p[CODE * H + o]; /* b1 */
        for (int i = 0; i < CODE; i++) {
            int bit = i < NBITS ? (int)((s >> i) & 1u) : (POPC(s & mk[i - NBITS]) & 1);
            float w = p[i * H + o];
            c = ADD(c, bit ? w : -w); /* (+-1) * w is exact */
        }
        C[k] = c;
    }
    SYNC();

    const float b4 = sm[2 * H * H + 3 * H], slope = sm[2 * H * H + 3 * H + 1];
    const float ss = (float)(S * S);
    const int P = tw * th;
    const size_t plane = (size_t)cell_w * (size_t)cell_h;

    /* Blocks along z share out the pixels of one tile and glyph group, so a call with too few glyphs
     * still fills the GPU. */
    const int SS = S * S;
    const int pixels = gpb * P, pchunk = (pixels + NCTAZ() - 1) / NCTAZ();
    const int plo = CTAZ() * pchunk;
    int phi = plo + pchunk;
    if (phi > pixels)
        phi = pixels;
    if (nglyph - g0 < gpb && phi > (nglyph - g0) * P) /* the last group may be partly empty */
        phi = (nglyph - g0) * P;
    if (H >= SAMPLE_ITEMS_H && per_sample) {
        /* One item is one sample: (glyph, pixel, sub-row, sub-column), in that order. Each pass of
         * the block takes `per` consecutive items, a whole number of pixels; the samples' values
         * meet in shared memory and one thread per pixel adds them in the library's order. */
        const int per = NTID() / SS * SS, lo = plo * SS, hi = phi * SS;
        float *V = C + gpb * H; /* one sample value per thread */
        for (int base = lo; base < hi; base += per) {
            const int it = base + TID();
            const bool on = TID() < per && it < hi;
            const int pix = it / SS, sub = it - pix * SS;
            const int g = pix / P, pi = pix - g * P;
            const int py = pi / tw, px = pi - py * tw;
            float v = 0.0f;
            if (on) {
                const int j = sub / S, i = sub - j * S;
                v = sample<EXACT>(C + g * H, X + (size_t)(px * S + i) * H, Y + (size_t)(py * S + j) * H,
                                  (const float *)__builtin_assume_aligned(sm + opaque_zero(it), 16), b4, slope);
            }
            float *dst = out + (size_t)(g0 + g) * plane + (size_t)(ty + py) * cell_w + tx + px;
            if (SS == 1) {
                if (on)
                    *dst = DIV(ADD(0.0f, v), ss);
            } else {
                V[TID()] = v;
                SYNC();
                if (on && sub == 0) {
                    float acc = 0.0f;
                    for (int k = 0; k < SS; k++)
                        acc = ADD(acc, V[TID() + k]);
                    *dst = DIV(acc, ss);
                }
                SYNC();
            }
        }
    } else {
        for (int pix = plo + TID(); pix < phi; pix += NTID()) {
            const int g = pix / P, pi = pix - g * P;
            const int py = pi / tw, px = pi - py * tw;
            float acc = 0.0f;
            for (int j = 0; j < S; j++)
                for (int i = 0; i < S; i++)
                    acc = ADD(acc, sample<EXACT>(C + g * H, X + (size_t)(px * S + i) * H, Y + (size_t)(py * S + j) * H,
                                                 (const float *)__builtin_assume_aligned(
                                                     sm + opaque_zero(pix * SS + j * S + i), 16),
                                                 b4, slope));
            out[(size_t)(g0 + g) * plane + (size_t)(ty + py) * cell_w + tx + px] = DIV(acc, ss);
        }
    }
}

#define ARGS                                                                                             \
    const float *__restrict__ params, const int *__restrict__ tiles, const unsigned *__restrict__ masks, \
        const unsigned *__restrict__ scalars, float *__restrict__ out, int nglyph, int gpb, int cell_w,  \
        int cell_h, int S, int per_sample

extern "C" __global__ void neuronte_cover_exact(ARGS)
{
    cover<1>(params, tiles, masks, scalars, out, nglyph, gpb, cell_w, cell_h, S, per_sample);
}

extern "C" __global__ void neuronte_cover_fast(ARGS)
{
    cover<0>(params, tiles, masks, scalars, out, nglyph, gpb, cell_w, cell_h, S, per_sample);
}
