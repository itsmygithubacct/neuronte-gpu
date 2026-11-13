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

template <int EXACT>
static __device__ INLINE void cover(const float *__restrict__ params, const int *__restrict__ tiles,
                                    const unsigned *__restrict__ masks, const unsigned *__restrict__ scalars,
                                    float *__restrict__ out, int nglyph, int gpb, int cell_w, int cell_h, int S)
{
    extern __shared__ ALIGN16 float sm[];
    const int *tl = tiles + CTAX() * T_INTS;
    const int tx = tl[T_X], ty = tl[T_Y], tw = tl[T_W], th = tl[T_H], mask = tl[T_MASK];
    const float *p = params + tl[T_PARAM];
    const float *X = params + tl[T_XTAB], *Y = params + tl[T_YTAB];
    const unsigned *mk = masks + tl[T_MASKS];
    const int g0 = CTAY() * gpb;
    /* shared: w2 [H][H], b2, w3 [H][H], b3, w4, b4, slope; then gpb character parts */
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

    for (int it = TID(); it < gpb * P; it += NTID()) {
        int g = it / P, pi = it - g * P, glyph = g0 + g;
        if (glyph >= nglyph)
            break;
        int py = pi / tw, px = pi - py * tw;
        const float *cg = C + g * H;
        float acc = 0.0f;
        for (int j = 0; j < S; j++) {
            const float *yr = Y + (size_t)(py * S + j) * H;
            for (int i = 0; i < S; i++) {
                const float *xr = X + (size_t)(px * S + i) * H;
                const float *w = (const float *)__builtin_assume_aligned(sm + opaque_zero(it * S * S + j * S + i), 16);
                const float *w2 = w, *b2 = w + H * H, *w3 = w + H * H + H, *b3 = w + 2 * H * H + H;
                const float *w4 = w + 2 * H * H + 2 * H;
                float a[H], b[H];
#pragma unroll
                for (int o = 0; o < H; o++) {
                    float v = ADD(ADD(cg[o], xr[o]), yr[o]);
                    a[o] = v > 0.0f ? v : 0.0f;
                }
                dense<EXACT>(w2, b2, a, b);
                dense<EXACT>(w3, b3, b, a);
                float f = b4;
#pragma unroll
                for (int o = 0; o < H; o++)
                    f = mac<EXACT>(f, w4[o], a[o]);
                float v = mac<EXACT>(0.5f, slope, f);
                acc = ADD(acc, v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v);
            }
        }
        out[(size_t)glyph * plane + (size_t)(ty + py) * cell_w + tx + px] = DIV(acc, ss);
    }
}

#define ARGS                                                                                             \
    const float *__restrict__ params, const int *__restrict__ tiles, const unsigned *__restrict__ masks, \
        const unsigned *__restrict__ scalars, float *__restrict__ out, int nglyph, int gpb, int cell_w,  \
        int cell_h, int S

extern "C" __global__ void neuronte_cover_exact(ARGS)
{
    cover<1>(params, tiles, masks, scalars, out, nglyph, gpb, cell_w, cell_h, S);
}

extern "C" __global__ void neuronte_cover_fast(ARGS)
{
    cover<0>(params, tiles, masks, scalars, out, nglyph, gpb, cell_w, cell_h, S);
}
