# neuronte-gpu

A GPU implementation of [Neuronte](https://github.com/itsmygithubacct/neuronte),
with a benchmark that compares it against Neuronte's standard CPU
implementation.

Neuronte is a small neural network whose weights are a font. Its C library
evaluates the network on the CPU to draw glyphs into a pixel buffer. This
project runs the same network on a GPU, with CUDA on NVIDIA GPUs, Vulkan
compute on GPUs without CUDA, and OpenCL on GPUs such as Arm Mali whose
drivers offer OpenCL but no Vulkan, or a slower one, and measures how they
compare. The goal is a GPU path that is faster than the CPU library on the
same machine.

Development happens on the `work/rc9-gpu` branch, for the RC9 release.

## Results

Measured on 2026-11-13 against the reference CPU library (Neuronte
`a6b8508`, built with its own flags on the same machine), over 16 Braille
model sets (one network of hidden width 48, up to 48 tiles of width 1) at
8×16, 16×32 and 48×96 pixels: 48 configurations, 256 glyphs per call, end to
end (upload, kernel, download, copy to the caller), back to back with the CPU
on the same machine.

| GPU | API | CPU of the same machine | × one CPU core (min / median / max) | × all CPU threads at once | single glyph × one core | output |
|---|---|---|---|---|---|---|
| GeForce RTX 3070 | CUDA | Core i7-10700F | 53 / 218 / 350 | 5.7 / 21 / 38 (16 threads) | 2.1 / 9.2 / 23 | byte-identical, 48 of 48 |
| Quadro RTX 3000 (laptop) | CUDA | Core i7-9850H | 47 / 212 / 1,058 | 3.9 / 22 / 60 (12 threads) | 2.0 / 9.6 / 41 | byte-identical, 48 of 48 |
| Quadro M2000M (laptop) | CUDA | Core i7-6820HQ | 9.9 / 48 / 189 | 2.3 / 11 / 45 (8 threads) | 1.8 / 6.4 / 13 | byte-identical, 48 of 48 |
| UHD Graphics 630 (laptop) | Vulkan | Core i7-9850H | 6.5 / 22 / 136 | 0.9 / 2.4 / 8.4 (12 threads) | 0.3 / 1.5 / 7.0 | byte-identical, 48 of 48 |
| Mali-G610 MP4 (RK3588) | OpenCL | Cortex-A76 + A55 | 9.3 / 16 / 32 | 1.9 / 3.4 / 6.5 (8 cores, bound) | 0.12 / 1.2 / 3.7 | byte-identical, 48 of 48 |
| Mali-G610 MP4 (RK3588) | Vulkan | Cortex-A76 + A55 | 1.2 / 4.7 / 29 | 0.26 / 0.94 / 5.9 (8 cores, bound) | 0.17 / 0.54 / 1.3 | byte-identical, 48 of 48 |
| Mali-T860 MP4 (RK3399) | OpenCL | Cortex-A72 + A53 | 2.4 / 3.7 / 22 | 0.75 / 1.1 / 6.5 (6 cores, bound) | 0.13 / 0.50 / 1.5 | byte-identical, 48 of 48 |

The library itself is single-threaded per call; "all CPU threads" runs one
CPU process per hardware thread at once and adds their rates. The RTX 3000
laptop's CPU figures were taken under load from other work, which inflates
its ratios. The Intel integrated GPU stays at its lowest clock for short,
spaced single-glyph calls, so for small models a single glyph is faster on
the CPU there; batched, it is faster than one core in every configuration
and faster than the whole CPU in 46 of 48.

On the Arm boards "one CPU core" is a big core (Cortex-A76 at up to
2.35 GHz, Cortex-A72 at 1.8 GHz), and the all-core figure is a bound: the
boards run other work, so each core type was measured alone and the rates
were added (4 × A55 + 4 × A76, 4 × A53 + 2 × A72). Cores run together share
memory and heat, so the GPU's true advantage over the whole CPU is at least
the figure shown. The Mali-G610 with OpenCL is faster than all eight cores
in every configuration. The Mali-T860 is faster than all six for hidden
widths up to 6, and 0.75 to 1.0 times as fast for widths of 12 and more.

## What it computes

A Neuronte distance network maps a position in the cell and the code of a
character to a signed distance: 90 inputs (26 position features, 64 code
features), three hidden layers of width H with ReLU, one output. The library
supersamples each pixel (2 × 2 below a 24 px cell width, otherwise 1), turns
each output into coverage with a clamped ramp and averages the samples.

neuronte-gpu evaluates a *model set*: one network over the whole cell, or
several networks ("tiles") that each draw a rectangle of the cell, as in the
Braille spatial-split experiments. Each tile is evaluated on its own local
grid with the scalar masked to the dots it draws, exactly as calling
`neuronte_coverage` per tile does. The result of one call is the coverage of
a batch of scalars, one raster per scalar.

## How it works

- **Precomputed layer 1.** Layer 1 is linear in the inputs, so it splits
  into a part that depends on the character (64 code features, computed once
  per glyph and tile on the GPU) and parts that depend on x and on y alone
  (computed once at open for every sample column and row of each tile, in
  double precision and rounded to float, exactly as the library does). A
  sample's layer 1 is then two additions and a ReLU.
- **One kernel for every model set.** A launch covers all tiles (grid x),
  groups of glyphs (grid y) and, when a call is too small to fill the GPU,
  slices of each tile's pixels (grid z). A block copies layers 2 to 4 into
  shared memory (or, on Vulkan, reads them from the parameter buffer at
  uniform addresses, which Intel's compiler turns into shared block loads),
  computes the character parts of its glyphs, then evaluates samples. For
  hidden widths of 24 and up one thread takes one sample and the samples of a
  pixel are added in a fixed order through shared memory; for narrower
  networks one thread takes a whole pixel.
- **CUDA without the CUDA toolkit.** The kernel is compiled to PTX by clang's
  NVPTX target (or by nvcc) for every supported hidden width, embedded in the
  library, and loaded through the driver library (`libcuda.so.1`) with
  `dlopen`; the driver compiles it for the GPU on first use and caches it.
- **Vulkan compute.** The same algorithm in GLSL, compiled to SPIR-V by
  glslc and embedded; the hidden width, workgroup size and weight source are
  specialization constants, so one module serves every model set. The Vulkan
  loader (`libvulkan.so.1`) is also opened with `dlopen`, so one binary runs
  on machines with either, both or neither.
- **OpenCL.** The same layout of work in OpenCL C 1.2, embedded as source and
  compiled by the driver at open for the model set's hidden width and sample
  count (`kernels/cover.cl`); the ICD loader (`libOpenCL.so.1`) is opened
  with `dlopen` and no OpenCL headers are needed. It is shaped for GPUs with
  few registers per thread and 4-wide arithmetic, such as Mali: weights are
  stored input-major so four outputs' weights are one 16-byte load. There are
  two kernels. The per-sample kernel evaluates layer 2 one input at a time
  into all H sums, as the library does, so only H values stay live. The tile
  kernel runs layers 2 and 3 as matrix products over 64 samples held in
  local memory, each thread computing 4 outputs × 4 samples, which does more
  arithmetic per load. The tile kernel is used from H = 12 on Midgard
  (Mali-T) and from H = 32 on other GPUs, where each measured faster.

## Exactness

By default the GPU uses the CPU library's operation order with separately
rounded multiplies and adds (CUDA `__fmul_rn`/`__fadd_rn`; GLSL `precise`,
which forbids contraction into fused multiply-adds; OpenCL
`#pragma OPENCL FP_CONTRACT OFF`). Its coverage then
matches the library's bit for bit wherever both round IEEE single precision
the same way, which held for every model set, cell size and GPU measured.
`NEURONTE_GPU_FAST` (bench `--fast`) allows fused multiply-adds; the largest
difference from the CPU measured with it is below 1e-5 and no pixel crossed
0.5. It gains little (a median 1.1× on the GPUs measured, most for wide
networks); on the Intel GPU measured it is slower than exact mode at H=48.

## Building

```sh
make NEURONTE_DIR=/path/to/neuronte
```

`NEURONTE_DIR` must contain `include/neuronte.h` and `build/libneuronte.a`
(the reference CPU library, used for the CPU path and the fallback). The build
needs a C11 compiler and Python 3; making the kernels needs clang with the
NVPTX target (or `PTXCC=nvcc`) and glslc. Generated kernels in `build/` are
reused, so a machine without clang or glslc can build from a copy of them.
The Vulkan backend is built when the Vulkan headers are found
(`VULKAN_INCLUDE=DIR` points at a copy; `VULKAN=0` leaves it out). The
OpenCL backend is always built; it needs no headers or SDK. `CUDA=0`
leaves out the CUDA kernels; `HIDDEN="4 12 48"` limits the widths compiled
(default 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64).

Outputs: `build/libneuronte_gpu.a`, `build/neuronte_gpu_bench` and the tools
listed under "Whole Neuronte files". The OpenGL backend needs no headers
or SDK; it loads EGL and OpenGL at run time.

## Using it

```c
#include "neuronte_gpu.h"

struct neuronte_gpu_options opt = {0};
opt.nrt_path = "braille48.nrt";     /* or opt.layout_path = "split.layout" */
opt.cell_w = 16;
opt.cell_h = 32;
opt.flags = NEURONTE_GPU_ALLOW_FALLBACK;
neuronte_gpu *g;
char why[512];
if (neuronte_gpu_open(&opt, &g, why, sizeof why) == 0) {
    float out[256 * 16 * 32];
    neuronte_gpu_coverage(g, scalars, 256, out, NULL);
    printf("%s on %s\n", neuronte_gpu_backend_name(neuronte_gpu_backend(g)), neuronte_gpu_device(g));
    neuronte_gpu_close(g);
}
```

The backend is chosen at open: CUDA, then Vulkan, then OpenCL
(`NEURONTE_GPU_AUTO`), or the one requested. Where Arm's kernel driver is
present (`/dev/mali0`), AUTO tries OpenCL before Vulkan, since Arm's OpenCL
is the faster of the two there (see "Arm Mali" below). A Vulkan or OpenCL
device that is a CPU (lavapipe, PoCL) is refused unless
`NEURONTE_GPU_ALLOW_CPU_DEVICE` is given, and is then reported as such. When
no GPU can be used, open fails with the reason, or with
`NEURONTE_GPU_ALLOW_FALLBACK` returns a handle on the reference CPU library
labelled `cpu-fallback`, with the reason in `neuronte_gpu_fallback_reason`.
The backend, device, driver versions and device type are always reported.

A layout file describes a model set (`tools/layout.py` writes one per case
and cell size from a Braille experiment manifest):

```
neuronte-gpu-layout 1
name quadrants12-s0
cell 16 32
tile 3 0 0 8 16 top-left-s0.nrt
tile 24 8 0 8 16 top-right-s0.nrt
tile 68 0 16 8 16 bottom-left-s0.nrt
tile 160 8 16 8 16 bottom-right-s0.nrt
```

Each `tile` line is `MASK X Y W H FILE` (MASK -1 passes the scalar through).
A bundle `.nrt` is resolved to the member its range table routes the
layout's `span` to; scalars outside the span are drawn by the CPU library.

### The benchmark

```sh
build/neuronte_gpu_bench --nrt braille48.nrt --cell 16x32 --backend cuda \
    --cpu 2 --warmup-ms 1000 --warmup 2 --repeats 9 --passes 4 --single 64 \
    --coverage-out cuda.f32 > cuda.json
build/neuronte_gpu_bench --nrt braille48.nrt --cell 16x32 --backend cpu \
    --cpu 2 --repeats 9 --passes 4 --single 64 --coverage-out cpu.f32 > cpu.json
cmp cuda.f32 cpu.f32
```

One process measures one backend: open, the first call, a timed warmup
(GPU clocks ramp up), warmup rounds, measured rounds of `--passes` calls of
the 256 Braille scalars in a shuffled order, and `--single` one-glyph calls.
Every round is written to the JSON output with its stage times and hardware
snapshots (CPU frequency, load, temperatures, Intel GPU frequency, NVIDIA
clocks, power and utilisation, and devfreq frequency and load, which covers
Mali GPUs) at start, at the start of measurement and at the end. The coverage
dump is in scalar order, for comparison with the CPU. `--backend` is `auto`,
`cuda`, `vulkan`, `opencl` or `cpu`.

## Whole Neuronte files

The single-net path above draws one distance net, or a layout of nets of one
hidden width. A *file handle* draws any file the reference library opens that
holds distance nets: a bundle (the library's own router sends each character
to one of its nets), a tiled net, a drawing file with its letter edge table,
and nets of any hidden width up to 1,024. It is taken by itself for every
file that is not one narrow net, and for that too with `NEURONTE_GPU_FILE`.

```c
struct neuronte_gpu_options opt = {0};
opt.nrt_path = "neuronte-split-v9.nrt";
opt.cell_w = 8;
opt.cell_h = 16;
opt.max_batch = 256;                 /* the most characters in one call */
opt.flags = NEURONTE_GPU_FILE;
```

`neuronte_gpu_coverage` is the same call. Routing, the answer to "is this
character trained", the mark drawn for an untrained one and colour nets
(emoji, icons) are the reference library's own, run on the host and counted
in the timing as `cpu_glyphs` and `marks`. The floats are those of
`neuronte_coverage`, bit for bit.

Backends for file handles: CUDA (one kernel per hidden width up to 64, one
"wide" kernel family above), Vulkan (one compute shader) and OpenGL 4.3
compute (`NEURONTE_GPU_OPENGL`, only when asked for by name). OpenCL does
not take file handles. An OpenGL handle makes a surfaceless EGL context of
its own, or with `NEURONTE_GPU_GL_CURRENT` uses the context current on the
calling thread; `neuronte_gpu_coverage_device` then leaves the pixels in a
shader storage buffer (`neuronte_gpu_gl_output`) for the caller's own
shaders. `neuronte_gpu_memory` reports what a handle holds on the device.
All handles of a process on one CUDA device share one context.

On a GPU that also draws the desktop a compute dispatch cannot be
preempted, so Vulkan and OpenGL cut a launch into dispatches of about 25 ms
(`NEURONTE_GPU_SLICE_MS`). An OpenGL context that a CPU draws (llvmpipe) is
refused unless `NEURONTE_GPU_ALLOW_CPU_DEVICE` is given: two glyphs took
327 s there.

### Measured

Core i7-6820HQ with a Quadro M2000M, CUDA, the machine otherwise idle, exact
mode, microseconds per glyph for a file's whole character set in one call
(in brackets the set), against the reference library on the same machine:

| File (characters) | Cell | GPU | CPU, one core | CPU, 8 threads |
|---|---|---:|---:|---:|
| `neuronte-split-v9` (1,341) | 8x16 | 225.7 (303 ms) | 354.4 | 88.2 |
| | 16x32 | 906.2 (1,215 ms) | 1,364.0 | 346.5 |
| `neuronte-term-latin-v1` (596) | 8x16 | 91.0 (54 ms) | 189.6 | 44.3 |
| | 16x32 | 367.1 (219 ms) | 734.0 | 171.7 |
| `neuronte-ui14-v1` (96) | 14x28 | 3.93 (0.38 ms) | 231.3 | 41.3 |
| scaled desktop face (154) | 14x28 | 593.3 (91 ms) | 1,000.9 | 226.4 |

Core i7-9850H with a Quadro RTX 3000, CUDA, the CPU loaded by other work
(load average 21): `neuronte-split-v9` 34.5 / 80.6 / 139.3 us per glyph at
8x16 / 12x24 / 16x32 (the set in 46 / 108 / 187 ms), against 617.7 /
1,187 / 2,168 on one loaded thread. Time per glyph falls with the size of
the call: 144 us for one glyph, 77 for 16, 63 for 256, 34.7 for 1,024.

A call for one glyph costs 280 to 520 us on the Quadro M2000M and 97 to 168
on the Quadro RTX 3000, so a GPU pays for tables and lines, not single
letters.

Device memory of the process, CUDA, calls of up to 256 glyphs: the context
alone is about 29 MiB on the Quadro M2000M and 86 MiB on the Quadro RTX
3000; `neuronte-split-v9` at 8x16 adds about 4 MiB; 38 files of the v2 set
open together take 88.5 MiB at 8x16 and 154.1 MiB at 16x32 on the Quadro
M2000M.

The first open for a user compiles the kernels (12.5 s for
`neuronte-split-v9` on the Quadro M2000M); the driver keeps them, and the
next open takes 0.09 s. The driver does not use a cache directory it cannot
write.

Power on the Quadro RTX 3000, read from the driver every 0.2 s: 5.8 W idle;
any call, however small, holds it at about 31 W for about 17 s; whole
tables back to back 77 W. One table of 1,341 glyphs at 8x16 took 3.2 J.

Exactness on the Quadro M2000M with CUDA: 11 terminal bundles at three cell
sizes with and without the untrained mark, 38 files of the v2 set at four
sizes, and hidden widths up to 1,024: 242 of 242 identical, for the first
call and a later one.

### Tools

- `tools/exact.py` compares files and cell sizes with the CPU library,
  the first call and the last.
- `build/neuronte_gpu_soak` compares many calls of changing size.
- `tools/measure.py` times a file's whole set and single glyphs against
  the CPU library on one thread and all threads.
- `build/neuronte_gpu_text` draws a line of text from a drawing file.
- `build/neuronte_gpu_mem` reports device memory per file and in total.
- `tools/sweep.py` gives time per glyph against glyphs per call.
- `tools/power.py` samples an NVIDIA GPU's power during work.
- `build/neuronte_colour_cost` times what the CPU library pays per
  character, colour files included.
- `make build/neuronte_gpu_atlas` builds a harness that fills a terminal's
  glyph atlas by the CPU library, by a separate device context and by a
  compute shader in the terminal's own context, and compares them.
- `make build/libneuronte_gpu.so` builds a shared library.

## Arm Mali

Measured on Rockchip boards running Arm's kernel driver (kbase) with
Rockchip's builds of Arm's user-space driver (libmali):

- **Mali-G610 MP4** (RK3588; kbase g21p0, libmali g24p0): OpenCL 3.0 and
  Vulkan 1.3. Both backends match the CPU bit for bit. OpenCL is the faster
  in 46 of 48 configurations, 5 to 8 times at hidden widths of 12 and more
  (a median of 4.1 times over all 48), which is why AUTO tries it first
  where `/dev/mali0` exists. The kernel driver loads its CSF firmware
  (`mali_csffw.bin`; the one from the kernel driver's own release, g21p0,
  was used) at the first open of `/dev/mali0`; without it the open fails
  with ENOENT.
- **Mali-T860 MP4** (RK3399; Midgard kbase, libmali r18p0): OpenCL 1.2
  only; the libmali builds for it contain no Vulkan driver, and Mesa's
  Vulkan driver (panvk) does not cover Midgard.

Notes:

- libmali can be used without installing it. Point the ICD loader at a
  directory holding a `.icd` file that names the library
  (`OCL_ICD_VENDORS=DIR`), and Vulkan at an ICD JSON (`VK_ICD_FILENAMES`).
  The process needs access to `/dev/mali0` (group `video` on these boards).
- libmali prints a version banner on standard output when it loads. The
  benchmark writes its JSON to a duplicate of standard output and points
  standard output at standard error, so the banner cannot corrupt it.
- Arm's OpenCL compiles the kernel at every open (0.15 to 0.7 s on the
  G610, 0.14 to 0.3 s on the T860). A call costs a fixed 0.15 to 0.6 ms on
  the G610 and 0.5 to 1 ms on the T860, so for small models a single glyph
  is faster on the CPU.
- The G610's default clock governor (`simple_ondemand`) holds 500 to
  700 MHz when calls leave the GPU partly idle, so light model sets run
  below the 1 GHz the heavy ones reach. The results here are under the
  boards' default governors. The T860's governor was `performance`
  (800 MHz).
- Fast mode gains little on Mali. On the G610, letting the compiler fuse
  multiply-adds (by `mad` or by contraction) makes it spill the per-sample
  kernel's registers (448 private bytes at H = 12, none in exact mode), up
  to 8 times slower; with the tile kernel (H = 48) it is 3 to 11% faster.
  On the T860 its output was identical to exact mode's, and it was no
  faster. Use exact mode on Mali.

## Limits

- The single-net path: one hidden width per model set on the GPU (mixed
  widths use the CPU); CUDA: the compiled hidden widths; Vulkan and
  OpenCL: hidden widths up to 64. File handles have neither limit.
- File handles: colour nets are drawn by the reference library on the
  host; OpenCL does not take them; OpenGL needs 4.3.
- Compute capability 5.0 or later for CUDA; Vulkan 1.1 for Vulkan; OpenCL
  1.2 for OpenCL.
- The OpenCL kernel is compiled at every open where the driver keeps no
  program cache, as Arm's do not.

## Licence

MIT. See [LICENSE](LICENSE).
