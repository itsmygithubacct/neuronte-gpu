# The backend chooser

Neuronte draws a cell with a small net. The reference library evaluates the
net on the CPU; `libneuronte_gpu` evaluates the same net on a GPU and, in
its exact mode, gives the same bytes. Which of the two is faster depends on
the machine and on how many glyphs a call asks for. The chooser measures
that once, at setup, and a program then asks one library for coverage and
gets it from the faster side.

Three pieces, all in this directory:

| Piece | File | What it is |
|---|---|---|
| Setup steps `10-backend` and `40-backend-warm` | `neuronte-backend-step.c` | One program under two names, run by `neuronte-setup` |
| The loader, `libneuronte-backend.so.0` | `neuronte_backend.c`, `neuronte_backend.h` | What a program links: six calls |
| `neuronte-backend` | `neuronte-backend.c` | Prints the choice; tries a file through the loader |

`common.h` and `decide.h` are shared by them; `decide.h` holds the rule
itself and has its own test.

## What is promised

- **The same bytes.** Step 10 compares every GPU backend with the CPU
  library on 18 calls per file and size (calls of 1 to 8 glyphs, 8 random
  subsets, the whole table twice). One differing byte and that backend is
  never used on that machine.
- **Nothing waits for a GPU.** The loader opens it on a thread of its own.
  Until it is open the CPU draws, and a call that is under way hands the
  GPU what is left.
- **No call fails because of a GPU.** No GPU, no GPU library, no
  `backend.conf`, a stale one, a driver that changed, a device that failed
  in the middle of a call: the CPU draws, and the call returns what the
  reference library would have returned.
- **On battery the CPU draws.** That is the default; `battery=N` allows the
  GPU above N percent and `battery=same` ignores the power supply.
- **A GPU is only measured in a child process with a time limit**, so a
  driver that hangs or crashes at setup costs that backend and nothing else.

## What "fastest" means

Step 10 times one CPU thread and each GPU backend on calls of 1, 16, 64 and
256 glyphs and on the whole table, at each cell size in `backend.cells`.
A GPU backend is timed in three rounds; in each round every call size is
timed on the CPU and then on the GPU with the same calls, so that other work
on the machine falls on both. A backend is used from the smallest call size
at which, and above which, it is at least `backend.margin` (1.3) times as
fast as one CPU thread on every file and size measured, in every one of the
three rounds. That size is `min_glyphs`. A size that is in doubt therefore
stays with the CPU. `backend.conf` records, for each size, the speed-up
between the medians (`times_cpu`) and the least of the three rounds
(`worst_round`), which is the one the choice is made on. A GPU that also draws the
desktop is held to `backend.desktop_margin` (2) and to whole tables only.
The first open, where a driver compiles kernels, is not counted against a
GPU: step 40 pays it at setup.

Where the reference tree has cells files, the timed calls are made of the
characters those files do not hold (`backend.covered=latin1`: everything
outside ASCII and Latin-1), since a cells file answers the others without a
net. Every character is still compared. They are left out at every size, not
only the sizes a cells file holds, because step 10 runs before the step that
writes the cells files.

A loaded machine makes the CPU look slower than it is, and with it the GPU
better than it is: `backend.conf` records the load average before and after
the measuring (`measure.load`, `measure.load_after`). `neuronte-setup` runs
the step again only when the driver, the GPU library, the GPUs, the kernel,
the measured files or the settings change, never by itself; a choice made
under heavy load stays until then, or until `neuronte-setup --force`.

## What was measured

Two machines, 2026-11-17, `neuronte-split-v9`, microseconds per glyph, both
on mains power. Full files: the `measure.*` lines of each `backend.conf`.

| | Core i7-9850H, Quadro RTX 3000 | Core i7-6820HQ, Quadro M2000M |
|---|---|---|
| Chosen | CUDA, every call (`min_glyphs=1`) | CUDA, whole tables only (`min_glyphs=1152`) |
| Compared calls identical | 36 of 36 | 36 of 36 |
| One glyph, 8x16: CPU / GPU | 320 / 122 | 203 / 320 |
| Whole table, 8x16: CPU / 4 threads / GPU | 553 / 337 / 57 | 358 / 175 / 222 |
| Whole table, 16x32: CPU / 4 threads / GPU | 2,271 / 1,245 / 200 | 1,258 / 674 / 889 |
| Least speed-up of three rounds: one glyph / whole table | 2.6 / 10.3 | 0.6 / 1.5 |
| Step 10 took | 60.9 s (load 10.7; two backends) | 30.9 s (load 0.4; one backend) |

The first machine was busy with other work when it was measured (load 10.7);
at load 7.9 an hour earlier the same step gave 269 / 102 for one glyph and
569 / 263 / 32 for a whole table at 8x16. The choice was the same.

The second machine is close to the margin: at 16 to 256 glyphs its GPU is
1.2 to 1.4 times one CPU thread, and four CPU threads beat it on a whole
table. Before the three rounds its choice moved between runs (whole tables
in one, 16 glyphs and more in the next); with them, three runs in a row
chose whole tables only.

## The two power settings

A discrete laptop GPU does not go back to sleep while a context is open on
it. Measured on the Quadro RTX 3000 (idle 5.8 W): after any call it stayed at
about 30 W for 17 s. Energy above idle over the 30 s after the call:

| | keep open | close after 250 ms | 1 s | 3 s | 1 s and wake after 50 ms |
|---|---|---|---|---|---|
| 3 glyphs | 423 J | 47 J | 59 to 64 J | 111 to 116 J | 0.9 J |
| 16 glyphs | 418 J | 39 J | 52 to 77 J | 99 to 104 J | 1.1 J |
| Whole table (1,341) | 432 J | 52 J | 59 to 67 J | 112 to 122 J | 60 J |
| Typing: 3 glyphs every 0.5 s for 10 s | 652 J | 288 J | 273 to 277 J | 328 to 334 J | 1.2 J |

- `backend.idle_close_ms=1000`: close the GPU after a second without a call.
  It trades a reopen (a whole table on a closed GPU takes 0.21 to 0.34 s, the
  CPU drawing meanwhile, against 0.05 to 0.10 s on an open one) for about
  370 J per burst of work. 250 ms costs less per call but reopens the GPU for
  every call of a typing burst; 0 keeps it open.
- `backend.wake_ms=50`: open a closed GPU only once the CPU has spent 50 ms
  on glyphs that were for it. It trades about 70 ms on the first whole table
  (0.34 s against 0.27 s) for never waking the GPU for a few glyphs: 1 J
  against about 60 J for a three-glyph call. The CPU's time on such glyphs is
  forgotten at 5% a second, so steady typing does not add up. 0 opens the GPU
  at the first call for it.

Both are written to `backend.conf` and read by the loader from there.

## Using the loader

```c
#include "neuronte_backend.h"

neuronte_backend *b;
if (neuronte_backend_open("/usr/share/neuronte/weights/NAME.nrt", 9, 18, &b) == 0) {
    uint32_t scalars[3] = { 0x2022, 0x25CF, 0x41 };
    float out[3 * 9 * 18];
    neuronte_backend_coverage(b, scalars, 3, out);   /* what neuronte_coverage() writes */
    neuronte_backend_close(b);
}
```

For each scalar of a call: the cells file first, when setup made one (no
computing at all); then the GPU, when a GPU is in use and the call has
`min_glyphs` or more scalars left; else the CPU.

- `neuronte_backend_info()` says what the handle is doing: `gpu`, `ready`
  (the GPU is open now), `reason`, and how many glyphs came from the cells
  file, the CPU and the GPU.
- `neuronte_backend_choice()` gives the choice alone and opens nothing.
- `neuronte_backend_expect(b, count)` is a hint for a caller that draws a
  job a few glyphs at a time: its parts are judged as the job. Never
  required; without it the pixels are the same.

A handle is for one thread at a time. `NEURONTE_BACKEND=cpu` in the
environment, or `backend=cpu` in the system's or the user's `setup.conf`,
keeps a program on the CPU. Nothing forces a GPU that setup did not choose.

"Warming": each user's driver keeps its own cache of compiled kernels (one
written by root cannot serve other users). Until that user's
`40-backend-warm` has run, the loader draws on the CPU, and once per process
asks `neuronte-setup-user.service` to run it.

## The steps

Both follow `neuronte-setup`'s protocol: `STEP check` exits 0 when current
and 1 when not, and writes nothing; `STEP run` makes it current.

- `10-backend` writes `STATE/backend.conf` and `10-backend.stamp`. It is
  current while its own program, the reference library, the GPU library, the
  settings, the measured files, the GPUs (PCI ids), the drivers and the
  kernel are what the stamp names. With `backend=cpu` or no GPU library it
  settles at once without measuring. Under `NEURONTE_SETUP_BOOT=1` it writes
  "the CPU, for now", no stamp, and exits 75 (deferred).
- `40-backend-warm` opens every weights file once on the chosen backend so
  that the driver compiles and caches its kernels, and writes
  `40-backend-warm.stamp`. With `NEURONTE_SETUP_SCOPE=user` it reads
  `backend.conf` from `NEURONTE_SYSTEM_STATE` and writes its stamp in the
  user's own state. Its `check` never touches a GPU.

Settings are listed at the top of `neuronte-backend-step.c`.
`backend.conf` is key=value, the last setting of a key wins; a file of
another version, for another GPU library or driver, or not marked identical
means the CPU.

## Build, test, install

```
make NEURONTE_DIR=/path/to/neuronte chooser
make NEURONTE_DIR=/path/to/neuronte test-chooser
make NEURONTE_DIR=/path/to/neuronte CHOOSER_SETUP=/path/to/neuronte-setup test-chooser-sanitized
make NEURONTE_DIR=/path/to/neuronte PREFIX=/usr install-chooser
```

`NEURONTE_DIR` is a built Neuronte tree (`include/neuronte.h`,
`build/libneuronte.a`). The tests need no GPU and no driver: a stand-in GPU
library (`tests/fake_gpu.c`) can be told to be slow, to hang, to crash, to
refuse, to give a wrong byte or to fail at the Nth call. With a tree that
has cells files, the loader is also tested against a real one made by that
tree's `neuronte-cells`; `tests/fake_cells.c` stands in where the tree has
none. `test-chooser-sanitized` runs the loader's cases again under the
address, undefined-behaviour and thread sanitizers.

`tools/hold.py` measures the energy table above on a machine with an NVIDIA
GPU.

## Not proven

- Suspend and resume with a GPU handle open. The device-lost path is tested
  with the stand-in, not with a machine that slept.
- OpenCL as a chooser backend (the library has one; `backend.try` does not
  list it by default and it was not measured here).
- Colour nets: the GPU library draws them on the host, so the chooser
  measures distance nets only.
- A machine whose desktop is drawn by the GPU being measured: the rule is
  tested, the measurement was made on machines where it is not.

## Where it belongs

The loader and `neuronte-backend` belong with `libneuronte_gpu`, in the
neuronte-gpu repository: they are the only code that needs
`neuronte_gpu.h`, and they change when the GPU library's interface does
(`NEURONTE_GPU_ABI`). The two steps can ship from either side; they are
installed into Neuronte's `setup.d` and `user-setup.d` as optional files,
and Neuronte's own documentation should describe `neuronte-setup`, the
settings and `backend.conf`.
