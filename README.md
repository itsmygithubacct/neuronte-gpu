# neuronte-gpu

A GPU implementation of [Neuronte](https://github.com/itsmygithubacct/neuronte),
with a benchmark that compares it against Neuronte's standard CPU
implementation.

Neuronte is a small neural network whose weights are a font. Its C library
evaluates the network on the CPU to draw glyphs into a pixel buffer. This
project runs the same network on a GPU, with CUDA on NVIDIA GPUs and with
Vulkan compute on GPUs without CUDA, and measures how the two compare. The
goal is a GPU path that is faster than the CPU library on the same machine.

Development happens on the `work/rc9-gpu` branch, for the RC9 release.

## Licence

MIT. See [LICENSE](LICENSE).
