# Build the GPU backend and benchmark.
#
#   make NEURONTE_DIR=/path/to/neuronte
#
# NEURONTE_DIR must contain include/neuronte.h and build/libneuronte.a (the
# reference CPU library, used for the CPU path and the fallback).
#
# CUDA kernels are PTX, made by clang's NVPTX target (no CUDA toolkit needed)
# or by nvcc (PTXCC=nvcc); CUDA=0 builds without them. Vulkan kernels are
# SPIR-V, made by glslc; VULKAN=0 builds without the Vulkan backend, which is
# the default when the Vulkan headers are not found (VULKAN_INCLUDE=DIR points
# at a copy; only the headers are needed to build). The OpenCL kernel is
# embedded as source and compiled by the driver at open; the backend needs no
# headers or tools. The driver libraries (libcuda.so.1, libvulkan.so.1,
# libOpenCL.so.1) are loaded at run time with dlopen, so the binary also runs,
# on the CPU path, where none exists.

CC ?= cc
CFLAGS ?= -O2
NEURONTE_DIR ?= ../neuronte
CUDA ?= 1
VULKAN_INCLUDE ?=
ifeq ($(origin VULKAN),undefined)
VULKAN := $(shell printf '\043include <vulkan/vulkan.h>\n' | \
    $(CC) $(if $(VULKAN_INCLUDE),-I$(VULKAN_INCLUDE)) -E -x c - >/dev/null 2>&1 && echo 1 || echo 0)
endif
PTXCC ?= clang
PTX_ARCH ?= sm_50
GLSLC ?= glslc
GLSLFLAGS ?=
PYTHON ?= python3
# hidden widths with a CUDA kernel (the Vulkan kernel takes any width up to 64)
HIDDEN ?= 1 2 3 4 6 8 12 16 24 32 48 64
# File handles (bundles, tiled nets, drawing files, wide nets): narrow hidden
# widths with a CUDA kernel of their own, and the accumulator counts R of the
# wide kernel, which draws hidden widths 65 to 32 R.
HIDDEN2 ?= 1 2 3 4 6 8 12 16 20 24 32 48 56 64
WIDE2 ?= 3 4 5 6 8 10 12 16 24 32

WARN = -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
# No contraction: the layer-1 tables are computed exactly as the reference library does.
BASE = -std=c11 -ffp-contract=off
CPPFLAGS += -Isrc -I$(NEURONTE_DIR)/include $(if $(VULKAN_INCLUDE),-I$(VULKAN_INCLUDE))
LDLIBS += $(NEURONTE_DIR)/build/libneuronte.a -ldl -lm -lpthread

PTX = $(if $(filter 1,$(CUDA)),$(foreach h,$(HIDDEN),build/cover_h$(h).ptx))
PTX2 = $(if $(filter 1,$(CUDA)),$(foreach h,$(HIDDEN2),build/cover2_h$(h).ptx) \
    $(foreach r,$(WIDE2),build/cover2_r$(r).ptx))
SPV = $(if $(filter 1,$(VULKAN)),build/cover_exact1.spv build/cover_exact0.spv \
    build/cover2_exact1.spv build/cover2_exact0.spv build/reduce.spv \
    build/cover2_gl_exact1.spv build/cover2_gl_exact0.spv)
VK_OBJ = $(if $(filter 1,$(VULKAN)),build/backend_vulkan.o,build/backend_none.o)
OBJS = build/neuronte_gpu.o build/neuronte_gpu_v2.o build/backend_cuda.o $(VK_OBJ) build/backend_opencl.o build/backend_gl.o build/kernels.o

all: build/libneuronte_gpu.a build/neuronte_gpu_bench build/neuronte_gpu_text build/neuronte_gpu_mem

build:
	mkdir -p build

# Rebuilds the kernels when a build variable that shapes them changes.
CONFIG = $(HIDDEN) | $(HIDDEN2) | $(WIDE2) | $(CUDA) $(PTXCC) $(PTX_ARCH) | $(VULKAN) $(GLSLFLAGS)
build/config.stamp: FORCE | build
	@echo '$(CONFIG)' | cmp -s - $@ || echo '$(CONFIG)' > $@

ifeq ($(PTXCC),nvcc)
build/cover_h%.ptx: kernels/cover.cu build/config.stamp | build
	nvcc -ptx -arch=compute_$(subst sm_,,$(PTX_ARCH)) -O3 -DNEURONTE_H=$* -o $@ $<
build/cover2_h%.ptx: kernels/cover2.cu build/config.stamp | build
	nvcc -ptx -arch=compute_$(subst sm_,,$(PTX_ARCH)) -O3 -DNEURONTE_H=$* -o $@ $<
build/cover2_r%.ptx: kernels/cover2.cu build/config.stamp | build
	nvcc -ptx -arch=compute_$(subst sm_,,$(PTX_ARCH)) -O3 -DNEURONTE_R=$* -o $@ $<
else
build/cover2_h%.ptx: kernels/cover2.cu build/config.stamp | build
	$(PTXCC) -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=$(PTX_ARCH) -O3 \
	    -DNEURONTE_H=$* -S -o $@ $<
build/cover2_r%.ptx: kernels/cover2.cu build/config.stamp | build
	$(PTXCC) -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=$(PTX_ARCH) -O3 \
	    -DNEURONTE_R=$* -S -o $@ $<
build/cover_h%.ptx: kernels/cover.cu build/config.stamp | build
	$(PTXCC) -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=$(PTX_ARCH) -O3 \
	    -DNEURONTE_H=$* -S -o $@ $<
endif

build/cover_exact%.spv: kernels/cover.comp build/config.stamp | build
	$(GLSLC) -fshader-stage=compute --target-env=vulkan1.1 -O $(GLSLFLAGS) -DEXACT=$* -o $@ $<
build/cover2_exact%.spv: kernels/cover2.comp build/config.stamp | build
	$(GLSLC) -fshader-stage=compute --target-env=vulkan1.1 -std=450core -O $(GLSLFLAGS) -DEXACT=$* -o $@ $<
# the same shader for OpenGL 4.6, which loads SPIR-V too: uniforms in place of push constants
build/cover2_gl_exact%.spv: kernels/cover2.comp build/config.stamp | build
	$(GLSLC) -fshader-stage=compute --target-env=opengl -std=450core -O $(GLSLFLAGS) -DGL=1 -DSPV_GL=1 -DEXACT=$* -o $@ $<
build/reduce.spv: kernels/reduce.comp build/config.stamp | build
	$(GLSLC) -fshader-stage=compute --target-env=vulkan1.1 -std=450core -O $(GLSLFLAGS) -o $@ $<

build/kernels.c: tools/embed.py $(PTX) $(PTX2) $(SPV) kernels/cover.cl kernels/cover2.comp kernels/reduce.comp \
    build/config.stamp | build
	$(PYTHON) tools/embed.py $@ --ptx $(foreach h,$(if $(PTX),$(HIDDEN)),$(h)=build/cover_h$(h).ptx) \
	    --ptx2 $(foreach h,$(if $(PTX2),$(HIDDEN2)),$(h)=build/cover2_h$(h).ptx) \
	    $(foreach r,$(if $(PTX2),$(WIDE2)),$(shell expr 1000 + $(r))=build/cover2_r$(r).ptx) \
	    --spirv $(if $(SPV),1=build/cover_exact1.spv 0=build/cover_exact0.spv 11=build/cover2_exact1.spv \
	    10=build/cover2_exact0.spv 20=build/reduce.spv \
	    31=build/cover2_gl_exact1.spv 30=build/cover2_gl_exact0.spv) \
	    --glsl cover2=kernels/cover2.comp reduce=kernels/reduce.comp --opencl kernels/cover.cl

build/%.o: src/%.c src/neuronte_gpu.h src/internal.h src/clmini.h | build
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) $(WARN) -c -o $@ $<

build/kernels.o: build/kernels.c src/internal.h
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) -c -o $@ $<

build/libneuronte_gpu.a: $(OBJS)
	rm -f $@
	$(AR) rcs $@ $^

# A shared library with the reference library inside, for programs that load it at run time
# (a terminal's glyph hook through a foreign-function interface): make build/libneuronte_gpu.so
build/pic:
	mkdir -p build/pic
build/pic/%.o: src/%.c src/neuronte_gpu.h src/internal.h src/clmini.h | build/pic
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) $(WARN) -fPIC -c -o $@ $<
build/pic/kernels.o: build/kernels.c src/internal.h | build/pic
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) -fPIC -c -o $@ $<
build/libneuronte_gpu.so: $(patsubst build/%.o,build/pic/%.o,$(OBJS))
	$(CC) -shared -o $@ $^ $(LDLIBS)

build/neuronte_gpu_bench: bench/neuronte_gpu_bench.c build/libneuronte_gpu.a | build
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) $(WARN) -o $@ $< build/libneuronte_gpu.a $(LDLIBS)

build/neuronte_gpu_text: bench/neuronte_gpu_text.c build/libneuronte_gpu.a | build
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) $(WARN) -o $@ $< build/libneuronte_gpu.a $(LDLIBS)

build/neuronte_gpu_mem: bench/neuronte_gpu_mem.c build/libneuronte_gpu.a | build
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) $(WARN) -o $@ $< build/libneuronte_gpu.a $(LDLIBS)

# The atlas comparison stands in for a terminal that draws with OpenGL; it needs the EGL and
# OpenGL headers and libraries, so it is built only when asked for: make build/neuronte_gpu_atlas
build/neuronte_gpu_atlas: bench/neuronte_gpu_atlas.c build/libneuronte_gpu.a | build
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) $(WARN) -o $@ $< build/libneuronte_gpu.a $(LDLIBS) -lEGL -lOpenGL

clean:
	rm -rf build

.PHONY: all clean FORCE
.SECONDARY: $(PTX) $(PTX2) $(SPV)
