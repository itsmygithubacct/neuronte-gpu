# Build the GPU backend and benchmark.
#
#   make NEURONTE_DIR=/path/to/neuronte
#
# NEURONTE_DIR must contain include/neuronte.h and build/libneuronte.a (the
# reference CPU library, used for the CPU path and the fallback).
#
# CUDA kernels are PTX, made by clang's NVPTX target (no CUDA toolkit needed)
# or by nvcc (PTXCC=nvcc); CUDA=0 builds without them. Vulkan kernels are
# SPIR-V, made by glslc; VULKAN=0 builds without the Vulkan backend. The
# driver libraries (libcuda.so.1, libvulkan.so.1) are loaded at run time with
# dlopen, so the binary also runs, on the CPU path, where neither exists.

CC ?= cc
CFLAGS ?= -O2
NEURONTE_DIR ?= ../neuronte
CUDA ?= 1
VULKAN ?= 1
PTXCC ?= clang
PTX_ARCH ?= sm_50
GLSLC ?= glslc
GLSLFLAGS ?=
PYTHON ?= python3
# hidden widths with a CUDA kernel (the Vulkan kernel takes any width up to 64)
HIDDEN ?= 1 2 3 4 6 8 12 16 24 32 48 64

WARN = -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
# No contraction: the layer-1 tables are computed exactly as the reference library does.
BASE = -std=c11 -ffp-contract=off
CPPFLAGS += -Isrc -I$(NEURONTE_DIR)/include
LDLIBS += $(NEURONTE_DIR)/build/libneuronte.a -ldl -lm -lpthread

PTX = $(if $(filter 1,$(CUDA)),$(foreach h,$(HIDDEN),build/cover_h$(h).ptx))
SPV = $(if $(filter 1,$(VULKAN)),build/cover_exact1.spv build/cover_exact0.spv)
VK_OBJ = $(if $(filter 1,$(VULKAN)),build/backend_vulkan.o,build/backend_none.o)
OBJS = build/neuronte_gpu.o build/backend_cuda.o $(VK_OBJ) build/kernels.o

all: build/libneuronte_gpu.a build/neuronte_gpu_bench

build:
	mkdir -p build

# Rebuilds the kernels when a build variable that shapes them changes.
CONFIG = $(HIDDEN) | $(CUDA) $(PTXCC) $(PTX_ARCH) | $(VULKAN) $(GLSLFLAGS)
build/config.stamp: FORCE | build
	@echo '$(CONFIG)' | cmp -s - $@ || echo '$(CONFIG)' > $@

ifeq ($(PTXCC),nvcc)
build/cover_h%.ptx: kernels/cover.cu build/config.stamp | build
	nvcc -ptx -arch=compute_$(subst sm_,,$(PTX_ARCH)) -O3 -DNEURONTE_H=$* -o $@ $<
else
build/cover_h%.ptx: kernels/cover.cu build/config.stamp | build
	$(PTXCC) -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=$(PTX_ARCH) -O3 \
	    -DNEURONTE_H=$* -S -o $@ $<
endif

build/cover_exact%.spv: kernels/cover.comp build/config.stamp | build
	$(GLSLC) -fshader-stage=compute --target-env=vulkan1.1 -O $(GLSLFLAGS) -DEXACT=$* -o $@ $<

build/kernels.c: tools/embed.py $(PTX) $(SPV) build/config.stamp | build
	$(PYTHON) tools/embed.py $@ --ptx $(foreach h,$(if $(PTX),$(HIDDEN)),$(h)=build/cover_h$(h).ptx) \
	    --spirv $(if $(SPV),1=build/cover_exact1.spv 0=build/cover_exact0.spv)

build/%.o: src/%.c src/neuronte_gpu.h src/internal.h | build
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) $(WARN) -c -o $@ $<

build/kernels.o: build/kernels.c src/internal.h
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) -c -o $@ $<

build/libneuronte_gpu.a: $(OBJS)
	rm -f $@
	$(AR) rcs $@ $^

build/neuronte_gpu_bench: bench/neuronte_gpu_bench.c build/libneuronte_gpu.a | build
	$(CC) $(CPPFLAGS) $(BASE) $(CFLAGS) $(WARN) -o $@ $< build/libneuronte_gpu.a $(LDLIBS)

clean:
	rm -rf build

.PHONY: all clean FORCE
.SECONDARY: $(PTX) $(SPV)
