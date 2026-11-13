/* The OpenCL 1.2 declarations the OpenCL backend uses, resolved at run time from the ICD loader
 * with dlopen, so the library builds without OpenCL headers and runs where OpenCL is absent. */
#ifndef NEURONTE_GPU_CLMINI_H
#define NEURONTE_GPU_CLMINI_H

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_ulong cl_bitfield;
typedef cl_uint cl_bool;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_bitfield cl_device_fp_config;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_uint cl_program_build_info;
typedef cl_uint cl_kernel_work_group_info;
typedef cl_uint cl_profiling_info;
typedef intptr_t cl_context_properties;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;

#define CL_SUCCESS 0
#define CL_FALSE 0
#define CL_TRUE 1
#define CL_DEVICE_TYPE_CPU (1u << 1)
#define CL_DEVICE_TYPE_GPU (1u << 2)
#define CL_DEVICE_TYPE_ACCELERATOR (1u << 3)
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFFu

#define CL_PLATFORM_VERSION 0x0901
#define CL_PLATFORM_NAME 0x0902

#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define CL_DEVICE_SINGLE_FP_CONFIG 0x101B
#define CL_DEVICE_LOCAL_MEM_TYPE 0x1022
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1023
#define CL_DEVICE_NAME 0x102B
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#define CL_DEVICE_OPENCL_C_VERSION 0x103D

#define CL_FP_DENORM (1u << 0)
#define CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT (1u << 7)
#define CL_LOCAL 0x1
#define CL_GLOBAL 0x2

#define CL_MEM_READ_WRITE (1u << 0)
#define CL_MEM_WRITE_ONLY (1u << 1)
#define CL_MEM_READ_ONLY (1u << 2)
#define CL_MEM_COPY_HOST_PTR (1u << 5)
#define CL_QUEUE_PROFILING_ENABLE (1u << 1)

#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_KERNEL_WORK_GROUP_SIZE 0x11B0
#define CL_KERNEL_LOCAL_MEM_SIZE 0x11B2
#define CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 0x11B3
#define CL_KERNEL_PRIVATE_MEM_SIZE 0x11B4
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283

typedef void (*clmini_notify)(const char *, const void *, size_t, void *);

#define CLMINI_FUNCS(X)                                                                            \
    X(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id *, cl_uint *))                           \
    X(cl_int, clGetPlatformInfo, (cl_platform_id, cl_platform_info, size_t, void *, size_t *))    \
    X(cl_int, clGetDeviceIDs, (cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *)) \
    X(cl_int, clGetDeviceInfo, (cl_device_id, cl_device_info, size_t, void *, size_t *))          \
    X(cl_context, clCreateContext,                                                                 \
      (const cl_context_properties *, cl_uint, const cl_device_id *, clmini_notify, void *, cl_int *)) \
    X(cl_command_queue, clCreateCommandQueue,                                                      \
      (cl_context, cl_device_id, cl_command_queue_properties, cl_int *))                           \
    X(cl_mem, clCreateBuffer, (cl_context, cl_mem_flags, size_t, void *, cl_int *))               \
    X(cl_program, clCreateProgramWithSource,                                                       \
      (cl_context, cl_uint, const char **, const size_t *, cl_int *))                              \
    X(cl_int, clBuildProgram,                                                                      \
      (cl_program, cl_uint, const cl_device_id *, const char *, void (*)(cl_program, void *), void *)) \
    X(cl_int, clGetProgramBuildInfo,                                                               \
      (cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *))                 \
    X(cl_kernel, clCreateKernel, (cl_program, const char *, cl_int *))                            \
    X(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void *))                         \
    X(cl_int, clGetKernelWorkGroupInfo,                                                            \
      (cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *))              \
    X(cl_int, clEnqueueNDRangeKernel,                                                              \
      (cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *,       \
       cl_uint, const cl_event *, cl_event *))                                                     \
    X(cl_int, clEnqueueWriteBuffer,                                                                \
      (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, \
       cl_event *))                                                                                \
    X(cl_int, clEnqueueReadBuffer,                                                                 \
      (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *,       \
       cl_event *))                                                                                \
    X(cl_int, clFlush, (cl_command_queue))                                                        \
    X(cl_int, clFinish, (cl_command_queue))                                                       \
    X(cl_int, clWaitForEvents, (cl_uint, const cl_event *))                                       \
    X(cl_int, clGetEventProfilingInfo, (cl_event, cl_profiling_info, size_t, void *, size_t *))   \
    X(cl_int, clReleaseEvent, (cl_event))                                                         \
    X(cl_int, clReleaseMemObject, (cl_mem))                                                       \
    X(cl_int, clReleaseKernel, (cl_kernel))                                                       \
    X(cl_int, clReleaseProgram, (cl_program))                                                     \
    X(cl_int, clReleaseCommandQueue, (cl_command_queue))                                          \
    X(cl_int, clReleaseContext, (cl_context))

struct clmini {
    void *handle;
#define CLMINI_MEMBER(ret, name, args) ret(*name) args;
    CLMINI_FUNCS(CLMINI_MEMBER)
#undef CLMINI_MEMBER
};

/* Returns 0 on success; on failure writes a reason (the handle, if any, stays open for the
 * caller to close). */
static inline int clmini_load(struct clmini *a, const char *lib, char *err, size_t errlen)
{
    a->handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!a->handle) {
        snprintf(err, errlen, "cannot load %s: %s", lib, dlerror());
        return -1;
    }
#define CLMINI_LOAD(ret, name, args)                                                               \
    *(void **)(&a->name) = dlsym(a->handle, #name);                                                \
    if (!a->name) {                                                                                \
        snprintf(err, errlen, "%s has no %s", lib, #name);                                         \
        return -1;                                                                                 \
    }
    CLMINI_FUNCS(CLMINI_LOAD)
#undef CLMINI_LOAD
    return 0;
}

#endif
