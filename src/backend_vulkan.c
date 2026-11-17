/* neuronte-gpu Vulkan compute backend, through the Vulkan loader opened at
 * run time. Works on any GPU with a Vulkan 1.1 driver. A device that is a CPU
 * implementation (such as lavapipe) is refused unless the caller allows it,
 * and is then reported as a CPU device. */
#define _GNU_SOURCE
#define VK_NO_PROTOTYPES
#include "internal.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define NBUF 5
#define NBUF2 8 /* file handles: params, segs, masks, pairs, out, blocks, samples, list */

struct vk {
    void *dl;
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    PFN_vkCreateInstance CreateInstance;
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;
    PFN_vkCreateDevice CreateDevice;
    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges;
    PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges;
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkDestroyPipeline DestroyPipeline;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyCommandPool DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkResetCommandBuffer ResetCommandBuffer;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdPushConstants CmdPushConstants;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkCmdCopyBuffer CmdCopyBuffer;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdResetQueryPool CmdResetQueryPool;
    PFN_vkCmdWriteTimestamp CmdWriteTimestamp;
    PFN_vkCreateQueryPool CreateQueryPool;
    PFN_vkDestroyQueryPool DestroyQueryPool;
    PFN_vkGetQueryPoolResults GetQueryPoolResults;
    PFN_vkCreateFence CreateFence;
    PFN_vkDestroyFence DestroyFence;
    PFN_vkResetFences ResetFences;
    PFN_vkWaitForFences WaitForFences;
    PFN_vkQueueSubmit QueueSubmit;
};

struct buf {
    VkBuffer b;
    VkDeviceMemory m;
    void *map;
    VkDeviceSize size;
    int coherent;
};

struct impl {
    struct vk vk;
    neuronte_gpu *g;
    VkInstance inst;
    VkPhysicalDevice pd;
    VkPhysicalDeviceMemoryProperties mem;
    VkDevice dev;
    VkQueue queue;
    uint32_t family;
    struct buf bufs[NBUF]; /* params, tiles, masks, scalars, out */
    VkShaderModule shader;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    VkCommandPool cpool;
    VkCommandBuffer cmd;
    VkFence fence;
    VkQueryPool qpool;
    double ts_period; /* ns per timestamp tick; 0 = no timestamps */
    uint64_t ts_mask;
    /* file handles */
    struct buf b2[NBUF2];
    VkShaderModule shader2;
    VkPipeline *pipes, reduce; /* one per kernel of the plan */
    uint32_t max_groups;       /* workgroups one dispatch may have along x */
    double *rate;              /* per kernel: multiply-adds per nanosecond, as last measured */
    double *work;              /* per kernel: multiply-adds of the call in hand */
    uint64_t *ts;
    double slice_ns;           /* the longest one dispatch is meant to run */
    uint32_t nquery;
};

enum { B_PARAMS, B_TILES, B_MASKS, B_SCALARS, B_OUT };

static int load_api(struct vk *vk, char *why, size_t len)
{
    static const char *names[] = {"libvulkan.so.1", "libvulkan.so"};
    size_t i;
    for (i = 0; i < sizeof names / sizeof *names && !vk->dl; i++)
        vk->dl = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!vk->dl) {
        neuronte_gpu_say(why, len, "no Vulkan loader (libvulkan.so.1)");
        return -1;
    }
    *(void **)&vk->GetInstanceProcAddr = dlsym(vk->dl, "vkGetInstanceProcAddr");
    if (!vk->GetInstanceProcAddr) {
        neuronte_gpu_say(why, len, "the Vulkan loader lacks vkGetInstanceProcAddr");
        return -1;
    }
    vk->CreateInstance = (PFN_vkCreateInstance)vk->GetInstanceProcAddr(NULL, "vkCreateInstance");
    if (!vk->CreateInstance) {
        neuronte_gpu_say(why, len, "the Vulkan loader lacks vkCreateInstance");
        return -1;
    }
    return 0;
}

static int load_instance(struct vk *vk, VkInstance inst, char *why, size_t len)
{
#define I(name)                                                                     \
    do {                                                                            \
        *(void **)&vk->name = (void *)vk->GetInstanceProcAddr(inst, "vk" #name);    \
        if (!vk->name) {                                                            \
            neuronte_gpu_say(why, len, "Vulkan lacks vk%s", #name);                 \
            return -1;                                                              \
        }                                                                           \
    } while (0)
    I(DestroyInstance);
    I(EnumeratePhysicalDevices);
    I(GetPhysicalDeviceProperties);
    I(GetPhysicalDeviceProperties2);
    I(GetPhysicalDeviceQueueFamilyProperties);
    I(GetPhysicalDeviceMemoryProperties);
    I(GetDeviceProcAddr);
    I(CreateDevice);
#undef I
    return 0;
}

static int load_device(struct vk *vk, VkDevice dev, char *why, size_t len)
{
#define D(name)                                                                     \
    do {                                                                            \
        *(void **)&vk->name = (void *)vk->GetDeviceProcAddr(dev, "vk" #name);       \
        if (!vk->name) {                                                            \
            neuronte_gpu_say(why, len, "Vulkan device lacks vk%s", #name);          \
            return -1;                                                              \
        }                                                                           \
    } while (0)
    D(DestroyDevice);
    D(GetDeviceQueue);
    D(CreateBuffer);
    D(DestroyBuffer);
    D(GetBufferMemoryRequirements);
    D(AllocateMemory);
    D(FreeMemory);
    D(BindBufferMemory);
    D(MapMemory);
    D(UnmapMemory);
    D(InvalidateMappedMemoryRanges);
    D(FlushMappedMemoryRanges);
    D(CreateShaderModule);
    D(DestroyShaderModule);
    D(CreateDescriptorSetLayout);
    D(DestroyDescriptorSetLayout);
    D(CreatePipelineLayout);
    D(DestroyPipelineLayout);
    D(CreateComputePipelines);
    D(DestroyPipeline);
    D(CreateDescriptorPool);
    D(DestroyDescriptorPool);
    D(AllocateDescriptorSets);
    D(UpdateDescriptorSets);
    D(CreateCommandPool);
    D(DestroyCommandPool);
    D(AllocateCommandBuffers);
    D(BeginCommandBuffer);
    D(EndCommandBuffer);
    D(ResetCommandBuffer);
    D(CmdBindPipeline);
    D(CmdBindDescriptorSets);
    D(CmdPushConstants);
    D(CmdDispatch);
    D(CmdCopyBuffer);
    D(CmdPipelineBarrier);
    D(CmdResetQueryPool);
    D(CmdWriteTimestamp);
    D(CreateQueryPool);
    D(DestroyQueryPool);
    D(GetQueryPoolResults);
    D(CreateFence);
    D(DestroyFence);
    D(ResetFences);
    D(WaitForFences);
    D(QueueSubmit);
#undef D
    return 0;
}

/* The first memory type allowed by `bits` with all of `want`, preferring one
 * that also has `prefer`. */
static int mem_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits, VkMemoryPropertyFlags want,
                    VkMemoryPropertyFlags prefer)
{
    uint32_t i;
    int pass;
    for (pass = 0; pass < 2; pass++)
        for (i = 0; i < mp->memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = mp->memoryTypes[i].propertyFlags;
            if (!(bits & (1u << i)) || (f & want) != want)
                continue;
            if (pass == 0 && (f & prefer) != prefer)
                continue;
            return (int)i;
        }
    return -1;
}

static int make_buf(struct impl *m, struct buf *b, VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags want, VkMemoryPropertyFlags prefer)
{
    struct vk *vk = &m->vk;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    VkMemoryRequirements req;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    int type;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vk->CreateBuffer(m->dev, &bi, NULL, &b->b))
        return -1;
    vk->GetBufferMemoryRequirements(m->dev, b->b, &req);
    type = mem_type(&m->mem, req.memoryTypeBits, want, prefer);
    if (type < 0)
        return -1;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = (uint32_t)type;
    if (vk->AllocateMemory(m->dev, &ai, NULL, &b->m) || vk->BindBufferMemory(m->dev, b->b, b->m, 0))
        return -1;
    b->size = size;
    b->coherent = !!(m->mem.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (want & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        if (vk->MapMemory(m->dev, b->m, 0, VK_WHOLE_SIZE, 0, &b->map))
            return -1;
    return 0;
}

static void free_buf(struct impl *m, struct buf *b)
{
    if (b->map)
        m->vk.UnmapMemory(m->dev, b->m);
    if (b->b)
        m->vk.DestroyBuffer(m->dev, b->b, NULL);
    if (b->m)
        m->vk.FreeMemory(m->dev, b->m, NULL);
    memset(b, 0, sizeof *b);
}

static int submit_wait(struct impl *m)
{
    struct vk *vk = &m->vk;
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m->cmd;
    if (vk->ResetFences(m->dev, 1, &m->fence) || vk->QueueSubmit(m->queue, 1, &si, m->fence))
        return -1;
    return vk->WaitForFences(m->dev, 1, &m->fence, VK_TRUE, UINT64_MAX) ? -1 : 0;
}

/* Copy host data into a device-local buffer through a staging buffer. */
static int upload(struct impl *m, struct buf *dst, const void *data, size_t size)
{
    struct vk *vk = &m->vk;
    struct buf st = {0};
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VkBufferCopy cp = {0, 0, size};
    int rc = -1;
    if (dst->map) {
        memcpy(dst->map, data, size);
        if (!dst->coherent) {
            VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, dst->m, 0, VK_WHOLE_SIZE};
            vk->FlushMappedMemoryRanges(m->dev, 1, &r);
        }
        return 0;
    }
    if (make_buf(m, &st, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0))
        goto out;
    memcpy(st.map, data, size);
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vk->ResetCommandBuffer(m->cmd, 0) || vk->BeginCommandBuffer(m->cmd, &bi))
        goto out;
    vk->CmdCopyBuffer(m->cmd, st.b, dst->b, 1, &cp);
    if (vk->EndCommandBuffer(m->cmd) || submit_wait(m))
        goto out;
    rc = 0;
out:
    free_buf(m, &st);
    return rc;
}

static void impl_close(void *p)
{
    struct impl *m = p;
    struct vk *vk;
    int i;
    if (!m)
        return;
    vk = &m->vk;
    if (m->dev) {
        for (i = 0; i < NBUF; i++)
            free_buf(m, &m->bufs[i]);
        for (i = 0; i < NBUF2; i++)
            free_buf(m, &m->b2[i]);
        for (i = 0; m->pipes && i < m->g->plan.nkern; i++)
            if (m->pipes[i])
                vk->DestroyPipeline(m->dev, m->pipes[i], NULL);
        if (m->reduce)
            vk->DestroyPipeline(m->dev, m->reduce, NULL);
        if (m->shader2)
            vk->DestroyShaderModule(m->dev, m->shader2, NULL);
        if (m->qpool)
            vk->DestroyQueryPool(m->dev, m->qpool, NULL);
        if (m->fence)
            vk->DestroyFence(m->dev, m->fence, NULL);
        if (m->cpool)
            vk->DestroyCommandPool(m->dev, m->cpool, NULL);
        if (m->pool)
            vk->DestroyDescriptorPool(m->dev, m->pool, NULL);
        if (m->pipe)
            vk->DestroyPipeline(m->dev, m->pipe, NULL);
        if (m->layout)
            vk->DestroyPipelineLayout(m->dev, m->layout, NULL);
        if (m->dsl)
            vk->DestroyDescriptorSetLayout(m->dev, m->dsl, NULL);
        if (m->shader)
            vk->DestroyShaderModule(m->dev, m->shader, NULL);
        vk->DestroyDevice(m->dev, NULL);
    }
    if (m->inst)
        vk->DestroyInstance(m->inst, NULL);
    if (vk->dl)
        dlclose(vk->dl);
    free(m->pipes);
    free(m->rate);
    free(m->work);
    free(m->ts);
    free(m);
}

struct push {
    int32_t nglyph, cell_w, cell_h, samples, per_sample;
};

static int impl_run(void *p, const uint32_t *scalars, int count, float *out, struct neuronte_gpu_timing *t)
{
    struct impl *m = p;
    struct vk *vk = &m->vk;
    neuronte_gpu *g = m->g;
    size_t plane = (size_t)g->cell_w * g->cell_h, bytes = (size_t)count * plane * sizeof(float);
    int groups = (count + g->gpb - 1) / g->gpb, nz = neuronte_gpu_splits(g, g->ntiles * groups);
    struct push pc = {count, g->cell_w, g->cell_h, g->samples, nz > 1};
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    uint64_t t0, t1, t2, ts[2];
    struct buf *sb = &m->bufs[B_SCALARS], *ob = &m->bufs[B_OUT];

    if (count <= 0)
        return 0;
    t0 = neuronte_gpu_now();
    memcpy(sb->map, scalars, (size_t)count * sizeof *scalars);
    if (!sb->coherent) {
        VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, sb->m, 0, VK_WHOLE_SIZE};
        vk->FlushMappedMemoryRanges(m->dev, 1, &r);
    }
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vk->ResetCommandBuffer(m->cmd, 0) || vk->BeginCommandBuffer(m->cmd, &bi))
        return -1;
    if (m->qpool) {
        vk->CmdResetQueryPool(m->cmd, m->qpool, 0, 2);
        vk->CmdWriteTimestamp(m->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m->qpool, 0);
    }
    vk->CmdBindPipeline(m->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m->pipe);
    vk->CmdBindDescriptorSets(m->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m->layout, 0, 1, &m->set, 0, NULL);
    vk->CmdPushConstants(m->cmd, m->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vk->CmdDispatch(m->cmd, (uint32_t)g->ntiles, (uint32_t)groups, (uint32_t)nz);
    if (m->qpool)
        vk->CmdWriteTimestamp(m->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m->qpool, 1);
    {
        /* make the shader's writes visible to the host */
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT,
                              VK_ACCESS_HOST_READ_BIT};
        vk->CmdPipelineBarrier(m->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb,
                               0, NULL, 0, NULL);
    }
    if (vk->EndCommandBuffer(m->cmd))
        return -1;
    t1 = neuronte_gpu_now();
    if (submit_wait(m))
        return -1;
    t2 = neuronte_gpu_now();
    if (!ob->coherent) {
        VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, ob->m, 0, VK_WHOLE_SIZE};
        vk->InvalidateMappedMemoryRanges(m->dev, 1, &r);
    }
    memcpy(out, ob->map, bytes);
    if (m->qpool && vk->GetQueryPoolResults(m->dev, m->qpool, 0, 2, sizeof ts, ts, sizeof *ts,
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
        t->run += (uint64_t)((double)((ts[1] - ts[0]) & m->ts_mask) * m->ts_period);
    t->upload += t1 - t0;
    t->wait += t2 - t1;
    t->download += neuronte_gpu_now() - t2;
    t->launches++;
    return 0;
}

static const struct backend_ops ops = {impl_close, impl_run};

static int type_rank(VkPhysicalDeviceType type)
{
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_OTHER: return 1;
    default: return 0; /* CPU */
    }
}

/* What open_device() found out about the device. */
struct devinfo {
    VkPhysicalDeviceProperties props;
    int rank;
    char driver[2 * VK_MAX_DRIVER_NAME_SIZE + 2], denorm[64];
};

/* Instance, device, queue, command buffer, fence and timestamp pool. */
static int open_device(struct impl *m, const struct neuronte_gpu_options *opt, struct devinfo *di_out, char *why,
                       size_t len)
{
    struct vk *vk = &m->vk;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    VkPhysicalDevice devs[16];
    VkPhysicalDeviceProperties props, best_props;
    uint32_t ndev = 16, i, nfam = 0;
    int best = -1, best_rank = -1, rc;
    VkQueueFamilyProperties fams[32];
    float prio = 1.0f;
    char *driver = di_out->driver, *denorm = di_out->denorm;

    driver[0] = denorm[0] = 0;
    if (load_api(vk, why, len))
        goto fail;
    app.pApplicationName = "neuronte-gpu";
    app.apiVersion = VK_API_VERSION_1_1;
    ici.pApplicationInfo = &app;
    if ((rc = vk->CreateInstance(&ici, NULL, &m->inst)) != VK_SUCCESS) {
        neuronte_gpu_say(why, len, "vkCreateInstance failed (%d): no Vulkan driver?", rc);
        goto fail;
    }
    if (load_instance(vk, m->inst, why, len))
        goto fail;
    if (vk->EnumeratePhysicalDevices(m->inst, &ndev, devs) < 0 || ndev == 0) {
        neuronte_gpu_say(why, len, "no Vulkan device");
        goto fail;
    }
    for (i = 0; i < ndev; i++) {
        int rank;
        vk->GetPhysicalDeviceProperties(devs[i], &props);
        rank = type_rank(props.deviceType);
        if (opt->device && !strcasestr(props.deviceName, opt->device))
            continue;
        if (rank == 0 && !(opt->flags & NEURONTE_GPU_ALLOW_CPU_DEVICE))
            continue;
        if (rank > best_rank) {
            best = (int)i;
            best_rank = rank;
            best_props = props;
        }
    }
    if (best < 0) {
        neuronte_gpu_say(why, len, "no Vulkan GPU%s%s%s (CPU devices are refused unless allowed)",
                         opt->device ? " matching \"" : "", opt->device ? opt->device : "", opt->device ? "\"" : "");
        goto fail;
    }
    m->pd = devs[best];
    props = best_props;
    if (VK_API_VERSION_MINOR(props.apiVersion) >= 2 || VK_API_VERSION_MAJOR(props.apiVersion) > 1) {
        VkPhysicalDeviceDriverProperties dp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceFloatControlsProperties fc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &dp;
        dp.pNext = &fc;
        vk->GetPhysicalDeviceProperties2(m->pd, &p2);
        snprintf(driver, sizeof di_out->driver, "%s %s", dp.driverName, dp.driverInfo);
        snprintf(denorm, sizeof di_out->denorm, "fp32 denorm preserve %d, flush %d", fc.shaderDenormPreserveFloat32,
                 fc.shaderDenormFlushToZeroFloat32);
    }
    vk->GetPhysicalDeviceQueueFamilyProperties(m->pd, &nfam, NULL);
    if (nfam > 32)
        nfam = 32;
    vk->GetPhysicalDeviceQueueFamilyProperties(m->pd, &nfam, fams);
    for (i = 0; i < nfam; i++)
        if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            break;
    if (i == nfam) {
        neuronte_gpu_say(why, len, "%s has no compute queue", props.deviceName);
        goto fail;
    }
    m->family = i;
    if (fams[i].timestampValidBits && props.limits.timestampPeriod > 0) {
        m->ts_period = props.limits.timestampPeriod;
        m->ts_mask = fams[i].timestampValidBits >= 64 ? ~0ull : (1ull << fams[i].timestampValidBits) - 1;
    }
    vk->GetPhysicalDeviceMemoryProperties(m->pd, &m->mem);
    {
        VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        qi.queueFamilyIndex = m->family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        if ((rc = vk->CreateDevice(m->pd, &di, NULL, &m->dev)) != VK_SUCCESS) {
            neuronte_gpu_say(why, len, "vkCreateDevice failed (%d)", rc);
            goto fail;
        }
    }
    if (load_device(vk, m->dev, why, len))
        goto fail;
    vk->GetDeviceQueue(m->dev, m->family, 0, &m->queue);
    {
        VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = m->family;
        if (vk->CreateCommandPool(m->dev, &pi, NULL, &m->cpool))
            goto vkfail;
        ai.commandPool = m->cpool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vk->AllocateCommandBuffers(m->dev, &ai, &m->cmd) || vk->CreateFence(m->dev, &fi, NULL, &m->fence))
            goto vkfail;
        if (m->ts_period) {
            VkQueryPoolCreateInfo qp = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qp.queryCount = m->nquery ? m->nquery : 2;
            if (vk->CreateQueryPool(m->dev, &qp, NULL, &m->qpool))
                m->qpool = VK_NULL_HANDLE;
        }
    }
    di_out->props = props;
    di_out->rank = best_rank;
    return 0;
vkfail:
    neuronte_gpu_say(why, len, "Vulkan setup failed");
fail:
    return -1;
}

int neuronte_gpu_vulkan_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len)
{
    struct impl *m = calloc(1, sizeof *m);
    struct vk *vk;
    const struct embedded_spirv *spv = NULL;
    struct devinfo info;
    VkPhysicalDeviceProperties props;
    uint32_t i;
    int best_rank, rc;
    size_t plane = (size_t)g->cell_w * g->cell_h;
    const char *driver = info.driver, *denorm = info.denorm;
    const int wshared = (opt->flags & NEURONTE_GPU_SHARED_WEIGHTS) != 0;
    int shared_bytes;

    if (!m)
        return -1;
    neuronte_gpu_set_block(g, opt, 64); /* best on the Intel GPUs measured; 256 and up lose occupancy */
    g->fill_blocks = 32; /* core Vulkan does not report compute units; enough for small GPUs */
    shared_bytes = (int)sizeof(float) * (wshared ? g->pk.shared_floats : 1 + g->gpb * g->hidden + g->block);
    m->g = g;
    vk = &m->vk;
    for (i = 0; i < (uint32_t)neuronte_gpu_spirv_count; i++)
        if (neuronte_gpu_spirv[i].exact == g->exact)
            spv = &neuronte_gpu_spirv[i];
    if (!spv) {
        neuronte_gpu_say(why, len, "no Vulkan kernel built");
        goto fail;
    }
    if (g->hidden > 64) {
        neuronte_gpu_say(why, len, "the Vulkan kernel supports hidden widths up to 64, not %d", g->hidden);
        goto fail;
    }
    if (open_device(m, opt, &info, why, len))
        goto fail;
    props = info.props;
    best_rank = info.rank;
    if ((size_t)shared_bytes > props.limits.maxComputeSharedMemorySize ||
        (uint32_t)g->block > props.limits.maxComputeWorkGroupInvocations ||
        (uint32_t)g->block > props.limits.maxComputeWorkGroupSize[0]) {
        neuronte_gpu_say(why, len, "%s: needs %d shared bytes and %d invocations per workgroup; allows %u and %u",
                         props.deviceName, shared_bytes, g->block,
                         props.limits.maxComputeSharedMemorySize, props.limits.maxComputeWorkGroupInvocations);
        goto fail;
    }
    {
        const VkMemoryPropertyFlags DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        const VkBufferUsageFlags SB = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (make_buf(m, &m->bufs[B_PARAMS], g->pk.nparams * sizeof(float), SB, DL, 0) ||
            make_buf(m, &m->bufs[B_TILES], sizeof g->pk.tiles, SB, DL, 0) ||
            make_buf(m, &m->bufs[B_MASKS], sizeof g->pk.masks, SB, DL, 0) ||
            make_buf(m, &m->bufs[B_SCALARS], (size_t)g->max_batch * sizeof(uint32_t), SB, HV,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
            make_buf(m, &m->bufs[B_OUT], (size_t)g->max_batch * plane * sizeof(float), SB, HV,
                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ||
            upload(m, &m->bufs[B_PARAMS], g->pk.params, g->pk.nparams * sizeof(float)) ||
            upload(m, &m->bufs[B_TILES], g->pk.tiles, sizeof g->pk.tiles) ||
            upload(m, &m->bufs[B_MASKS], g->pk.masks, sizeof g->pk.masks)) {
            neuronte_gpu_say(why, len, "allocating or uploading buffers failed");
            goto fail;
        }
    }
    {
        VkShaderModuleCreateInfo si = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        VkDescriptorSetLayoutBinding binds[NBUF];
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(struct push)};
        VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        int32_t spec_vals[5] = {g->block, g->hidden, g->gpb, wshared, g->block};
        VkSpecializationMapEntry ents[5] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {4, 16, 4}};
        VkSpecializationInfo spec = {5, ents, sizeof spec_vals, spec_vals};
        VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NBUF};
        VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        VkDescriptorBufferInfo infos[NBUF];
        VkWriteDescriptorSet writes[NBUF];

        si.codeSize = (size_t)spv->words * 4;
        si.pCode = spv->code;
        if (vk->CreateShaderModule(m->dev, &si, NULL, &m->shader))
            goto vkfail;
        for (i = 0; i < NBUF; i++) {
            binds[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                      VK_SHADER_STAGE_COMPUTE_BIT, NULL};
            infos[i] = (VkDescriptorBufferInfo){m->bufs[i].b, 0, VK_WHOLE_SIZE};
        }
        li.bindingCount = NBUF;
        li.pBindings = binds;
        if (vk->CreateDescriptorSetLayout(m->dev, &li, NULL, &m->dsl))
            goto vkfail;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &m->dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        if (vk->CreatePipelineLayout(m->dev, &pli, NULL, &m->layout))
            goto vkfail;
        cpi.stage = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                                                      VK_SHADER_STAGE_COMPUTE_BIT, m->shader, "main", &spec};
        cpi.layout = m->layout;
        if ((rc = vk->CreateComputePipelines(m->dev, VK_NULL_HANDLE, 1, &cpi, NULL, &m->pipe)) != VK_SUCCESS) {
            neuronte_gpu_say(why, len, "creating the compute pipeline failed (%d)", rc);
            goto fail;
        }
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        if (vk->CreateDescriptorPool(m->dev, &dpi, NULL, &m->pool))
            goto vkfail;
        dai.descriptorPool = m->pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &m->dsl;
        if (vk->AllocateDescriptorSets(m->dev, &dai, &m->set))
            goto vkfail;
        for (i = 0; i < NBUF; i++)
            writes[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, m->set, i, 0, 1,
                                               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &infos[i], NULL};
        vk->UpdateDescriptorSets(m->dev, NBUF, writes, 0, NULL);
    }
    snprintf(g->device, sizeof g->device, "%s", props.deviceName);
    snprintf(g->versions, sizeof g->versions,
             "vulkan %u.%u.%u; driver %s (0x%x); %s; weights from %s; %d shared bytes per workgroup; timestamps %s",
             VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
             VK_API_VERSION_PATCH(props.apiVersion), driver[0] ? driver : "?", props.driverVersion,
             denorm[0] ? denorm : "float controls unknown", wshared ? "shared memory" : "the parameter buffer",
             shared_bytes, m->qpool ? "yes" : "no");
    g->device_type = best_rank == 0 ? "cpu" : best_rank == 3 ? "integrated-gpu" : "gpu";
    g->ops = &ops;
    g->impl = m;
    return 0;
vkfail:
    neuronte_gpu_say(why, len, "Vulkan setup failed");
fail:
    impl_close(m);
    return -1;
}

/* ---- file handles ---- */

enum { F_PARAMS, F_SEGS, F_MASKS, F_PAIRS, F_OUT, F_BLOCKS, F_SAMP, F_LIST };

struct push2 {
    int32_t a, cell_w, cell_h, samples, first;
};

static void bind_buf(struct impl *m, uint32_t binding, struct buf *b)
{
    VkDescriptorBufferInfo info = {b->b, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, m->set, binding, 0, 1,
                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &info, NULL};
    m->vk.UpdateDescriptorSets(m->dev, 1, &w, 0, NULL);
}

static void flush(struct impl *m, struct buf *b)
{
    if (!b->coherent) {
        VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, b->m, 0, VK_WHOLE_SIZE};
        m->vk.FlushMappedMemoryRanges(m->dev, 1, &r);
    }
}

/* A mapped buffer of at least `bytes`, grown when a call needs more. Nothing is in flight here. */
static int fit(struct impl *m, int which, size_t bytes)
{
    struct buf *b = &m->b2[which];
    size_t want = b->size ? (size_t)b->size : 16384;
    if (bytes <= b->size)
        return 0;
    while (want < bytes)
        want *= 2;
    m->g->device_bytes -= (size_t)b->size;
    free_buf(m, b);
    if (make_buf(m, b, want, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        return -1;
    m->g->device_bytes += want;
    bind_buf(m, (uint32_t)which, b);
    return 0;
}

static int impl_run_plan(void *p, int count, float *out, struct neuronte_gpu_timing *t)
{
    struct impl *m = p;
    struct vk *vk = &m->vk;
    neuronte_gpu *g = m->g;
    struct v2plan *pl = &g->plan;
    size_t plane = (size_t)g->cell_w * g->cell_h, bytes = (size_t)count * plane * sizeof(float);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VkMemoryBarrier host = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_HOST_READ_BIT};
    struct push2 pc = {0, g->cell_w, g->cell_h, g->samples, 0};
    struct buf *ob = &m->b2[F_OUT];
    uint64_t t0, t1, t2;
    int k;

    t0 = neuronte_gpu_now();
    if (fit(m, F_PAIRS, (size_t)pl->npairs * 2 * sizeof(uint32_t)) ||
        fit(m, F_BLOCKS, (size_t)pl->nblocks * B_INTS * sizeof(int32_t)))
        return -1;
    memcpy(m->b2[F_PAIRS].map, pl->pairs, (size_t)pl->npairs * 2 * sizeof(uint32_t));
    flush(m, &m->b2[F_PAIRS]);
    memcpy(m->b2[F_BLOCKS].map, pl->blocks, (size_t)pl->nblocks * B_INTS * sizeof(int32_t));
    flush(m, &m->b2[F_BLOCKS]);
    if (pl->nreduce) {
        memcpy(m->b2[F_LIST].map, pl->reduce, (size_t)pl->nreduce * sizeof(uint32_t));
        flush(m, &m->b2[F_LIST]);
    }
    if (pl->dynamic) {
        memcpy((int32_t *)m->b2[F_SEGS].map + (size_t)pl->nstatic_segs * S_INTS,
               pl->segs + (size_t)pl->nstatic_segs * S_INTS,
               (size_t)(pl->nsegs - pl->nstatic_segs) * S_INTS * sizeof(int32_t));
        flush(m, &m->b2[F_SEGS]);
        memcpy((float *)m->b2[F_PARAMS].map + pl->nstatic, pl->params + pl->nstatic,
               (pl->nparams - pl->nstatic) * sizeof(float));
        flush(m, &m->b2[F_PARAMS]);
    }
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vk->ResetCommandBuffer(m->cmd, 0) || vk->BeginCommandBuffer(m->cmd, &bi))
        return -1;
    if (m->qpool) {
        vk->CmdResetQueryPool(m->cmd, m->qpool, 0, m->nquery);
        vk->CmdWriteTimestamp(m->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m->qpool, 0);
    }
    vk->CmdBindDescriptorSets(m->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m->layout, 0, 1, &m->set, 0, NULL);
    for (k = 0; k < pl->nkern; k++) {
        const struct v2kern *kn = &pl->kern[k];
        int done;
        m->work[k] = 0;
        if (kn->nblocks) {
            vk->CmdBindPipeline(m->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m->pipes[k]);
            for (done = 0; done < kn->nblocks;) {
                /* One dispatch cannot be preempted on some GPUs (Intel's: the desktop waits for it, and
                 * the kernel resets a context that holds the GPU too long), so a dispatch takes only
                 * as many blocks as should finish within the slice, by the rate last measured. */
                double budget = m->rate[k] * m->slice_ns, cost = 0;
                uint32_t n = 0;
                while (done + (int)n < kn->nblocks && n < m->max_groups) {
                    const int32_t *b = pl->blocks + (size_t)(kn->block0 + done + (int)n) * B_INTS;
                    double hid = pl->segs[(size_t)b[B_SEG] * S_INTS + S_HID];
                    double c = (double)(b[B_HI] - b[B_LO]) * (b[B_MODE] < BM_SAMPLE ? g->samples * g->samples : 1) *
                               (2 * hid * hid + 3 * hid);
                    if (n && cost + c > budget)
                        break;
                    cost += c;
                    n++;
                }
                pc.a = kn->block0 + done;
                vk->CmdPushConstants(m->cmd, m->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
                vk->CmdDispatch(m->cmd, n, 1, 1);
                done += (int)n;
                m->work[k] += cost;
                t->launches++;
            }
        }
        if (m->qpool)
            vk->CmdWriteTimestamp(m->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m->qpool, (uint32_t)k + 1);
    }
    if (pl->nreduce) {
        /* the sample buffer was written by the dispatches above */
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT,
                              VK_ACCESS_SHADER_READ_BIT};
        size_t per = (size_t)m->max_groups * 64 / plane; /* glyphs one dispatch can take */
        int done;
        vk->CmdPipelineBarrier(m->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               1, &mb, 0, NULL, 0, NULL);
        vk->CmdBindPipeline(m->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m->reduce);
        if (per < 1)
            return -1;
        for (done = 0; done < pl->nreduce;) {
            int n = (size_t)(pl->nreduce - done) < per ? pl->nreduce - done : (int)per;
            pc.a = n;
            pc.first = done;
            vk->CmdPushConstants(m->cmd, m->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
            vk->CmdDispatch(m->cmd, (uint32_t)(((size_t)n * plane + 63) / 64), 1, 1);
            done += n;
            t->launches++;
        }
    }
    if (m->qpool)
        vk->CmdWriteTimestamp(m->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m->qpool, (uint32_t)pl->nkern + 1);
    vk->CmdPipelineBarrier(m->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0,
                           NULL, 0, NULL);
    if (vk->EndCommandBuffer(m->cmd))
        return -1;
    t1 = neuronte_gpu_now();
    if (submit_wait(m))
        return -1;
    t2 = neuronte_gpu_now();
    if (!ob->coherent) {
        VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, ob->m, 0, VK_WHOLE_SIZE};
        vk->InvalidateMappedMemoryRanges(m->dev, 1, &r);
    }
    memcpy(out, ob->map, bytes);
    if (m->qpool && vk->GetQueryPoolResults(m->dev, m->qpool, 0, m->nquery, m->nquery * sizeof *m->ts, m->ts,
                                            sizeof *m->ts, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
                        VK_SUCCESS) {
        t->run += (uint64_t)((double)((m->ts[pl->nkern + 1] - m->ts[0]) & m->ts_mask) * m->ts_period);
        for (k = 0; k < pl->nkern; k++) {
            double ns = (double)((m->ts[k + 1] - m->ts[k]) & m->ts_mask) * m->ts_period;
            /* a launch too short to time well teaches nothing */
            if (m->work[k] > 0 && ns > 200000.0)
                m->rate[k] = m->work[k] / ns;
        }
    }
    t->upload += t1 - t0;
    t->wait += t2 - t1;
    t->download += neuronte_gpu_now() - t2;
    return 0;
}

static const struct backend_ops ops2 = {impl_close, NULL, impl_run_plan};

int neuronte_gpu_vulkan_open2(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len)
{
    struct impl *m = calloc(1, sizeof *m);
    struct vk *vk;
    struct v2plan *pl = &g->plan;
    const struct embedded_spirv *spv = NULL, *rspv = NULL;
    struct devinfo info;
    const VkPhysicalDeviceLimits *lim;
    const VkMemoryPropertyFlags DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkBufferUsageFlags SB = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    size_t plane = (size_t)g->cell_w * g->cell_h, splane = plane * g->samples * g->samples, bytes = 0;
    size_t sizes[NBUF2];
    int i, k, rc, nt = opt->block > 0 ? opt->block : 64, dyn = pl->params_cap > pl->nstatic;
    char kinds[256] = "";

    if (!m)
        return -1;
    m->g = g;
    vk = &m->vk;
    for (i = 0; i < neuronte_gpu_spirv_count; i++) {
        if (neuronte_gpu_spirv[i].exact == 10 + g->exact)
            spv = &neuronte_gpu_spirv[i];
        if (neuronte_gpu_spirv[i].exact == 20)
            rspv = &neuronte_gpu_spirv[i];
    }
    if (!spv || !rspv) {
        neuronte_gpu_say(why, len, "no Vulkan kernel built");
        goto fail;
    }
    m->nquery = (uint32_t)pl->nkern + 2;
    m->rate = calloc((size_t)pl->nkern + 1, sizeof *m->rate);
    m->work = calloc((size_t)pl->nkern + 1, sizeof *m->work);
    m->ts = calloc(m->nquery, sizeof *m->ts);
    if (!m->rate || !m->work || !m->ts)
        goto fail;
    {
        /* Until a launch has been timed, assume a slow device: 2 multiply-adds per nanosecond. */
        const char *tune = getenv("NEURONTE_GPU_SLICE_MS");
        m->slice_ns = (tune && atof(tune) > 0 ? atof(tune) : 25.0) * 1e6;
        for (k = 0; k < pl->nkern; k++)
            m->rate[k] = 2.0;
    }
    if (open_device(m, opt, &info, why, len))
        goto fail;
    lim = &info.props.limits;
    g->fill_blocks = g->fill_wide = 32; /* core Vulkan does not report compute units; enough for small GPUs */
    if ((uint32_t)nt > lim->maxComputeWorkGroupInvocations || (uint32_t)nt > lim->maxComputeWorkGroupSize[0]) {
        neuronte_gpu_say(why, len, "%s allows %u invocations per workgroup, not %d", info.props.deviceName,
                         lim->maxComputeWorkGroupInvocations, nt);
        goto fail;
    }
    if (neuronte_gpu_v2_size(g, nt, nt, nt, 0, lim->maxComputeSharedMemorySize / sizeof(float), why, len))
        goto fail;
    m->max_groups = lim->maxComputeWorkGroupCount[0];
    sizes[F_PARAMS] = pl->params_cap * sizeof(float);
    sizes[F_SEGS] = (size_t)pl->segs_cap * S_INTS * sizeof(int32_t);
    sizes[F_MASKS] = (size_t)pl->nmasks * sizeof(uint32_t);
    sizes[F_PAIRS] = 16384;
    sizes[F_OUT] = (size_t)g->max_batch * plane * sizeof(float);
    sizes[F_BLOCKS] = 16384;
    sizes[F_SAMP] = pl->need_samples ? (size_t)g->max_batch * splane * sizeof(float) : 16;
    sizes[F_LIST] = (size_t)g->max_batch * sizeof(uint32_t);
    for (i = 0; i < NBUF2; i++) {
        /* what a call rewrites is mapped; a file whose letters can be warped rewrites the end of
         * its tables too; the rest lives on the device */
        int mapped = i == F_PAIRS || i == F_BLOCKS || i == F_LIST || i == F_OUT || (dyn && (i == F_PARAMS || i == F_SEGS));
        if (!sizes[i])
            sizes[i] = 16;
        if (make_buf(m, &m->b2[i], sizes[i], SB, mapped ? HV : DL,
                     i == F_OUT ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : mapped ? DL : 0)) {
            neuronte_gpu_say(why, len, "allocating %zu bytes on %s failed", sizes[i], info.props.deviceName);
            goto fail;
        }
        bytes += sizes[i];
    }
    if (upload(m, &m->b2[F_PARAMS], pl->params, pl->nstatic * sizeof(float)) ||
        (pl->nstatic_segs && upload(m, &m->b2[F_SEGS], pl->segs, (size_t)pl->nstatic_segs * S_INTS * sizeof(int32_t))) ||
        (pl->nmasks && upload(m, &m->b2[F_MASKS], pl->masks, (size_t)pl->nmasks * sizeof(uint32_t)))) {
        neuronte_gpu_say(why, len, "uploading the weights failed");
        goto fail;
    }
    {
        VkShaderModuleCreateInfo si = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        VkDescriptorSetLayoutBinding binds[NBUF2];
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(struct push2)};
        VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NBUF2};
        VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};

        si.codeSize = (size_t)spv->words * 4;
        si.pCode = spv->code;
        if (vk->CreateShaderModule(m->dev, &si, NULL, &m->shader))
            goto vkfail;
        si.codeSize = (size_t)rspv->words * 4;
        si.pCode = rspv->code;
        if (vk->CreateShaderModule(m->dev, &si, NULL, &m->shader2))
            goto vkfail;
        for (i = 0; i < NBUF2; i++)
            binds[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                      VK_SHADER_STAGE_COMPUTE_BIT, NULL};
        li.bindingCount = NBUF2;
        li.pBindings = binds;
        if (vk->CreateDescriptorSetLayout(m->dev, &li, NULL, &m->dsl))
            goto vkfail;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &m->dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        if (vk->CreatePipelineLayout(m->dev, &pli, NULL, &m->layout))
            goto vkfail;
        m->pipes = calloc((size_t)pl->nkern + 1, sizeof *m->pipes);
        if (!m->pipes)
            goto vkfail;
        for (k = 0; k < pl->nkern; k++) {
            const struct v2kern *kn = &pl->kern[k];
            /* workgroup size, hidden width (the padded width for a wide kernel), pairs per block, wide */
            int32_t spec_vals[4] = {kn->threads, kn->wide ? kn->n * LANES : kn->n, kn->gp_max, kn->wide};
            VkSpecializationMapEntry ents[4] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}};
            VkSpecializationInfo spec = {4, ents, sizeof spec_vals, spec_vals};
            size_t used = strlen(kinds);
            if ((size_t)kn->gp_max * (size_t)spec_vals[1] * sizeof(float) > lim->maxComputeSharedMemorySize) {
                neuronte_gpu_say(why, len, "%s: hidden width %d needs more shared memory than %u bytes",
                                 info.props.deviceName, spec_vals[1], lim->maxComputeSharedMemorySize);
                goto fail;
            }
            cpi.stage = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL,
                                                          0, VK_SHADER_STAGE_COMPUTE_BIT, m->shader, "main", &spec};
            cpi.layout = m->layout;
            if ((rc = vk->CreateComputePipelines(m->dev, VK_NULL_HANDLE, 1, &cpi, NULL, &m->pipes[k])) != VK_SUCCESS) {
                neuronte_gpu_say(why, len, "creating the compute pipeline for hidden width %d failed (%d)",
                                 spec_vals[1], rc);
                goto fail;
            }
            snprintf(kinds + used, sizeof kinds - used, "%s%s%d", used ? " " : "", kn->wide ? "wide" : "H",
                     spec_vals[1]);
        }
        cpi.stage = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                                                      VK_SHADER_STAGE_COMPUTE_BIT, m->shader2, "main", NULL};
        cpi.layout = m->layout;
        if ((rc = vk->CreateComputePipelines(m->dev, VK_NULL_HANDLE, 1, &cpi, NULL, &m->reduce)) != VK_SUCCESS) {
            neuronte_gpu_say(why, len, "creating the reduction pipeline failed (%d)", rc);
            goto fail;
        }
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        if (vk->CreateDescriptorPool(m->dev, &dpi, NULL, &m->pool))
            goto vkfail;
        dai.descriptorPool = m->pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &m->dsl;
        if (vk->AllocateDescriptorSets(m->dev, &dai, &m->set))
            goto vkfail;
        for (i = 0; i < NBUF2; i++)
            bind_buf(m, (uint32_t)i, &m->b2[i]);
    }
    g->device_bytes = bytes;
    snprintf(g->device, sizeof g->device, "%s", info.props.deviceName);
    snprintf(g->versions, sizeof g->versions,
             "vulkan %u.%u.%u; driver %s (0x%x); %s; timestamps %s; file handle: %d nets, %d segments, kernels %s",
             VK_API_VERSION_MAJOR(info.props.apiVersion), VK_API_VERSION_MINOR(info.props.apiVersion),
             VK_API_VERSION_PATCH(info.props.apiVersion), info.driver[0] ? info.driver : "?",
             info.props.driverVersion, info.denorm[0] ? info.denorm : "float controls unknown",
             m->qpool ? "yes" : "no", g->nmembers, pl->nstatic_segs, kinds);
    g->device_type = info.rank == 0 ? "cpu" : info.rank == 3 ? "integrated-gpu" : "gpu";
    g->ops = &ops2;
    g->impl = m;
    return 0;
vkfail:
    neuronte_gpu_say(why, len, "Vulkan setup failed");
fail:
    impl_close(m);
    return -1;
}
