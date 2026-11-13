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
    free(m);
}

struct push {
    int32_t nglyph, cell_w, cell_h, samples;
};

static int impl_run(void *p, const uint32_t *scalars, int count, float *out, struct neuronte_gpu_timing *t)
{
    struct impl *m = p;
    struct vk *vk = &m->vk;
    neuronte_gpu *g = m->g;
    size_t plane = (size_t)g->cell_w * g->cell_h, bytes = (size_t)count * plane * sizeof(float);
    struct push pc = {count, g->cell_w, g->cell_h, g->samples};
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
    vk->CmdDispatch(m->cmd, (uint32_t)g->ntiles, (uint32_t)((count + g->gpb - 1) / g->gpb), 1);
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

int neuronte_gpu_vulkan_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len)
{
    struct impl *m = calloc(1, sizeof *m);
    struct vk *vk;
    const struct embedded_spirv *spv = NULL;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    VkPhysicalDevice devs[16];
    VkPhysicalDeviceProperties props, best_props;
    uint32_t ndev = 16, i, nfam = 0;
    int best = -1, best_rank = -1, rc;
    VkQueueFamilyProperties fams[32];
    float prio = 1.0f;
    size_t plane = (size_t)g->cell_w * g->cell_h;
    char driver[256] = "", denorm[64] = "";
    const int wshared = (opt->flags & NEURONTE_GPU_SHARED_WEIGHTS) != 0;
    int shared_bytes;

    if (!m)
        return -1;
    neuronte_gpu_set_block(g, opt, 64); /* best on the Intel GPUs measured; 256 and up lose occupancy */
    shared_bytes = (int)sizeof(float) * (wshared ? g->pk.shared_floats : 1 + g->gpb * g->hidden);
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
        snprintf(driver, sizeof driver, "%s %s", dp.driverName, dp.driverInfo);
        snprintf(denorm, sizeof denorm, "fp32 denorm preserve %d, flush %d", fc.shaderDenormPreserveFloat32,
                 fc.shaderDenormFlushToZeroFloat32);
    }
    if ((size_t)shared_bytes > props.limits.maxComputeSharedMemorySize ||
        (uint32_t)g->block > props.limits.maxComputeWorkGroupInvocations ||
        (uint32_t)g->block > props.limits.maxComputeWorkGroupSize[0]) {
        neuronte_gpu_say(why, len, "%s: needs %d shared bytes and %d invocations per workgroup; allows %u and %u",
                         props.deviceName, shared_bytes, g->block,
                         props.limits.maxComputeSharedMemorySize, props.limits.maxComputeWorkGroupInvocations);
        goto fail;
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
            qp.queryCount = 2;
            if (vk->CreateQueryPool(m->dev, &qp, NULL, &m->qpool))
                m->qpool = VK_NULL_HANDLE;
        }
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
        int32_t spec_vals[4] = {g->block, g->hidden, g->gpb, wshared};
        VkSpecializationMapEntry ents[4] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}};
        VkSpecializationInfo spec = {4, ents, sizeof spec_vals, spec_vals};
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
