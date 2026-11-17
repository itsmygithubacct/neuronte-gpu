/* Stand-in for the Vulkan backend in builds without Vulkan headers or a
 * SPIR-V compiler (make VULKAN=0). */
#include "internal.h"

int neuronte_gpu_vulkan_open(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len)
{
    (void)g;
    (void)opt;
    neuronte_gpu_say(why, len, "this build has no Vulkan backend");
    return -1;
}

int neuronte_gpu_vulkan_open2(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len)
{
    (void)g;
    (void)opt;
    neuronte_gpu_say(why, len, "this build has no Vulkan backend");
    return -1;
}
