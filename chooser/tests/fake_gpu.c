/* A stand-in for libneuronte_gpu, for the chooser's tests: the same calls,
 * answered by the reference library, with switches in the environment that
 * make it misbehave the ways a real GPU can. It keeps each glyph it has
 * drawn, so a second request is a copy: that is what makes it "faster than
 * the CPU" where a test wants the GPU chosen.
 *
 *   FAKE_GPU_LOG=FILE        append a line for every open, call and close
 *   FAKE_GPU_OPEN_MS=N       opening takes N milliseconds
 *   FAKE_GPU_HANG=1          opening never returns
 *   FAKE_GPU_REFUSE=NAME     opening fails for the backend NAME (cuda, vulkan, all)
 *   FAKE_GPU_CRASH=1         the first call aborts the process
 *   FAKE_GPU_WRONG_FROM=N    calls of N glyphs or more get one value wrong by one bit
 *   FAKE_GPU_FAIL_AT=N       the Nth call in the process fails, and every call after it
 *   FAKE_GPU_SLOW_US=N       every glyph costs N microseconds more
 *   FAKE_GPU_ABI=N           the interface number to report
 *   FAKE_GPU_TYPE=TEXT       the device type to report (default "gpu") */
#define _GNU_SOURCE
#include "neuronte.h"
#include "neuronte_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

struct neuronte_gpu {
    neuronte *cpu;
    int cw, ch, backend;
    size_t plane;
    uint32_t *key; /* glyphs drawn so far, open addressing */
    float *val;
    int slots;
};

static int calls;

static long env_long(const char *name, long dflt)
{
    const char *v = getenv(name);
    return v && *v ? atol(v) : dflt;
}

static void note(const char *fmt, long a, long b, long c)
{
    const char *path = getenv("FAKE_GPU_LOG");
    FILE *f;
    if (!path || !*path)
        return;
    f = fopen(path, "a");
    if (!f)
        return;
    fprintf(f, fmt, a, b, c);
    fputc('\n', f);
    fclose(f);
}

static void nap(long us)
{
    struct timespec t = {us / 1000000, (us % 1000000) * 1000};
    nanosleep(&t, NULL);
}

int neuronte_gpu_abi(void)
{
    return (int)env_long("FAKE_GPU_ABI", NEURONTE_GPU_ABI);
}

int neuronte_gpu_open(const struct neuronte_gpu_options *opt, neuronte_gpu **out, char *reason, size_t reason_len)
{
    const char *refuse = getenv("FAKE_GPU_REFUSE");
    const char *name = opt->backend == NEURONTE_GPU_CUDA ? "cuda" : opt->backend == NEURONTE_GPU_VULKAN ? "vulkan" : "other";
    neuronte_gpu *g;
    *out = NULL;
    note("open %ldx%ld backend %ld", opt->cell_w, opt->cell_h, opt->backend);
    if (getenv("FAKE_GPU_HANG"))
        for (;;)
            nap(1000000);
    nap(env_long("FAKE_GPU_OPEN_MS", 0) * 1000);
    if (refuse && (!strcmp(refuse, name) || !strcmp(refuse, "all"))) {
        snprintf(reason, reason_len, "the stand-in has no %s device", name);
        return -1;
    }
    g = calloc(1, sizeof *g);
    if (!g || neuronte_open_file(opt->nrt_path, &g->cpu) != NEURONTE_OK) {
        snprintf(reason, reason_len, "the stand-in cannot open %s", opt->nrt_path);
        free(g);
        return -1;
    }
    g->cw = opt->cell_w;
    g->ch = opt->cell_h;
    g->backend = opt->backend;
    g->plane = (size_t)g->cw * g->ch;
    g->slots = 1 << 14;
    g->key = calloc((size_t)g->slots, sizeof *g->key);
    g->val = malloc((size_t)g->slots * g->plane * sizeof *g->val);
    *out = g;
    return 0;
}

void neuronte_gpu_close(neuronte_gpu *g)
{
    if (!g)
        return;
    note("close %ld%.0ld%.0ld", g->backend, 0, 0);
    neuronte_close(g->cpu);
    free(g->key);
    free(g->val);
    free(g);
}

int neuronte_gpu_coverage(neuronte_gpu *g, const uint32_t *scalars, int count, float *out, struct neuronte_gpu_timing *timing)
{
    long fail_at = env_long("FAKE_GPU_FAIL_AT", 0), wrong_from = env_long("FAKE_GPU_WRONG_FROM", 0);
    int i;
    (void)timing;
    calls++;
    note("coverage %ld glyphs, call %ld%.0ld", count, calls, 0);
    if (getenv("FAKE_GPU_CRASH"))
        abort();
    if (fail_at && calls >= fail_at)
        return -1;
    for (i = 0; i < count; i++) {
        uint32_t s = scalars[i], slot = (s * 2654435761u) & (uint32_t)(g->slots - 1);
        float *dst = out + (size_t)i * g->plane;
        int tries = 0;
        while (g->key[slot] && g->key[slot] != s + 1 && tries++ < g->slots)
            slot = (slot + 1) & (uint32_t)(g->slots - 1);
        if (g->key[slot] == s + 1) {
            memcpy(dst, g->val + (size_t)slot * g->plane, g->plane * sizeof *dst);
        } else {
            if (neuronte_coverage(g->cpu, s, g->cw, g->ch, 0, dst) != NEURONTE_OK)
                return -1;
            if (!g->key[slot]) {
                g->key[slot] = s + 1;
                memcpy(g->val + (size_t)slot * g->plane, dst, g->plane * sizeof *dst);
            }
        }
    }
    if (wrong_from && count >= wrong_from) {
        /* one bit of one value of the last glyph */
        uint32_t bits;
        float *v = out + (size_t)(count - 1) * g->plane + g->plane / 2;
        memcpy(&bits, v, sizeof bits);
        bits ^= 1;
        memcpy(v, &bits, sizeof bits);
    }
    nap(env_long("FAKE_GPU_SLOW_US", 0) * count);
    return 0;
}

int neuronte_gpu_backend(const neuronte_gpu *g)
{
    return g ? g->backend : NEURONTE_GPU_CPU;
}

const char *neuronte_gpu_backend_name(int backend)
{
    return backend == NEURONTE_GPU_CUDA ? "gpu-cuda" : backend == NEURONTE_GPU_VULKAN ? "gpu-vulkan" : "gpu-other";
}

const char *neuronte_gpu_device(const neuronte_gpu *g)
{
    (void)g;
    return "Stand-in GPU";
}

const char *neuronte_gpu_device_type(const neuronte_gpu *g)
{
    const char *t = getenv("FAKE_GPU_TYPE");
    (void)g;
    return t && *t ? t : "gpu";
}
