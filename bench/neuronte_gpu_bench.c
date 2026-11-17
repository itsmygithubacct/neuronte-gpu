/* Benchmark one model set on a GPU backend or the reference CPU path.
 *
 * One process measures one backend: opening (network parse, device and
 * kernel setup), the first full-corpus call, an optional timed warmup (GPU
 * clocks ramp up over tens to hundreds of milliseconds), warmup rounds, measured rounds
 * and optional single-glyph calls. Every round is printed; nothing is
 * discarded. The coverage of the final measured call is written in scalar
 * order so its quality can be checked against teacher masks and the CPU
 * output.
 *
 * Output: one JSON object on stdout. Anything else written to stdout while
 * the bench runs (Rockchip's Mali driver prints its version there when it is
 * loaded) is sent to stderr instead.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "neuronte_gpu.h"

#define MAX_ROUNDS 1000

static uint64_t clock_ns(clockid_t id)
{
    struct timespec ts;
    clock_gettime(id, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static void read_text(const char *path, char *buf, size_t len)
{
    FILE *f = fopen(path, "r");
    size_t n = 0;
    buf[0] = 0;
    if (!f)
        return;
    n = fread(buf, 1, len - 1, f);
    fclose(f);
    buf[n] = 0;
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == ' '))
        buf[--n] = 0;
    for (size_t i = 0; i < n; i++)
        if (buf[i] == '"' || buf[i] == '\\' || buf[i] == '\n')
            buf[i] = ' ';
}

/* NVIDIA state through NVML, when the library is present. */
typedef struct nvmlDevice_st *nvmlDevice_t;
typedef struct {
    unsigned gpu, memory;
} nvml_util;
static struct {
    int tried, ok;
    unsigned count;
    int (*count_fn)(unsigned *);
    int (*handle)(unsigned, nvmlDevice_t *);
    int (*name)(nvmlDevice_t, char *, unsigned);
    int (*temp)(nvmlDevice_t, int, unsigned *);
    int (*clock)(nvmlDevice_t, int, unsigned *);
    int (*power)(nvmlDevice_t, unsigned *);
    int (*util)(nvmlDevice_t, nvml_util *);
    int (*pstate)(nvmlDevice_t, int *);
    int (*energy)(nvmlDevice_t, unsigned long long *);
    int (*procs)(nvmlDevice_t, unsigned *, void *);
} nv;

/* nvmlProcessInfo_t of nvmlDeviceGetComputeRunningProcesses_v3 */
struct nvml_proc {
    unsigned pid;
    unsigned long long used;
    unsigned gpu_instance, compute_instance;
};

static void nvml_load(void)
{
    void *dl;
    int (*init)(void);
    nv.tried = 1;
    dl = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!dl)
        return;
    *(void **)&init = dlsym(dl, "nvmlInit_v2");
    *(void **)&nv.count_fn = dlsym(dl, "nvmlDeviceGetCount_v2");
    *(void **)&nv.handle = dlsym(dl, "nvmlDeviceGetHandleByIndex_v2");
    *(void **)&nv.name = dlsym(dl, "nvmlDeviceGetName");
    *(void **)&nv.temp = dlsym(dl, "nvmlDeviceGetTemperature");
    *(void **)&nv.clock = dlsym(dl, "nvmlDeviceGetClockInfo");
    *(void **)&nv.power = dlsym(dl, "nvmlDeviceGetPowerUsage");
    *(void **)&nv.util = dlsym(dl, "nvmlDeviceGetUtilizationRates");
    *(void **)&nv.pstate = dlsym(dl, "nvmlDeviceGetPerformanceState");
    *(void **)&nv.energy = dlsym(dl, "nvmlDeviceGetTotalEnergyConsumption");
    *(void **)&nv.procs = dlsym(dl, "nvmlDeviceGetComputeRunningProcesses_v3");
    if (init && nv.count_fn && nv.handle && init() == 0 && nv.count_fn(&nv.count) == 0)
        nv.ok = 1;
}

/* A snapshot of frequency, temperature, power and load, as a JSON object. */
static void snapshot(FILE *out, int cpu)
{
    char path[256], v[256];
    int z, c;
    fprintf(out, "{");
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu < 0 ? 0 : cpu);
    read_text(path, v, sizeof v);
    fprintf(out, "\"cpu_khz\":\"%s\"", v);
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpu < 0 ? 0 : cpu);
    read_text(path, v, sizeof v);
    fprintf(out, ",\"cpu_governor\":\"%s\"", v);
    read_text("/proc/loadavg", v, sizeof v);
    fprintf(out, ",\"loadavg\":\"%s\"", v);
    fprintf(out, ",\"drm_gt_mhz\":{");
    for (c = 0, z = 0; c < 8; c++) {
        snprintf(path, sizeof path, "/sys/class/drm/card%d/gt_cur_freq_mhz", c);
        read_text(path, v, sizeof v);
        if (v[0])
            fprintf(out, "%s\"card%d\":\"%s\"", z++ ? "," : "", c, v);
    }
    /* devfreq devices (Mali GPUs, NPUs, memory controllers): current frequency and, where the
       governor reports it, the load as "percent@frequency" */
    fprintf(out, "},\"devfreq\":{");
    {
        DIR *d = opendir("/sys/class/devfreq");
        struct dirent *e;
        z = 0;
        while (d && (e = readdir(d))) {
            char cur[64], load[96];
            if (e->d_name[0] == '.' || strpbrk(e->d_name, "\"\\"))
                continue;
            snprintf(path, sizeof path, "/sys/class/devfreq/%.200s/cur_freq", e->d_name);
            read_text(path, cur, sizeof cur);
            snprintf(path, sizeof path, "/sys/class/devfreq/%.200s/load", e->d_name);
            read_text(path, load, sizeof load);
            fprintf(out, "%s\"%s\":{\"cur_freq\":\"%s\",\"load\":\"%s\"}", z++ ? "," : "", e->d_name, cur, load);
        }
        if (d)
            closedir(d);
    }
    fprintf(out, "},\"thermal\":{");
    for (z = 0; z < 32; z++) {
        char type[64], temp[64];
        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/type", z);
        read_text(path, type, sizeof type);
        if (!type[0])
            break;
        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/temp", z);
        read_text(path, temp, sizeof temp);
        fprintf(out, "%s\"%d:%s\":\"%s\"", z ? "," : "", z, type, temp);
    }
    fprintf(out, "},\"nvml\":[");
    if (!nv.tried)
        nvml_load();
    for (unsigned d = 0; nv.ok && d < nv.count; d++) {
        nvmlDevice_t dev;
        char name[96] = "";
        unsigned temp = 0, sm = 0, mem = 0, mw = 0;
        nvml_util u = {0, 0};
        int ps = -1;
        if (nv.handle(d, &dev))
            continue;
        if (nv.name)
            nv.name(dev, name, sizeof name);
        if (nv.temp)
            nv.temp(dev, 0, &temp);
        if (nv.clock) {
            nv.clock(dev, 1, &sm);
            nv.clock(dev, 2, &mem);
        }
        if (nv.power)
            nv.power(dev, &mw);
        if (nv.util)
            nv.util(dev, &u);
        if (nv.pstate)
            nv.pstate(dev, &ps);
        /* the device's energy counter in millijoules, where the driver keeps one; 0 otherwise */
        unsigned long long mj = 0;
        if (nv.energy && nv.energy(dev, &mj))
            mj = 0;
        fprintf(out,
                "%s{\"name\":\"%s\",\"temp_c\":%u,\"sm_mhz\":%u,\"mem_mhz\":%u,\"power_mw\":%u,"
                "\"util_gpu\":%u,\"util_mem\":%u,\"pstate\":%d,\"energy_mj\":%llu}",
                d ? "," : "", name, temp, sm, mem, mw, u.gpu, u.memory, ps, mj);
    }
    fprintf(out, "]");
    {
        char bat[64], ac[64];
        read_text("/sys/class/power_supply/BAT0/capacity", bat, sizeof bat);
        read_text("/sys/class/power_supply/AC/online", ac, sizeof ac);
        if (!ac[0])
            read_text("/sys/class/power_supply/AC0/online", ac, sizeof ac);
        if (!ac[0])
            read_text("/sys/class/power_supply/ADP1/online", ac, sizeof ac);
        fprintf(out, ",\"battery_percent\":\"%s\",\"ac_online\":\"%s\"}", bat, ac);
    }
}

/* Device memory this process holds on any NVIDIA GPU, in bytes, as the driver reports it; 0 when
 * it does not. */
static unsigned long long nvml_process_bytes(void)
{
    unsigned long long sum = 0;
    if (!nv.tried)
        nvml_load();
    for (unsigned d = 0; nv.ok && nv.procs && d < nv.count; d++) {
        nvmlDevice_t dev;
        struct nvml_proc pr[64];
        unsigned n = 64;
        if (nv.handle(d, &dev) || nv.procs(dev, &n, pr))
            continue;
        for (unsigned i = 0; i < n && i < 64; i++)
            if (pr[i].pid == (unsigned)getpid() && pr[i].used != ~0ull)
                sum += pr[i].used;
    }
    return sum;
}

static void memory_json(FILE *o, const char *key, const neuronte_gpu *g)
{
    struct neuronte_gpu_memory m = {0};
    neuronte_gpu_memory(g, &m);
    fprintf(o, ",\"%s\":{\"process_device_bytes\":%llu,\"buffers\":%zu,\"weights\":%zu,\"staging\":%zu,"
            "\"driver_measured\":%zu}",
            key, nvml_process_bytes(), m.buffers, m.weights, m.staging, m.measured);
}

/* Scalars from a text file: one per line, as U+XXXX, 0x... or decimal. */
static int read_scalars(const char *path, uint32_t **out)
{
    FILE *f = fopen(path, "r");
    char line[128];
    int n = 0, cap = 0;
    uint32_t *v = NULL;
    if (!f)
        return -1;
    while (fgets(line, sizeof line, f)) {
        char *q = line;
        while (*q == ' ' || *q == '\t')
            q++;
        if (!*q || *q == '\n' || *q == '#')
            continue;
        if (n == cap) {
            cap = cap ? cap * 2 : 256;
            v = realloc(v, (size_t)cap * sizeof *v);
            if (!v)
                return -1;
        }
        v[n++] = (q[0] == 'U' || q[0] == 'u') && q[1] == '+' ? (uint32_t)strtoul(q + 2, NULL, 16)
                                                             : (uint32_t)strtoul(q, NULL, 0);
    }
    fclose(f);
    *out = v;
    return n;
}

static void shuffle(uint32_t *v, int n, uint64_t seed)
{
    uint64_t x = seed * 6364136223846793005ull + 1442695040888963407ull;
    for (int i = n - 1; i > 0; i--) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        int j = (int)((x >> 33) % (uint64_t)(i + 1));
        uint32_t t = v[i];
        v[i] = v[j];
        v[j] = t;
    }
}

static void usage(void)
{
    fprintf(stderr,
            "usage: neuronte_gpu_bench (--layout FILE [--models DIR] | --nrt FILE --cell WxH)\n"
            "       [--backend auto|cuda|vulkan|opencl|cpu] [--device NAME] [--fast] [--no-fallback]\n"
            "       [--allow-cpu-device] [--shared-weights] [--file-handle] [--samples S] [--block N] [--max-batch N] [--cpu K]\n"
            "       [--first SCALAR] [--count N] [--scalars FILE] [--per-call N] [--untrained 0|1] [--snap PX]\n"
            "       [--warmup-ms MS] [--warmup N] [--repeats N] [--passes N] [--single N] [--seed N] [--coverage-out FILE]\n");
    exit(2);
}

struct round {
    uint64_t wall, cpu;
    struct neuronte_gpu_timing t;
};

static void timing_json(FILE *o, const struct neuronte_gpu_timing *t)
{
    fprintf(o,
            "{\"upload\":%llu,\"run\":%llu,\"wait\":%llu,\"download\":%llu,\"cpu_eval\":%llu,\"total\":%llu,"
            "\"launches\":%d,\"cpu_glyphs\":%d,\"plan\":%llu,\"marks\":%d,\"warped\":%d}",
            (unsigned long long)t->upload, (unsigned long long)t->run, (unsigned long long)t->wait,
            (unsigned long long)t->download, (unsigned long long)t->cpu_eval, (unsigned long long)t->total,
            t->launches, t->cpu_glyphs, (unsigned long long)t->plan, t->marks, t->warped);
}

static void add(struct neuronte_gpu_timing *a, const struct neuronte_gpu_timing *b)
{
    a->upload += b->upload;
    a->run += b->run;
    a->wait += b->wait;
    a->download += b->download;
    a->cpu_eval += b->cpu_eval;
    a->total += b->total;
    a->launches += b->launches;
    a->cpu_glyphs += b->cpu_glyphs;
    a->plan += b->plan;
    a->marks += b->marks;
    a->warped += b->warped;
}

int main(int argc, char **argv)
{
    struct neuronte_gpu_options opt = {0};
    const char *backend = "auto", *coverage_out = NULL, *scalars_path = NULL;
    int untrained = -1, snap = -2;
    uint32_t *listed = NULL;
    int warmup = 1, warmup_ms = 0, warmup_calls = 0, repeats = 10, passes = 4, single = 0, cpu = -1, count = 256, per_call = 0, i, p, r;
    uint32_t first = 0x2800;
    uint64_t seed = 1;
    char reason[1024] = "";
    neuronte_gpu *g = NULL;
    opt.flags = NEURONTE_GPU_ALLOW_FALLBACK;

    for (i = 1; i < argc; i++) {
#define ARG() (i + 1 < argc ? argv[++i] : (usage(), (char *)0))
        if (!strcmp(argv[i], "--layout")) opt.layout_path = ARG();
        else if (!strcmp(argv[i], "--models")) opt.model_dir = ARG();
        else if (!strcmp(argv[i], "--nrt")) opt.nrt_path = ARG();
        else if (!strcmp(argv[i], "--cell")) {
            if (sscanf(ARG(), "%dx%d", &opt.cell_w, &opt.cell_h) != 2) usage();
        }
        else if (!strcmp(argv[i], "--backend")) backend = ARG();
        else if (!strcmp(argv[i], "--device")) opt.device = ARG();
        else if (!strcmp(argv[i], "--fast")) opt.flags |= NEURONTE_GPU_FAST;
        else if (!strcmp(argv[i], "--no-fallback")) opt.flags &= ~NEURONTE_GPU_ALLOW_FALLBACK;
        else if (!strcmp(argv[i], "--allow-cpu-device")) opt.flags |= NEURONTE_GPU_ALLOW_CPU_DEVICE;
        else if (!strcmp(argv[i], "--shared-weights")) opt.flags |= NEURONTE_GPU_SHARED_WEIGHTS;
        else if (!strcmp(argv[i], "--file-handle")) opt.flags |= NEURONTE_GPU_FILE;
        else if (!strcmp(argv[i], "--samples")) opt.samples = atoi(ARG());
        else if (!strcmp(argv[i], "--block")) opt.block = atoi(ARG());
        else if (!strcmp(argv[i], "--max-batch")) opt.max_batch = atoi(ARG());
        else if (!strcmp(argv[i], "--cpu")) cpu = atoi(ARG());
        else if (!strcmp(argv[i], "--first")) first = (uint32_t)strtoul(ARG(), NULL, 0);
        else if (!strcmp(argv[i], "--count")) count = atoi(ARG());
        else if (!strcmp(argv[i], "--per-call")) per_call = atoi(ARG());
        else if (!strcmp(argv[i], "--scalars")) scalars_path = ARG();
        else if (!strcmp(argv[i], "--untrained")) untrained = atoi(ARG());
        else if (!strcmp(argv[i], "--snap")) snap = atoi(ARG());
        else if (!strcmp(argv[i], "--warmup")) warmup = atoi(ARG());
        else if (!strcmp(argv[i], "--warmup-ms")) warmup_ms = atoi(ARG());
        else if (!strcmp(argv[i], "--repeats")) repeats = atoi(ARG());
        else if (!strcmp(argv[i], "--passes")) passes = atoi(ARG());
        else if (!strcmp(argv[i], "--single")) single = atoi(ARG());
        else if (!strcmp(argv[i], "--seed")) seed = strtoull(ARG(), NULL, 10);
        else if (!strcmp(argv[i], "--coverage-out")) coverage_out = ARG();
        else usage();
#undef ARG
    }
    if ((!opt.layout_path && !opt.nrt_path) || warmup < 0 || warmup_ms < 0 || repeats < 1 || repeats > MAX_ROUNDS || passes < 1 ||
        single < 0 || count < 1 || per_call < 0)
        usage();
    if (scalars_path && (count = read_scalars(scalars_path, &listed)) < 1) {
        fprintf(stderr, "cannot read scalars from %s\n", scalars_path);
        return 1;
    }
    if (!per_call)
        per_call = count;
    if (!strcmp(backend, "auto")) opt.backend = NEURONTE_GPU_AUTO;
    else if (!strcmp(backend, "cuda")) opt.backend = NEURONTE_GPU_CUDA;
    else if (!strcmp(backend, "vulkan")) opt.backend = NEURONTE_GPU_VULKAN;
    else if (!strcmp(backend, "opencl")) opt.backend = NEURONTE_GPU_OPENCL;
    else if (!strcmp(backend, "opengl")) opt.backend = NEURONTE_GPU_OPENGL;
    else if (!strcmp(backend, "cpu")) opt.backend = NEURONTE_GPU_CPU;
    else usage();
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof set, &set)) {
            fprintf(stderr, "cannot pin to CPU %d: %s\n", cpu, strerror(errno));
            return 1;
        }
    }

    /* The JSON goes to the original stdout; file descriptor 1 is pointed at stderr so that a
       driver printing to stdout cannot corrupt it. */
    fflush(stdout);
    int jfd = dup(1);
    FILE *o = jfd >= 0 ? fdopen(jfd, "w") : NULL;
    if (!o || dup2(2, 1) < 0) {
        fprintf(stderr, "cannot separate the JSON output from stdout: %s\n", strerror(errno));
        return 1;
    }
    fprintf(o, "{\"tool\":\"neuronte_gpu_bench\",\"requested_backend\":\"%s\",\"layout\":\"%s\",\"nrt\":\"%s\","
            "\"cpu\":%d,\"flags\":%d,\"device_request\":\"%s\",\"block\":%d,\"max_batch\":%d,\"warmup\":%d,"
            "\"repeats\":%d,\"passes\":%d,\"seed\":%llu,\"first\":%u,\"count\":%d,\"per_call\":%d,",
            backend, opt.layout_path ? opt.layout_path : "", opt.nrt_path ? opt.nrt_path : "", cpu, opt.flags,
            opt.device ? opt.device : "", opt.block, opt.max_batch, warmup, repeats, passes,
            (unsigned long long)seed, first, count, per_call);
    fprintf(o, "\"hardware_start\":");
    snapshot(o, cpu);

    uint64_t w0 = clock_ns(CLOCK_MONOTONIC), c0 = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
    int rc = neuronte_gpu_open(&opt, &g, reason, sizeof reason);
    uint64_t open_wall = clock_ns(CLOCK_MONOTONIC) - w0, open_cpu = clock_ns(CLOCK_PROCESS_CPUTIME_ID) - c0;
    if (rc) {
        for (char *q = reason; *q; q++)
            if (*q == '"' || *q == '\\' || *q == '\n')
                *q = ' ';
        fprintf(o, ",\"error\":\"open failed: %s\"}\n", reason);
        return 1;
    }
    int cw, ch, b = neuronte_gpu_backend(g);
    neuronte_gpu_cell(g, &cw, &ch);
    {
        char fr[1024];
        snprintf(fr, sizeof fr, "%s", neuronte_gpu_fallback_reason(g));
        for (char *q = fr; *q; q++)
            if (*q == '"' || *q == '\\' || *q == '\n')
                *q = ' ';
        fprintf(o, ",\"backend\":\"%s\",\"fallback_reason\":\"%s\",\"device\":\"%s\",\"device_type\":\"%s\","
                "\"versions\":\"%s\",\"cell\":[%d,%d],\"samples\":%d,\"hidden\":%d,\"tiles\":%d,\"exact\":%s",
                /* a Vulkan or OpenCL device that is a CPU (lavapipe, PoCL; with --allow-cpu-device) is never
                   labelled a GPU */
                !strcmp(neuronte_gpu_device_type(g), "cpu")
                    ? (b == NEURONTE_GPU_OPENCL ? "opencl-cpu-device"
                       : b == NEURONTE_GPU_OPENGL ? "opengl-cpu-device" : "vulkan-cpu-device")
                    : neuronte_gpu_backend_name(b),
                fr, neuronte_gpu_device(g), neuronte_gpu_device_type(g),
                neuronte_gpu_versions(g), cw, ch, neuronte_gpu_samples(g), neuronte_gpu_hidden(g),
                neuronte_gpu_tiles(g), neuronte_gpu_exact(g) ? "true" : "false");
    }
    fprintf(o, ",\"open_ns\":%llu,\"open_cpu_ns\":%llu", (unsigned long long)open_wall, (unsigned long long)open_cpu);
    memory_json(o, "memory_open", g);
    if ((untrained >= 0 && neuronte_gpu_set_untrained(g, untrained)) || (snap > -2 && neuronte_gpu_set_snap(g, snap))) {
        fprintf(o, ",\"error\":\"--untrained and --snap need a file handle\"}\n");
        return 1;
    }

    size_t plane = (size_t)cw * (size_t)ch;
    uint32_t *corpus = malloc((size_t)count * sizeof *corpus);
    uint32_t *call = malloc((size_t)per_call * sizeof *call);
    float *out = malloc(plane * (size_t)per_call * sizeof *out);
    struct round *rounds = calloc((size_t)(warmup + repeats), sizeof *rounds);
    if (!corpus || !call || !out || !rounds)
        return 1;
    for (i = 0; i < count; i++)
        corpus[i] = listed ? listed[i] : first + (uint32_t)i;
    shuffle(corpus, count, seed);
    for (i = 0; i < per_call; i++)
        call[i] = corpus[i % count];

    struct neuronte_gpu_timing t;
    w0 = clock_ns(CLOCK_MONOTONIC);
    c0 = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
    if (neuronte_gpu_coverage(g, call, per_call, out, &t)) {
        fprintf(o, ",\"error\":\"first coverage call failed\"}\n");
        return 1;
    }
    fprintf(o, ",\"first_call\":{\"wall\":%llu,\"cpu\":%llu,\"stages\":",
            (unsigned long long)(clock_ns(CLOCK_MONOTONIC) - w0),
            (unsigned long long)(clock_ns(CLOCK_PROCESS_CPUTIME_ID) - c0));
    timing_json(o, &t);
    fprintf(o, "}");
    memory_json(o, "memory_first_call", g);
    /* the first call's rasters, to compare with the last call's below */
    float *first_out = malloc(plane * (size_t)per_call * sizeof *first_out);
    if (!first_out)
        return 1;
    memcpy(first_out, out, plane * (size_t)per_call * sizeof *first_out);

    if (warmup_ms > 0) {
        uint64_t end = clock_ns(CLOCK_MONOTONIC) + (uint64_t)warmup_ms * 1000000u;
        do {
            if (neuronte_gpu_coverage(g, call, per_call, out, &t)) {
                fprintf(o, ",\"error\":\"coverage call failed in the timed warmup\"}\n");
                return 1;
            }
            warmup_calls++;
        } while (clock_ns(CLOCK_MONOTONIC) < end);
    }
    fprintf(o, ",\"warmup_ms\":%d,\"warmup_ms_calls\":%d,\"hardware_measure\":", warmup_ms, warmup_calls);
    snapshot(o, cpu);

    for (r = 0; r < warmup + repeats; r++) {
        struct round *rd = &rounds[r];
        w0 = clock_ns(CLOCK_MONOTONIC);
        c0 = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
        for (p = 0; p < passes; p++) {
            if (neuronte_gpu_coverage(g, call, per_call, out, &t)) {
                fprintf(o, ",\"error\":\"coverage call failed in round %d\"}\n", r);
                return 1;
            }
            add(&rd->t, &t);
        }
        rd->wall = clock_ns(CLOCK_MONOTONIC) - w0;
        rd->cpu = clock_ns(CLOCK_PROCESS_CPUTIME_ID) - c0;
    }
    fprintf(o, ",\"glyphs_per_round\":%d,\"rounds\":[", per_call * passes);
    for (r = 0; r < warmup + repeats; r++) {
        fprintf(o, "%s{\"round\":%d,\"warmup\":%s,\"wall\":%llu,\"cpu\":%llu,\"stages\":", r ? "," : "", r - warmup,
                r < warmup ? "true" : "false", (unsigned long long)rounds[r].wall, (unsigned long long)rounds[r].cpu);
        timing_json(o, &rounds[r].t);
        fprintf(o, "}");
    }
    fprintf(o, "]");

    if (single) {
        float *one = malloc(plane * sizeof *one);
        fprintf(o, ",\"single\":[");
        for (i = 0; i < single; i++) {
            w0 = clock_ns(CLOCK_MONOTONIC);
            if (neuronte_gpu_coverage(g, &corpus[i % count], 1, one, &t))
                return 1;
            fprintf(o, "%s%llu", i ? "," : "", (unsigned long long)(clock_ns(CLOCK_MONOTONIC) - w0));
        }
        fprintf(o, "]");
        free(one);
    }

    fprintf(o, ",\"first_call_equals_last\":%s",
            memcmp(first_out, out, plane * (size_t)per_call * sizeof *out) == 0 ? "true" : "false");
    free(first_out);
    if (coverage_out) {
        /* one raster per corpus scalar, in scalar order */
        FILE *f = fopen(coverage_out, "wb");
        int ok = f != NULL;
        for (i = 0; ok && i < count; i++) {
            int k;
            for (k = 0; k < per_call && call[k] != (listed ? listed[i] : first + (uint32_t)i); k++)
                ;
            ok = k < per_call && fwrite(out + (size_t)k * plane, sizeof *out, plane, f) == plane;
        }
        if (f)
            ok = fclose(f) == 0 && ok;
        fprintf(o, ",\"coverage_out\":\"%s\",\"coverage_written\":%s", coverage_out, ok ? "true" : "false");
    }
    fprintf(o, ",\"hardware_end\":");
    snapshot(o, cpu);
    fprintf(o, "}\n");
    neuronte_gpu_close(g);
    free(out);
    free(call);
    free(corpus);
    free(rounds);
    return 0;
}
