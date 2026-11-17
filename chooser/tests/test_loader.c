/* The loader against a stand-in GPU library: every way it must fall back
 * to the CPU, and that what it answers is the reference library's bytes
 * wherever they were computed.
 *
 *   test_loader DIR STATE NRT [cells]
 *
 * DIR is a scratch directory; STATE holds a backend.conf that chose the
 * stand-in GPU (made by the step 10-backend); the environment names a
 * current warm stamp (XDG_STATE_HOME), the settings, an empty power-supply
 * directory, the driver's stand-in name and FAKE_GPU_LOG. Each case runs in
 * a process of its own. With "cells" only the cases for the cells file run
 * (the program is then built with the stand-in cells file). */
#include "../common.h"

#include "neuronte.h"
#include "neuronte_backend.h"

#include <sys/wait.h>

static const char *DIR_, *STATE, *NRT;
static char conf[4096], gpulog[4096];
static int fails, checks;

#define OK(cond) \
    do { \
        checks++; \
        if (!(cond)) { \
            fprintf(stderr, "    FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            fails++; \
        } \
    } while (0)

enum { CW = 8, CH = 16, PLANE = CW * CH, MANY = 256 };

static void put(const char *path, const char *text, const char *mode)
{
    FILE *f = fopen(path, mode);
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(2);
    }
    fputs(text, f);
    fclose(f);
}

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "r");
    char *s = calloc(1, 1 << 20);
    if (f) {
        if (fread(s, 1, (1 << 20) - 1, f) == 0)
            s[0] = 0;
        fclose(f);
    }
    return s;
}

/* A copy of the good backend.conf that this case may add lines to (the last setting of a key
 * is the one read). */
static void fresh_conf(void)
{
    char src[4096], *text;
    snprintf(src, sizeof src, "%s/backend.conf", STATE);
    snprintf(conf, sizeof conf, "%s/case.conf", DIR_);
    text = slurp(src);
    put(conf, text, "w");
    free(text);
    /* what the stand-in measured as its smallest worthwhile call is not what is tested here */
    put(conf, "min_glyphs=1\n", "a");
    setenv("NEURONTE_BACKEND_CONF", conf, 1);
    snprintf(gpulog, sizeof gpulog, "%s/gpu.log", DIR_);
    unlink(gpulog);
    setenv("FAKE_GPU_LOG", gpulog, 1);
}

static void conf_add(const char *line)
{
    put(conf, line, "a");
    put(conf, "\n", "a");
}

/* Lines of the stand-in's log that start with `word`. */
static int logged(const char *word)
{
    char *text = slurp(gpulog), *p = text;
    size_t n = strlen(word);
    int k = 0;
    while (p && *p) {
        if (!strncmp(p, word, n))
            k++;
        p = strchr(p, '\n');
        if (p)
            p++;
    }
    free(text);
    return k;
}

static void nap_ms(int ms)
{
    struct timespec t = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&t, NULL);
}

static void battery(const char *name, const char *status, int percent, const char *scope)
{
    char dir[4096], path[4200], v[32];
    snprintf(dir, sizeof dir, "%s/power", DIR_);
    mkdir(dir, 0755);
    setenv("NEURONTE_BACKEND_POWER_DIR", dir, 1);
    snprintf(dir, sizeof dir, "%s/power/%s", DIR_, name);
    mkdir(dir, 0755);
    snprintf(path, sizeof path, "%s/type", dir);
    put(path, "Battery\n", "w");
    snprintf(path, sizeof path, "%s/status", dir);
    put(path, status, "w");
    snprintf(v, sizeof v, "%d\n", percent);
    snprintf(path, sizeof path, "%s/capacity", dir);
    put(path, v, "w");
    snprintf(path, sizeof path, "%s/scope", dir);
    if (scope)
        put(path, scope, "w");
    else
        unlink(path);
}

/* count scalars from `first`, through the loader, compared with the reference library. */
static int same(neuronte_backend *b, uint32_t first, int count)
{
    static uint32_t s[MANY * 2];
    static float got[MANY * 2 * PLANE], want[MANY * 2 * PLANE];
    neuronte *n;
    int i, ok;
    for (i = 0; i < count; i++)
        s[i] = first + (uint32_t)i;
    memset(got, 0x7f, sizeof(float) * (size_t)count * PLANE);
    if (neuronte_backend_coverage(b, s, count, got) != 0)
        return 0;
    if (neuronte_open_file(NRT, &n) != NEURONTE_OK)
        return 0;
    for (i = 0; i < count; i++)
        neuronte_coverage(n, s[i], CW, CH, 0, want + (size_t)i * PLANE);
    neuronte_close(n);
    ok = memcmp(got, want, sizeof(float) * (size_t)count * PLANE) == 0;
    return ok;
}

static neuronte_backend *open_it(void)
{
    neuronte_backend *b = NULL;
    OK(neuronte_backend_open(NRT, CW, CH, &b) == 0 && b);
    if (!b)
        exit(1);
    return b;
}

/* Calls of `count` glyphs until the GPU answers one: it is opened in the background by the first
 * call that wants it, and the CPU draws meanwhile. Exactly one GPU call is made. Returns the
 * handle's counters as they then stand. */
static struct neuronte_backend_info awake(neuronte_backend *b, int count)
{
    struct neuronte_backend_info i;
    uint64_t before;
    int k;
    neuronte_backend_info(b, &i);
    before = i.from_gpu;
    for (k = 0; k < 150; k++) {
        nap_ms(20);
        if (!same(b, 0x20, count))
            break;
        neuronte_backend_info(b, &i);
        if (i.from_gpu > before)
            return i;
    }
    memset(&i, 0, sizeof i);
    return i;
}

/* ---- the cases ---- */

static void no_conf(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b;
    fresh_conf();
    setenv("NEURONTE_BACKEND_CONF", "/nonexistent/backend.conf", 1);
    b = open_it();
    OK(same(b, 0x30, 5) && same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && i.min_glyphs == 0 && !strcmp(i.backend, "cpu") && strstr(i.reason, "no backend.conf"));
    OK(i.from_cpu == 5 + MANY && i.from_gpu == 0 && logged("open") == 0);
    OK(neuronte_backend_choice(&i) == 0 && strstr(i.reason, "no backend.conf"));
    neuronte_backend_close(b);
}

static void cpu_chosen(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b;
    fresh_conf();
    conf_add("min_glyphs=0");
    conf_add("reason=the CPU is fastest here: a test says so");
    b = open_it();
    OK(same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && strstr(i.reason, "a test says so") && i.from_cpu == MANY && logged("open") == 0);
    neuronte_backend_close(b);
}

static void gpu_chosen(void)
{
    struct neuronte_backend_info i, i0;
    neuronte_backend *b;
    fresh_conf();
    OK(neuronte_backend_choice(&i) == 1 && i.min_glyphs == 1 && !strcmp(i.backend, "gpu-cuda"));
    OK(logged("open") == 0); /* the choice alone opens nothing */
    b = open_it();
    OK(logged("open") == 0); /* nor does opening a handle */
    i0 = awake(b, 1);        /* the first call that wants the GPU has it opened */
    OK(i0.gpu && !strcmp(i0.device, "Stand-in GPU") && logged("open") == 1);
    OK(same(b, 0x41, 1) && same(b, 0x2500, 3) && same(b, 0x20, MANY * 2));
    neuronte_backend_info(b, &i);
    OK(i.from_gpu == i0.from_gpu + 1 + 3 + MANY * 2 && i.from_cpu == i0.from_cpu && logged("open") == 1);
    neuronte_backend_close(b);
    OK(logged("close") == 1);
}

static void small_calls(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b;
    fresh_conf();
    conf_add("min_glyphs=16");
    setenv("FAKE_GPU_OPEN_MS", "100", 1);
    b = open_it();
    OK(same(b, 0x20, 15)); /* under min_glyphs: the CPU, and the GPU is not woken */
    neuronte_backend_info(b, &i);
    OK(i.from_cpu == 15 && logged("open") == 0);
    OK(same(b, 0x20, 16)); /* wanted on the GPU, which is not open: the CPU answers, nothing waits */
    neuronte_backend_info(b, &i);
    OK(i.from_cpu == 31 && i.from_gpu == 0);
    nap_ms(300);
    OK(same(b, 0x40, 16)); /* and now it is */
    neuronte_backend_info(b, &i);
    OK(i.from_gpu == 16 && i.from_cpu == 31 && logged("open") == 1);
    OK(same(b, 0x20, 15));
    neuronte_backend_info(b, &i);
    OK(i.from_gpu == 16 && i.from_cpu == 46);
    neuronte_backend_close(b);
}

static void forced_by_program(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b;
    fresh_conf();
    setenv("NEURONTE_BACKEND", "cpu", 1);
    b = open_it();
    OK(same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && !strcmp(i.reason, "forced by NEURONTE_BACKEND") && logged("open") == 0);
    neuronte_backend_close(b);
    /* nothing forces a GPU: any other value is the measured choice */
    setenv("NEURONTE_BACKEND", "gpu", 1);
    conf_add("min_glyphs=0");
    OK(neuronte_backend_choice(&i) == 0);
}

static void forced_by_settings(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b;
    char path[4096], *text;
    fresh_conf();
    snprintf(path, sizeof path, "%s/case-setup.conf", DIR_);
    text = slurp(getenv("NEURONTE_SETUP_CONF"));
    put(path, text, "w");
    put(path, "backend=cpu\n", "a");
    free(text);
    setenv("NEURONTE_SETUP_CONF", path, 1);
    b = open_it();
    OK(same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && strstr(i.reason, "backend=cpu in the settings") && logged("open") == 0);
    neuronte_backend_close(b);
}

static void stale(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b;
    fresh_conf();
    setenv("NEURONTE_BACKEND_TEST_DRIVER", "another driver", 1);
    b = open_it();
    OK(same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && strstr(i.reason, "driver changed") && logged("open") == 0);
    neuronte_backend_close(b);
    unsetenv("NEURONTE_BACKEND_TEST_DRIVER");

    fresh_conf();
    conf_add("gpu.library.id=1:1");
    OK(neuronte_backend_choice(&i) == 0 && strstr(i.reason, "library changed"));

    fresh_conf();
    conf_add("version=99");
    OK(neuronte_backend_choice(&i) == 0 && strstr(i.reason, "another version"));

    fresh_conf();
    conf_add("gpu.identical=no");
    OK(neuronte_backend_choice(&i) == 0 && strstr(i.reason, "byte check"));

    fresh_conf();
    put(conf, "\0\1\2 not a conf at all", "w");
    b = open_it();
    OK(same(b, 0x20, 20));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && logged("open") == 0);
    neuronte_backend_close(b);
}

static void warming(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b, *b2 = NULL;
    char empty[4096], kick[4200], mark[4096], other[4096], *text;
    int k;
    fresh_conf();
    /* a weights file the user's stamp does not name: its kernels may not be compiled */
    snprintf(other, sizeof other, "%s/another.nrt", DIR_);
    {
        FILE *in = fopen(NRT, "rb"), *out = fopen(other, "wb");
        int c;
        while (in && out && (c = fgetc(in)) != EOF)
            fputc(c, out);
        if (in)
            fclose(in);
        if (out)
            fclose(out);
    }
    snprintf(mark, sizeof mark, "%s/kicked", DIR_);
    unlink(mark);
    snprintf(kick, sizeof kick, "echo kicked >> '%s' &", mark);
    setenv("NEURONTE_BACKEND_KICK", kick, 1);
    OK(neuronte_backend_open(other, CW, CH, &b2) == 0 && b2);
    neuronte_backend_info(b2, &i);
    OK(!i.gpu && !strncmp(i.reason, "warming", 7));
    neuronte_backend_close(b2);

    /* a user with no stamp at all */
    snprintf(empty, sizeof empty, "%s/no-state", DIR_);
    setenv("XDG_STATE_HOME", empty, 1);
    b = open_it();
    OK(same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && !strncmp(i.reason, "warming", 7) && i.from_cpu == MANY && logged("open") == 0);
    neuronte_backend_close(b);
    for (k = 0; k < 20 && access(mark, F_OK); k++)
        nap_ms(50);
    text = slurp(mark);
    OK(!strcmp(text, "kicked\n")); /* the user's setup was asked to run, once */
    free(text);
}

static void on_battery(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b;
    fresh_conf();
    battery("BAT0", "Discharging\n", 95, NULL);
    OK(neuronte_backend_choice(&i) == 0 && !strcmp(i.reason, "on battery"));
    b = open_it();
    OK(same(b, 0x20, MANY));
    nap_ms(100);
    OK(same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && !strcmp(i.reason, "on battery") && i.from_cpu == 2 * MANY && logged("open") == 0);
    neuronte_backend_close(b);

    /* a floor in place of the default: above it the GPU, under it the CPU */
    conf_add("battery=90");
    b = open_it();
    i = awake(b, 1);
    OK(i.gpu && i.from_gpu > 0);
    neuronte_backend_close(b);
    battery("BAT0", "Discharging\n", 89, NULL);
    b = open_it();
    OK(same(b, 0x20, MANY));
    nap_ms(100);
    OK(same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && i.from_gpu == 0 && !strcmp(i.reason, "on battery") && logged("open") == 1);
    neuronte_backend_close(b);

    /* battery=same: the power supply changes nothing */
    conf_add("battery=same");
    b = open_it();
    i = awake(b, 1);
    OK(i.gpu && i.from_gpu > 0);
    neuronte_backend_close(b);

    /* a mouse's battery is not the machine's */
    conf_add("battery=cpu");
    battery("BAT0", "Charging\n", 60, NULL);
    battery("hidpp_battery_0", "Discharging\n", 20, "Device\n");
    OK(neuronte_backend_choice(&i) == 1);
}

static void battery_while_open(void)
{
    struct neuronte_backend_info i, i0, i1;
    neuronte_backend *b;
    int k;
    fresh_conf();
    battery("BAT0", "Charging\n", 100, NULL);
    b = open_it();
    i0 = awake(b, 1);
    OK(i0.gpu && logged("open") == 1);
    battery("BAT0", "Discharging\n", 99, NULL); /* the plug is pulled */
    nap_ms(1100);
    OK(same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && i.from_gpu == i0.from_gpu && i.from_cpu == i0.from_cpu + MANY && !strcmp(i.reason, "on battery"));
    for (k = 0; k < 20 && logged("close") == 0; k++)
        nap_ms(50);
    OK(logged("close") == 1); /* and the GPU is let go */
    battery("BAT0", "Charging\n", 99, NULL);
    nap_ms(1100);
    i1 = awake(b, 1);
    OK(i1.gpu && i1.from_gpu > i0.from_gpu && logged("open") == 2);
    neuronte_backend_close(b);
}

static void lost(void)
{
    struct neuronte_backend_info i, i0;
    neuronte_backend *b;
    fresh_conf();
    setenv("FAKE_GPU_FAIL_AT", "3", 1);
    b = open_it();
    i0 = awake(b, 1); /* the GPU's first call */
    OK(i0.gpu && same(b, 0x30, 40)); /* its second */
    neuronte_backend_info(b, &i);
    OK(i.gpu && i.from_gpu == i0.from_gpu + 40);
    OK(same(b, 0x40, 50)); /* the device fails under this call: it is answered all the same */
    neuronte_backend_info(b, &i);
    OK(!i.gpu && i.from_cpu == i0.from_cpu + 50 && strstr(i.reason, "device lost") && !strcmp(i.backend, "cpu"));
    OK(same(b, 0x20, MANY)); /* and the handle stays on the CPU */
    neuronte_backend_info(b, &i);
    OK(i.from_cpu == i0.from_cpu + 50 + MANY && i.from_gpu == i0.from_gpu + 40 && logged("coverage") == 3);
    neuronte_backend_close(b);
    OK(logged("close") == 1);
    /* a handle opened afterwards tries the GPU again */
    unsetenv("FAKE_GPU_FAIL_AT");
    b = open_it();
    i = awake(b, 1);
    OK(i.gpu && i.from_gpu > 0);
    neuronte_backend_close(b);
}

static void refused(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b;
    fresh_conf();
    setenv("FAKE_GPU_REFUSE", "all", 1);
    b = open_it();
    OK(same(b, 0x20, MANY));
    nap_ms(100);
    OK(same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && i.from_cpu == 2 * MANY && strstr(i.reason, "did not open") && logged("open") == 1);
    neuronte_backend_close(b);
}

static void wrong_library(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b;
    fresh_conf();
    setenv("FAKE_GPU_ABI", "99", 1); /* a library of another interface is not called */
    b = open_it();
    OK(same(b, 0x20, MANY));
    nap_ms(100);
    OK(same(b, 0x20, MANY));
    neuronte_backend_info(b, &i);
    OK(!i.gpu && strstr(i.reason, "interface 99") && i.from_cpu == 2 * MANY && logged("open") == 0);
    neuronte_backend_close(b);

    fresh_conf();
    unsetenv("FAKE_GPU_ABI");
    conf_add("gpu.library=/nonexistent/libneuronte_gpu.so.0");
    OK(neuronte_backend_choice(&i) == 0); /* the library named is not there: stale */
}

static void slow_open(void)
{
    struct neuronte_backend_info i;
    neuronte_backend *b;
    double t0;
    fresh_conf();
    setenv("FAKE_GPU_OPEN_MS", "1500", 1);
    t0 = nb_now();
    b = open_it();
    OK(nb_now() - t0 < 0.2); /* opening a handle does not wait for the GPU */
    t0 = nb_now();
    OK(same(b, 0x20, 4)); /* nor does a small call */
    OK(nb_now() - t0 < 0.2);
    t0 = nb_now();
    OK(same(b, 0x20, MANY)); /* nor a large one: the CPU draws it */
    OK(nb_now() - t0 < 0.5);
    neuronte_backend_info(b, &i);
    OK(i.from_cpu == 4 + MANY && i.from_gpu == 0);
    t0 = nb_now();
    neuronte_backend_close(b); /* nor closing */
    OK(nb_now() - t0 < 0.2);
    nap_ms(1700); /* the helper finishes by itself and lets the GPU go */
    OK(logged("open") == 1 && logged("close") == 1);

    b = open_it();
    OK(same(b, 0x20, 4));
    nap_ms(1700);
    OK(same(b, 0x20, 4));
    neuronte_backend_info(b, &i);
    OK(i.from_gpu == 4 && i.from_cpu == 4);
    neuronte_backend_close(b);
}

static void idle_close(void)
{
    struct neuronte_backend_info i0, i1;
    neuronte_backend *b;
    fresh_conf();
    conf_add("idle_close_ms=200");
    b = open_it();
    i0 = awake(b, 1);
    OK(i0.gpu && logged("open") == 1 && logged("close") == 0);
    nap_ms(100);
    OK(same(b, 0x20, 8)); /* a call within the spell keeps it open */
    nap_ms(150);
    OK(logged("close") == 0);
    nap_ms(400);
    OK(logged("close") == 1); /* closed after the spell without a call */
    i1 = awake(b, 1);         /* and opened again when wanted */
    OK(i1.gpu && i1.from_gpu == i0.from_gpu + 8 + 1 && logged("open") == 2);
    neuronte_backend_close(b);
    OK(logged("close") == 2);
}

static void arguments(void)
{
    neuronte_backend *b = NULL;
    uint32_t s = 0x41;
    float out[PLANE];
    fresh_conf();
    OK(neuronte_backend_open("/nonexistent/file.nrt", CW, CH, &b) != 0 && !b);
    OK(neuronte_backend_open(NRT, 0, CH, &b) != 0 && !b);
    OK(neuronte_backend_open(NULL, CW, CH, &b) != 0);
    OK(neuronte_backend_open(conf, CW, CH, &b) != 0 && !b); /* not a weights file */
    b = open_it();
    OK(neuronte_backend_coverage(b, &s, 0, out) == 0);
    OK(neuronte_backend_coverage(b, NULL, 1, out) != 0);
    OK(neuronte_backend_coverage(b, &s, -1, out) != 0);
    OK(neuronte_backend_coverage(NULL, &s, 1, out) != 0);
    neuronte_backend_info(NULL, NULL);
    neuronte_backend_close(NULL);
    neuronte_backend_close(b);
    /* two handles at once, at two sizes, each its own GPU handle */
    {
        neuronte_backend *a = open_it(), *c = NULL;
        struct neuronte_backend_info i;
        static float big[16 * 32];
        uint32_t one = 0x41;
        int k;
        OK(neuronte_backend_open(NRT, 16, 32, &c) == 0 && c);
        i = awake(a, 1);
        OK(i.gpu && i.from_gpu > 0);
        memset(&i, 0, sizeof i);
        for (k = 0; c && k < 150 && i.from_gpu == 0; k++) {
            nap_ms(20);
            OK(neuronte_backend_coverage(c, &one, 1, big) == 0);
            neuronte_backend_info(c, &i);
        }
        OK(i.from_gpu == 1 && logged("open") == 2);
        neuronte_backend_close(c);
        neuronte_backend_close(a);
        OK(logged("close") == 2);
    }
}

/* The GPU arrives while a call is being drawn on the CPU: the rest of the call goes to it. */
static void arrives(void)
{
    enum { N = 2048, W = 32, H = 64 };
    struct neuronte_backend_info i;
    neuronte_backend *b = NULL;
    uint32_t *list = malloc(N * sizeof *list);
    float *got = malloc((size_t)N * W * H * sizeof *got), *want = malloc((size_t)N * W * H * sizeof *want);
    neuronte *n;
    int k;
    fresh_conf();
    setenv("FAKE_GPU_OPEN_MS", "30", 1);
    OK(list && got && want && neuronte_open_file(NRT, &n) == NEURONTE_OK);
    for (k = 0; k < N; k++) {
        list[k] = 0x20 + (uint32_t)k;
        neuronte_coverage(n, list[k], W, H, 0, want + (size_t)k * W * H);
    }
    neuronte_close(n);
    OK(neuronte_backend_open(NRT, W, H, &b) == 0 && b);
    OK(neuronte_backend_coverage(b, list, N, got) == 0);
    neuronte_backend_info(b, &i);
    OK(i.from_cpu > 0 && i.from_gpu > 0 && i.from_cpu + i.from_gpu == N);
    OK(memcmp(got, want, (size_t)N * W * H * sizeof *got) == 0);
    OK(logged("open") == 1 && logged("coverage") == 1);
    neuronte_backend_close(b);
}

/* With the stand-in cells file: U+0041 to U+005A at 8x16 are pre-drawn. */
static void cells_first(void)
{
    struct neuronte_backend_info i, i0;
    neuronte_backend *b;
    uint32_t mixed[6] = {0x41, 0x61, 0x42, 0x62, 0x5A, 0x7A};
    float got[6 * PLANE], want[6 * PLANE];
    neuronte *n;
    int k;
    fresh_conf();
    setenv("FAKE_CELLS_SIZE", "8x16", 1);
    b = open_it();
    OK(same(b, 0x41, 26)); /* all pre-drawn: nothing is computed and the GPU is not woken */
    neuronte_backend_info(b, &i);
    OK(i.from_cells == 26 && i.from_cpu == 0 && i.from_gpu == 0 && logged("open") == 0);
    i0 = awake(b, 1);        /* U+0020 is not in the file: the GPU is opened for it */
    OK(i0.gpu && i0.from_cells == 26 && logged("coverage") == 1);
    OK(same(b, 0x20, MANY)); /* 26 of these from the file, the rest in one GPU call */
    neuronte_backend_info(b, &i);
    OK(i.from_cells == 52 && i.from_gpu == i0.from_gpu + MANY - 26 && i.from_cpu == i0.from_cpu && logged("coverage") == 2);
    OK(neuronte_backend_coverage(b, mixed, 6, got) == 0);
    OK(neuronte_open_file(NRT, &n) == NEURONTE_OK);
    for (k = 0; k < 6; k++)
        neuronte_coverage(n, mixed[k], CW, CH, 0, want + k * PLANE);
    neuronte_close(n);
    OK(memcmp(got, want, sizeof got) == 0); /* each glyph in its own place */
    neuronte_backend_info(b, &i);
    OK(i.from_cells == 55 && i.from_gpu == i0.from_gpu + MANY - 26 + 3);
    neuronte_backend_close(b);

    /* what is left after the file decides CPU or GPU: 3 glyphs under min_glyphs=4 */
    fresh_conf();
    conf_add("min_glyphs=4");
    b = open_it();
    OK(neuronte_backend_coverage(b, mixed, 6, got) == 0 && memcmp(got, want, sizeof got) == 0);
    neuronte_backend_info(b, &i);
    OK(i.from_cells == 3 && i.from_cpu == 3 && i.from_gpu == 0 && logged("open") == 0);
    neuronte_backend_close(b);

    /* another size is not in the file */
    {
        neuronte_backend *c = NULL;
        uint32_t s = 0x41;
        static float big[16 * 32];
        OK(neuronte_backend_open(NRT, 16, 32, &c) == 0 && c);
        OK(neuronte_backend_coverage(c, &s, 1, big) == 0);
        neuronte_backend_info(c, &i);
        OK(i.from_cells == 0 && i.from_cpu + i.from_gpu == 1);
        neuronte_backend_close(c);
    }
}

static int run(const char *name, void (*fn)(void))
{
    pid_t pid;
    int status = 0;
    fflush(NULL);
    pid = fork();
    if (pid == 0) {
        fn();
        printf("  %-22s %3d checks, %d failed\n", name, checks, fails);
        fflush(NULL);
        _exit(fails ? 1 : 0);
    }
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status)) {
        printf("  %-22s ended by signal %d\n", name, WTERMSIG(status));
        return 1;
    }
    return WEXITSTATUS(status) != 0;
}

int main(int argc, char **argv)
{
    int bad = 0;
    if (argc < 4) {
        fprintf(stderr, "usage: test_loader DIR STATE NRT [cells]\n");
        return 2;
    }
    DIR_ = argv[1];
    STATE = argv[2];
    NRT = argv[3];
    if (argc > 4) {
        bad += run("cells file first", cells_first);
    } else {
        bad += run("no backend.conf", no_conf);
        bad += run("the CPU chosen", cpu_chosen);
        bad += run("a GPU chosen", gpu_chosen);
        bad += run("small calls", small_calls);
        bad += run("forced, program", forced_by_program);
        bad += run("forced, settings", forced_by_settings);
        bad += run("stale conf", stale);
        bad += run("warming", warming);
        bad += run("on battery", on_battery);
        bad += run("plug pulled", battery_while_open);
        bad += run("device lost", lost);
        bad += run("GPU refused", refused);
        bad += run("wrong library", wrong_library);
        bad += run("slow opening", slow_open);
        bad += run("GPU arrives in a call", arrives);
        bad += run("idle close", idle_close);
        bad += run("arguments", arguments);
    }
    printf("test_loader%s: %d of the cases failed\n", argc > 4 ? " (cells)" : "", bad);
    return bad != 0;
}
