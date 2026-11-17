/* neuronte_backend: see neuronte_backend.h.
 *
 * What is decided once, when a handle is opened (choose()): the overrides,
 * setup's backend.conf, whether that file still describes this machine, and
 * whether this user's GPU kernels are compiled. What is decided at each
 * call: the cells file first, then the number of glyphs left against
 * min_glyphs, the power supply (read at most once a second), and whether
 * the GPU is open.
 *
 * The GPU belongs to a helper thread of the handle: it opens it when a call
 * first wants it, closes it after idle_close_ms without a call or when the
 * machine goes on battery, and closes a device that failed. The caller's
 * thread only ever runs coverage on an open GPU and never waits for one:
 * while the GPU is opening a call draws on the CPU, and hands what is left
 * of it to the GPU as soon as that is open. */
#include "common.h"

#include "neuronte.h"
#include "neuronte_backend.h"
#include "neuronte_gpu.h"

#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>

#include "decide.h"

extern char **environ;

/* ---- the cells file (setup step 30-cells) ----
 * Pre-drawn coverage for a weights file, mapped read-only. These are its
 * calls as they were agreed (header neuronte_cells.h, not yet written when
 * this was); they are weak references, so with a reference library that has
 * none the step is skipped. find() returns cell_w * cell_h floats that are
 * byte for byte neuronte_coverage()'s for the scalar, or NULL when the file
 * does not hold it at that size. The loader asks for the plain style, the
 * library's own sampling, and snap_below -1, which is what every file the
 * setup step writes was made under. */
typedef struct neuronte_cells neuronte_cells;
#if defined(__GNUC__) && !defined(NEURONTE_BACKEND_NO_CELLS)
extern int neuronte_cells_open(const char *cells_path, const char *weights_path, unsigned flags, neuronte_cells **out)
    __attribute__((weak));
extern const float *neuronte_cells_find(const neuronte_cells *c, uint32_t scalar, int cell_w, int cell_h, int samples,
                                        unsigned style, int snap_below) __attribute__((weak));
extern void neuronte_cells_close(neuronte_cells *c) __attribute__((weak));
#define HAVE_CELLS (neuronte_cells_open && neuronte_cells_find && neuronte_cells_close)
#else
#define HAVE_CELLS 0
#define neuronte_cells_open(p, w, f, o) (-1)
#define neuronte_cells_find(c, s, w, h, n, y, b) ((const float *)NULL)
#define neuronte_cells_close(c) ((void)0)
#endif

/* ---- the GPU library: loaded once in a process, never unloaded ---- */

struct gpu_api {
    void *dl;
    char path[4096];
    int (*abi)(void);
    int (*open)(const struct neuronte_gpu_options *, neuronte_gpu **, char *, size_t);
    void (*close)(neuronte_gpu *);
    int (*coverage)(neuronte_gpu *, const uint32_t *, int, float *, struct neuronte_gpu_timing *);
    const char *(*device)(const neuronte_gpu *);
    int (*backend)(const neuronte_gpu *);
};

static pthread_mutex_t lib_mu = PTHREAD_MUTEX_INITIALIZER;
static struct gpu_api the_lib;

static const struct gpu_api *gpu_api_get(const char *path, char *why, size_t len)
{
    struct gpu_api g;
    const struct gpu_api *ret = NULL;
    pthread_mutex_lock(&lib_mu);
    if (the_lib.dl) {
        if (!strcmp(the_lib.path, path))
            ret = &the_lib;
        else
            snprintf(why, len, "another GPU library is loaded in this program already");
        goto out;
    }
    memset(&g, 0, sizeof g);
    g.dl = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!g.dl) {
        snprintf(why, len, "the GPU library cannot be loaded: %s", dlerror());
        goto out;
    }
    g.abi = (int (*)(void))dlsym(g.dl, "neuronte_gpu_abi");
    g.open = (int (*)(const struct neuronte_gpu_options *, neuronte_gpu **, char *, size_t))dlsym(g.dl, "neuronte_gpu_open");
    g.close = (void (*)(neuronte_gpu *))dlsym(g.dl, "neuronte_gpu_close");
    g.coverage = (int (*)(neuronte_gpu *, const uint32_t *, int, float *, struct neuronte_gpu_timing *))dlsym(
        g.dl, "neuronte_gpu_coverage");
    g.device = (const char *(*)(const neuronte_gpu *))dlsym(g.dl, "neuronte_gpu_device");
    g.backend = (int (*)(const neuronte_gpu *))dlsym(g.dl, "neuronte_gpu_backend");
    if (!g.abi || !g.open || !g.close || !g.coverage || !g.device || !g.backend) {
        snprintf(why, len, "the GPU library lacks a call this program needs");
        dlclose(g.dl);
        goto out;
    }
    if (g.abi() != NEURONTE_GPU_ABI) {
        snprintf(why, len, "the GPU library has interface %d, this program was built for %d", g.abi(), NEURONTE_GPU_ABI);
        dlclose(g.dl);
        goto out;
    }
    snprintf(g.path, sizeof g.path, "%s", path);
    the_lib = g;
    ret = &the_lib;
out:
    pthread_mutex_unlock(&lib_mu);
    return ret;
}

/* ---- the choice that does not change while a handle is open ---- */

enum { BATTERY_SAME, BATTERY_CPU, BATTERY_FLOOR };

struct choice {
    int gpu; /* setup chose a GPU for this machine and nothing forbids it */
    int min_glyphs, max_batch, idle_close_ms;
    int battery_mode, battery_floor;
    char library[4096], backend[32], device[128], reason[256];
};

/* Ask the user's setup unit to run, without waiting for it: once in a process. The command is
 * NEURONTE_BACKEND_KICK when set ("0": none), else systemctl's. */
static void kick_user_setup(void)
{
    static int kicked;
    const char *cmd = nb_env("NEURONTE_BACKEND_KICK");
    char *argv[] = {"sh", "-c", NULL, NULL};
    pid_t pid;
    int was;
    pthread_mutex_lock(&lib_mu);
    was = kicked;
    kicked = 1;
    pthread_mutex_unlock(&lib_mu);
    if (was || (cmd && !strcmp(cmd, "0")) || geteuid() == 0)
        return;
    argv[2] = (char *)(cmd ? cmd : "systemctl --user --no-block start neuronte-setup-user.service >/dev/null 2>&1 &");
    /* the shell starts it in the background and ends at once, so the wait is short */
    if (posix_spawn(&pid, "/bin/sh", NULL, NULL, argv, environ) == 0)
        while (waitpid(pid, NULL, 0) < 0 && errno == EINTR)
            ;
}

static int setting_says_cpu(const char *path)
{
    struct nb_kv kv;
    int yes;
    if (nb_kv_read(path, &kv))
        return 0;
    yes = !strcmp(nb_kv_get(&kv, "backend", ""), "cpu");
    nb_kv_free(&kv);
    return yes;
}

/* file: the weights file a handle is for, or NULL for the machine's choice alone. */
static void choose(const char *file, struct choice *c)
{
    struct nb_kv conf;
    char path[4096], stamp[4096], why[256];
    const char *v, *x, *h;
    memset(c, 0, sizeof *c);
    snprintf(c->backend, sizeof c->backend, "cpu");

    v = nb_env("NEURONTE_BACKEND");
    if (v && !strcmp(v, "cpu")) {
        snprintf(c->reason, sizeof c->reason, "forced by NEURONTE_BACKEND");
        return;
    }
    x = nb_env("XDG_CONFIG_HOME");
    h = nb_env("HOME");
    if (x)
        snprintf(path, sizeof path, "%s/neuronte/setup.conf", x);
    else
        snprintf(path, sizeof path, "%s/.config/neuronte/setup.conf", h ? h : "/nonexistent");
    if (setting_says_cpu(nb_setup_conf_path()) || setting_says_cpu(path)) {
        snprintf(c->reason, sizeof c->reason, "backend=cpu in the settings");
        return;
    }
    nb_backend_conf_path(path, sizeof path);
    if (nb_kv_read(path, &conf)) {
        snprintf(c->reason, sizeof c->reason, "no backend.conf: Neuronte's setup has not measured this machine");
        return;
    }
    if (nb_conf_stale(&conf, why, sizeof why)) {
        snprintf(c->reason, sizeof c->reason, "%.200s; setup must measure again", why);
        goto out;
    }
    snprintf(c->reason, sizeof c->reason, "%s", nb_kv_get(&conf, "reason", "setup chose the CPU"));
    c->min_glyphs = atoi(nb_kv_get(&conf, "min_glyphs", "0"));
    v = nb_kv_get(&conf, "gpu.backend", NULL);
    if (!v || c->min_glyphs <= 0) {
        c->min_glyphs = 0;
        goto out;
    }
    /* this user's kernels: the user's own stamp, or the system's for the system's own programs */
    v = nb_env("NEURONTE_BACKEND_WARM_STAMP");
    if (v)
        snprintf(stamp, sizeof stamp, "%s", v);
    else if (geteuid() == 0) {
        nb_system_state(path, sizeof path);
        snprintf(stamp, sizeof stamp, "%.4000s/40-backend-warm.stamp", path);
    } else {
        nb_user_state(path, sizeof path);
        snprintf(stamp, sizeof stamp, "%.4000s/40-backend-warm.stamp", path);
    }
    if (!nb_warm_current(stamp, &conf, file)) {
        snprintf(c->reason, sizeof c->reason, "warming: this user's GPU kernels are not compiled yet (neuronte-setup --user)");
        c->min_glyphs = 0;
        nb_kv_free(&conf);
        kick_user_setup();
        return;
    }
    c->gpu = 1;
    snprintf(c->library, sizeof c->library, "%s", nb_kv_get(&conf, "gpu.library", ""));
    snprintf(c->backend, sizeof c->backend, "gpu-%.20s", nb_kv_get(&conf, "gpu.backend", ""));
    snprintf(c->device, sizeof c->device, "%s", nb_kv_get(&conf, "gpu.device", ""));
    c->max_batch = atoi(nb_kv_get(&conf, "gpu.max_batch", "1024"));
    c->idle_close_ms = atoi(nb_kv_get(&conf, "idle_close_ms", "0"));
    v = nb_kv_get(&conf, "battery", "cpu");
    if (!strcmp(v, "same"))
        c->battery_mode = BATTERY_SAME;
    else if (*v >= '0' && *v <= '9') {
        c->battery_mode = BATTERY_FLOOR;
        c->battery_floor = atoi(v);
    } else
        c->battery_mode = BATTERY_CPU;
out:
    nb_kv_free(&conf);
}

static int battery_forbids(const struct choice *c)
{
    int percent, discharging;
    if (c->battery_mode == BATTERY_SAME)
        return 0;
    discharging = nb_discharging(&percent);
    return discharging && (c->battery_mode == BATTERY_CPU || percent < c->battery_floor);
}

/* ---- a handle ---- */

enum { ST_CLOSED, ST_OPENING, ST_OPEN, ST_LOST };

struct neuronte_backend {
    neuronte *cpu;
    neuronte_cells *cells;
    char path[4096];
    int cw, ch;
    size_t plane;
    struct choice c;

    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_t helper;
    int helper_on;
    int state, want_open, drop, busy, quit, orphan;
    const struct gpu_api *api;
    neuronte_gpu *gpu, *dead;
    double last_use;
    char device[128], lost[256];

    double power_at;
    int on_battery;
    uint64_t from_cells, from_cpu, from_gpu;
    uint32_t *miss; /* the scalars of a call that the cells file does not hold */
    int *miss_at;   /* and their places in the call */
    float *tmp;
    int cap;
};

static int backend_number(const char *name)
{
    return !strcmp(name, "gpu-cuda") ? NEURONTE_GPU_CUDA : !strcmp(name, "gpu-vulkan") ? NEURONTE_GPU_VULKAN
         : !strcmp(name, "gpu-opencl") ? NEURONTE_GPU_OPENCL : -1;
}

static neuronte_gpu *open_gpu(neuronte_backend *b, char *why, size_t len)
{
    struct neuronte_gpu_options opt = {0};
    neuronte_gpu *g = NULL;
    char reason[256] = "";
    int kind;
    b->api = gpu_api_get(b->c.library, why, len);
    if (!b->api)
        return NULL;
    opt.nrt_path = b->path;
    opt.cell_w = b->cw;
    opt.cell_h = b->ch;
    opt.backend = backend_number(b->c.backend);
    opt.flags = NEURONTE_GPU_FILE;
    opt.max_batch = b->c.max_batch > 0 ? b->c.max_batch : 0;
    if (opt.backend < 0) {
        snprintf(why, len, "backend.conf names a backend this program does not know");
        return NULL;
    }
    if (b->api->open(&opt, &g, reason, sizeof reason) || !g) {
        snprintf(why, len, "the GPU did not open: %.200s", reason[0] ? reason : "no reason given");
        return NULL;
    }
    kind = b->api->backend(g);
    if (kind == NEURONTE_GPU_CPU || kind == NEURONTE_GPU_CPU_FALLBACK) {
        b->api->close(g);
        snprintf(why, len, "the GPU library answered with its CPU path");
        return NULL;
    }
    return g;
}

static void free_handle(neuronte_backend *b)
{
    pthread_mutex_destroy(&b->mu);
    pthread_cond_destroy(&b->cv);
    free(b->miss);
    free(b->miss_at);
    free(b->tmp);
    free(b);
}

static void wait_until(neuronte_backend *b, double when)
{
    struct timespec ts;
    ts.tv_sec = (time_t)when;
    ts.tv_nsec = (long)((when - (double)ts.tv_sec) * 1e9);
    pthread_cond_timedwait(&b->cv, &b->mu, &ts);
}

static void *helper_main(void *arg)
{
    neuronte_backend *b = arg;
    neuronte_gpu *g;
    int orphan;
    pthread_mutex_lock(&b->mu);
    for (;;) {
        if (b->dead) {
            g = b->dead;
            b->dead = NULL;
            pthread_mutex_unlock(&b->mu);
            b->api->close(g);
            pthread_mutex_lock(&b->mu);
            continue;
        }
        if (b->quit)
            break;
        if (b->state == ST_OPEN && !b->busy &&
            (b->drop || (b->c.idle_close_ms > 0 && nb_now() - b->last_use >= b->c.idle_close_ms / 1000.0))) {
            g = b->gpu;
            b->gpu = NULL;
            b->state = ST_CLOSED;
            b->drop = b->want_open = 0;
            pthread_mutex_unlock(&b->mu);
            b->api->close(g);
            pthread_mutex_lock(&b->mu);
            continue;
        }
        if (b->want_open && b->state == ST_CLOSED) {
            char why[256] = "";
            b->state = ST_OPENING;
            pthread_mutex_unlock(&b->mu);
            g = open_gpu(b, why, sizeof why);
            pthread_mutex_lock(&b->mu);
            b->want_open = 0;
            if (g) {
                b->gpu = g;
                b->state = ST_OPEN;
                b->last_use = nb_now();
                snprintf(b->device, sizeof b->device, "%s", b->api->device(g));
            } else {
                b->state = ST_LOST;
                snprintf(b->lost, sizeof b->lost, "%s", why);
            }
            pthread_cond_broadcast(&b->cv);
            continue;
        }
        if (b->state == ST_OPEN && !b->busy && b->c.idle_close_ms > 0)
            wait_until(b, b->last_use + b->c.idle_close_ms / 1000.0);
        else
            pthread_cond_wait(&b->cv, &b->mu);
    }
    g = b->gpu;
    b->gpu = NULL;
    orphan = b->orphan;
    pthread_mutex_unlock(&b->mu);
    if (g)
        b->api->close(g);
    if (orphan)
        free_handle(b);
    return NULL;
}

int neuronte_backend_open(const char *nrt_path, int cell_w, int cell_h, neuronte_backend **out)
{
    neuronte_backend *b;
    pthread_condattr_t ca;
    float *probe;
    int rc;
    if (out)
        *out = NULL;
    if (!nrt_path || !out || cell_w < 1 || cell_h < 1 || strlen(nrt_path) >= sizeof b->path)
        return NEURONTE_ERR_ARG;
    b = calloc(1, sizeof *b);
    if (!b)
        return NEURONTE_ERR_NOMEM;
    rc = neuronte_open_file(nrt_path, &b->cpu);
    if (rc != NEURONTE_OK) {
        free(b);
        return rc;
    }
    /* a size the library does not take is refused here, as its coverage call would */
    probe = malloc((size_t)cell_w * cell_h * sizeof *probe);
    rc = probe ? neuronte_coverage(b->cpu, 0x20, cell_w, cell_h, 0, probe) : NEURONTE_ERR_NOMEM;
    free(probe);
    if (rc != NEURONTE_OK) {
        neuronte_close(b->cpu);
        free(b);
        return rc;
    }
    snprintf(b->path, sizeof b->path, "%s", nrt_path);
    b->cw = cell_w;
    b->ch = cell_h;
    b->plane = (size_t)cell_w * cell_h;
    pthread_mutex_init(&b->mu, NULL);
    pthread_condattr_init(&ca);
    pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
    pthread_cond_init(&b->cv, &ca);
    pthread_condattr_destroy(&ca);
    choose(nrt_path, &b->c);
    if (HAVE_CELLS) {
        /* STATE/cells/NAME.cells for NAME.nrt */
        char state[2048], cells[4096];
        const char *dir = nb_env("NEURONTE_CELLS_DIR"), *base = nb_base(nrt_path);
        size_t n = strlen(base);
        if (n > 4 && !strcmp(base + n - 4, ".nrt"))
            n -= 4;
        nb_system_state(state, sizeof state);
        if (dir)
            snprintf(cells, sizeof cells, "%s/%.*s.cells", dir, (int)n, base);
        else
            snprintf(cells, sizeof cells, "%s/cells/%.*s.cells", state, (int)n, base);
        if (neuronte_cells_open(cells, nrt_path, 0, &b->cells) != 0)
            b->cells = NULL;
    }
    *out = b;
    return NEURONTE_OK;
}

/* 1 when a call with `n` glyphs to compute is one for the GPU: setup chose one, the call is
 * large enough, the machine is not on battery and the device has not failed. The GPU is then
 * open or being opened; the call does not wait for it. */
static int gpu_wanted(neuronte_backend *b, int n)
{
    double now;
    int wanted;
    if (!b->c.gpu || n < b->c.min_glyphs)
        return 0;
    now = nb_now();
    if (b->power_at == 0 || now - b->power_at >= 1.0) {
        b->power_at = now;
        b->on_battery = battery_forbids(&b->c);
    }
    pthread_mutex_lock(&b->mu);
    if (b->on_battery) {
        if (b->state == ST_OPEN) {
            b->drop = 1; /* a context kept open is of no use on battery */
            pthread_cond_broadcast(&b->cv);
        }
        pthread_mutex_unlock(&b->mu);
        return 0;
    }
    if (b->state == ST_CLOSED && !b->want_open) {
        if (!b->helper_on) {
            sigset_t all, old;
            /* the helper takes none of the program's signals */
            sigfillset(&all);
            pthread_sigmask(SIG_BLOCK, &all, &old);
            b->helper_on = pthread_create(&b->helper, NULL, helper_main, b) == 0;
            pthread_sigmask(SIG_SETMASK, &old, NULL);
            if (!b->helper_on) {
                b->state = ST_LOST;
                snprintf(b->lost, sizeof b->lost, "no thread could be started for the GPU");
            }
        }
        if (b->helper_on) {
            b->want_open = 1;
            pthread_cond_broadcast(&b->cv);
        }
    }
    wanted = b->state != ST_LOST;
    pthread_mutex_unlock(&b->mu);
    return wanted;
}

/* 1 when the GPU is open: it is then marked busy until gpu_done(). -1 when it will not open. */
static int gpu_take(neuronte_backend *b)
{
    int r = 0;
    pthread_mutex_lock(&b->mu);
    if (b->state == ST_LOST)
        r = -1;
    else if (b->state == ST_OPEN && !b->drop) {
        b->busy = 1;
        r = 1;
    }
    pthread_mutex_unlock(&b->mu);
    return r;
}

static void gpu_done(neuronte_backend *b, int failed)
{
    pthread_mutex_lock(&b->mu);
    b->busy = 0;
    b->last_use = nb_now();
    if (failed) {
        /* the device is gone (reset, driver reloaded, a suspend): this handle draws on the CPU
         * from here on; a handle opened afterwards tries the GPU again */
        b->dead = b->gpu;
        b->gpu = NULL;
        b->state = ST_LOST;
        snprintf(b->lost, sizeof b->lost, "device lost: a GPU call failed; reopen the handle to try the GPU again");
    }
    pthread_cond_broadcast(&b->cv);
    pthread_mutex_unlock(&b->mu);
}

int neuronte_backend_coverage(neuronte_backend *b, const uint32_t *scalars, int count, float *out)
{
    const uint32_t *list = scalars;
    int n = count, i, rc, indirect = 0, wanted, done = 0;
    if (!b || count < 0 || (count > 0 && (!scalars || !out)))
        return NEURONTE_ERR_ARG;
    if (count == 0)
        return NEURONTE_OK;
    if (b->cells) {
        if (count > b->cap) {
            uint32_t *m = realloc(b->miss, (size_t)count * sizeof *m);
            int *a;
            if (!m)
                return NEURONTE_ERR_NOMEM;
            b->miss = m;
            a = realloc(b->miss_at, (size_t)count * sizeof *a);
            if (!a)
                return NEURONTE_ERR_NOMEM;
            b->miss_at = a;
            free(b->tmp);
            b->tmp = NULL;
            b->cap = count;
        }
        n = 0;
        for (i = 0; i < count; i++) {
            const float *have = neuronte_cells_find(b->cells, scalars[i], b->cw, b->ch, 0, 0, -1);
            if (have) {
                memcpy(out + (size_t)i * b->plane, have, b->plane * sizeof *out);
                b->from_cells++;
            } else {
                b->miss[n] = scalars[i];
                b->miss_at[n++] = i;
            }
        }
        if (n == 0)
            return NEURONTE_OK;
        if (n < count) {
            list = b->miss;
            indirect = 1;
        }
    }
    wanted = gpu_wanted(b, n);
    while (done < n) {
        int left = n - done, few;
        if (wanted && left >= b->c.min_glyphs) {
            int r = gpu_take(b);
            if (r > 0) {
                float *dst = out + (size_t)done * b->plane;
                if (indirect) {
                    if (!b->tmp)
                        b->tmp = malloc((size_t)b->cap * b->plane * sizeof *b->tmp);
                    dst = b->tmp;
                }
                rc = dst ? b->api->coverage(b->gpu, list + done, left, dst, NULL) : -1;
                gpu_done(b, rc != 0 && dst != NULL);
                if (rc == 0) {
                    if (indirect)
                        for (i = 0; i < left; i++)
                            memcpy(out + (size_t)b->miss_at[done + i] * b->plane, dst + (size_t)i * b->plane,
                                   b->plane * sizeof *out);
                    b->from_gpu += (uint64_t)left;
                    return NEURONTE_OK;
                }
                wanted = 0; /* the device failed under this call: the CPU answers it all the same */
            } else if (r < 0)
                wanted = 0;
        } else
            wanted = 0;
        /* on the CPU: all of it, or a few glyphs while the GPU is still opening */
        few = wanted && left > 4 ? 4 : left;
        for (i = done; i < done + few; i++) {
            rc = neuronte_coverage(b->cpu, list[i], b->cw, b->ch, 0, out + (size_t)(indirect ? b->miss_at[i] : i) * b->plane);
            if (rc != NEURONTE_OK)
                return rc;
            b->from_cpu++;
        }
        done += few;
    }
    return NEURONTE_OK;
}

void neuronte_backend_info(const neuronte_backend *cb, struct neuronte_backend_info *info)
{
    neuronte_backend *b = (neuronte_backend *)cb;
    if (!info)
        return;
    memset(info, 0, sizeof *info);
    snprintf(info->backend, sizeof info->backend, "cpu");
    if (!b)
        return;
    info->from_cells = b->from_cells;
    info->from_cpu = b->from_cpu;
    info->from_gpu = b->from_gpu;
    snprintf(info->reason, sizeof info->reason, "%s", b->c.reason);
    if (!b->c.gpu)
        return;
    pthread_mutex_lock(&b->mu);
    if (b->state == ST_LOST)
        snprintf(info->reason, sizeof info->reason, "%s", b->lost);
    else if (b->on_battery)
        snprintf(info->reason, sizeof info->reason, "on battery");
    else {
        info->gpu = 1;
        info->min_glyphs = b->c.min_glyphs;
        snprintf(info->backend, sizeof info->backend, "%s", b->c.backend);
        snprintf(info->device, sizeof info->device, "%s", b->device[0] ? b->device : b->c.device);
    }
    pthread_mutex_unlock(&b->mu);
}

void neuronte_backend_close(neuronte_backend *b)
{
    pthread_t helper;
    int orphan = 0;
    if (!b)
        return;
    neuronte_close(b->cpu);
    b->cpu = NULL;
    if (b->cells)
        neuronte_cells_close(b->cells);
    b->cells = NULL;
    if (!b->helper_on) {
        free_handle(b);
        return;
    }
    pthread_mutex_lock(&b->mu);
    b->quit = 1;
    /* a GPU that is still opening (kernels being compiled) is not waited for: the helper
     * finishes, closes it and frees the handle by itself */
    if (b->state == ST_OPENING)
        orphan = b->orphan = 1;
    helper = b->helper;
    pthread_cond_broadcast(&b->cv);
    pthread_mutex_unlock(&b->mu);
    if (orphan) {
        pthread_detach(helper); /* b is the helper's from here on */
        return;
    }
    pthread_join(helper, NULL);
    free_handle(b);
}

int neuronte_backend_choice(struct neuronte_backend_info *info)
{
    struct choice c;
    choose(NULL, &c);
    if (!info)
        return c.gpu && !battery_forbids(&c);
    memset(info, 0, sizeof *info);
    snprintf(info->backend, sizeof info->backend, "cpu");
    snprintf(info->reason, sizeof info->reason, "%s", c.reason);
    if (c.gpu && battery_forbids(&c))
        snprintf(info->reason, sizeof info->reason, "on battery");
    else if (c.gpu) {
        info->gpu = 1;
        info->min_glyphs = c.min_glyphs;
        snprintf(info->backend, sizeof info->backend, "%s", c.backend);
        snprintf(info->device, sizeof info->device, "%s", c.device);
    }
    return info->gpu;
}
