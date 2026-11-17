/* The two setup steps of the backend chooser, one program under two names.
 *
 *   10-backend check|run          measure the CPU and every GPU backend, write backend.conf
 *   40-backend-warm check|run     compile and cache the chosen backend's kernels for every file
 *   neuronte-backend-step --step NAME check|run     the same, named explicitly
 *
 * "check" exits 0 when the step is current and 1 when it is not, and writes
 * nothing. "run" makes it current; it may be run again at any time. What a
 * step was made from is written beside its output in STEP.stamp.
 *
 * Environment, set by neuronte-setup: NEURONTE_WEIGHTS (the directory of
 * .nrt files), NEURONTE_STATE (where this run keeps its files),
 * NEURONTE_THREADS, NEURONTE_FORCE=1 (make it again even when current),
 * NEURONTE_SETUP_SCOPE (system, or user: a user's own run, where
 * NEURONTE_STATE is that user's and backend.conf is read from
 * NEURONTE_SYSTEM_STATE and never written), NEURONTE_SETUP_BOOT=1 (a login
 * screen is waiting: a step that would measure or compile writes no stamp,
 * leaves "the CPU, for now" and exits 75, deferred; the run after the
 * start does the work).
 *
 * Settings in setup.conf (NEURONTE_SETUP_CONF, else /etc/neuronte/setup.conf):
 *   backend=cpu               never use a GPU on this machine: no GPU library is loaded
 *   backend.library=PATH      libneuronte_gpu, when it is not in a usual place
 *   backend.try=cuda,vulkan   the GPU backends to measure, in this order
 *   backend.files=A.nrt,...   the files to measure (default: the terminal table found)
 *   backend.cells=8x16,16x32  the cell sizes to measure
 *   backend.margin=1.3        how many times faster than one CPU thread a GPU must be
 *   backend.desktop_margin=2  the same for a GPU that also draws the desktop (whole tables only)
 *   backend.step_seconds=150  all of step 10's GPU work; backend.compile_seconds=60 for a
 *                             backend's first open (kernels may be compiled) and
 *                             backend.probe_seconds=60 for its comparing and timing, inside that
 *   backend.warm_seconds=300  all of step 40
 *   backend.max_batch=1024    glyphs per GPU call
 *   backend.covered=latin1    the characters cells files hold, which are compared but left out of
 *                             the timed calls (none: time them all); latin1 where the reference
 *                             library has cells files, else none. They are left out at every
 *                             size, not only at the sizes a cells file holds: this step runs
 *                             before the step that writes the cells files (30-cells), so it
 *                             cannot ask a file which sizes it has
 *   backend.idle_close_ms=1000  close the GPU after this long without a call (0: keep it open)
 *   backend.wake_ms=50        open a closed GPU once the CPU has worked this long on glyphs that
 *                             were for it (0: at the first such call)
 *   battery=cpu               on battery the CPU draws (default); a number N: only under N percent;
 *                             same: the battery changes nothing
 *
 * A GPU is only ever touched in a child process with a time limit, so a
 * driver that hangs or crashes costs that backend and nothing else.
 * C11 and POSIX; links the reference library and libdl. */
#include "common.h"

#include "neuronte.h"
#include "neuronte_gpu.h"

#include <dlfcn.h>
#include <libgen.h>
#include <link.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>

#include "decide.h"

#define EX_DEFERRED 75

/* ---- settings ---- */

static struct nb_kv setup, user_setup; /* the machine's, and over it the user's in a user's run */
static char weights_dir[2048], state_dir[2048], self_path[4096];
static int threads = 2, user_mode, boot_mode, force;

static const char *setting(const char *key, const char *dflt)
{
    const char *v = nb_kv_get(&user_setup, key, NULL);
    return v ? v : nb_kv_get(&setup, key, dflt);
}

static void say(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static int ends_with(const char *s, const char *tail)
{
    size_t a = strlen(s), b = strlen(tail);
    return a >= b && !strcmp(s + a - b, tail);
}

/* The .nrt files of the weights directory, sorted by name. */
static int list_weights(char ***out)
{
    struct dirent **names = NULL;
    char **list = NULL;
    int n = scandir(weights_dir, &names, NULL, alphasort), i, k = 0;
    if (n > 0)
        list = calloc((size_t)n, sizeof *list);
    for (i = 0; i < n; i++) {
        if (list && ends_with(names[i]->d_name, ".nrt"))
            list[k++] = strdup(names[i]->d_name);
        free(names[i]);
    }
    free(names);
    *out = list;
    return k;
}

static void weight_path(const char *name, char *out, size_t len)
{
    snprintf(out, len, "%s/%s", weights_dir, name);
}

/* Where libneuronte_gpu is, or "" when there is none. Only the file system is asked. */
static void gpu_library_path(char *out, size_t len)
{
    static const char *const places[] = {
        "/usr/lib/neuronte/libneuronte_gpu.so.0", "/usr/lib/x86_64-linux-gnu/libneuronte_gpu.so.0",
        "/usr/lib/aarch64-linux-gnu/libneuronte_gpu.so.0", "/usr/lib64/libneuronte_gpu.so.0",
        "/usr/local/lib/libneuronte_gpu.so.0", "/usr/lib/libneuronte_gpu.so.0", NULL};
    const char *v = setting("backend.library", NULL);
    int i;
    out[0] = 0;
    if (!v)
        v = nb_env("NEURONTE_GPU_LIB");
    if (v) {
        if (!access(v, R_OK))
            snprintf(out, len, "%s", v);
        return;
    }
    for (i = 0; places[i]; i++)
        if (!access(places[i], R_OK)) {
            snprintf(out, len, "%s", places[i]);
            return;
        }
}

/* ---- the stamp: what a step was made from ---- */

static int hash_loaded(struct dl_phdr_info *info, size_t size, void *data)
{
    struct nb_str *b = data;
    char h[65];
    (void)size;
    if (info->dlpi_name && strstr(info->dlpi_name, "libneuronte") && !strstr(info->dlpi_name, "libneuronte_gpu")) {
        nb_sha256_file(info->dlpi_name, h);
        nb_add(b, "input.libneuronte=%s\n", h);
    }
    return 0;
}

/* The lines every stamp of ours starts with: this program, the reference library where it is
 * a shared one, the GPU library, the settings, the machine. */
static void stamp_common(struct nb_str *b, const char *step)
{
    char h[65], lib[4096], v[1024];
    nb_add(b, "step=%s\nversion=%s\n", step, NB_STEP_VERSION);
    nb_sha256_file(self_path, h);
    nb_add(b, "input.self=%s\n", h);
    dl_iterate_phdr(hash_loaded, b);
    gpu_library_path(lib, sizeof lib);
    if (lib[0])
        nb_sha256_file(lib, h);
    nb_add(b, "input.gpu_library=%s\n", lib[0] ? h : "absent");
    nb_sha256_file(nb_setup_conf_path(), h);
    nb_add(b, "input.setup_conf=%s\n", h[0] ? h : "absent");
    nb_gpu_list(v, sizeof v);
    nb_add(b, "machine.gpus=%s\n", v);
    nb_driver_id("cuda", v, sizeof v);
    nb_add(b, "machine.driver.cuda=%s\n", v);
    nb_driver_id("vulkan", v, sizeof v);
    nb_add(b, "machine.driver.vulkan=%s\n", v);
    nb_kernel_release(v, sizeof v);
    nb_add(b, "machine.kernel=%s\n", v);
}

/* 1 when every line of `want` that starts with "input." or "machine." or "step=" or "version="
 * is in the stamp file with the same value. */
static int stamp_matches(const char *stamp_path, const char *want)
{
    struct nb_kv have;
    char *copy = strdup(want), *line, *save = NULL;
    int ok = nb_kv_read(stamp_path, &have) == 0;
    for (line = strtok_r(copy, "\n", &save); ok && line; line = strtok_r(NULL, "\n", &save)) {
        char *eq = strchr(line, '=');
        const char *v;
        if (!eq)
            continue;
        *eq = 0;
        v = nb_kv_get(&have, line, NULL);
        if (!v || strcmp(v, eq + 1))
            ok = 0;
    }
    free(copy);
    nb_kv_free(&have);
    return ok;
}

#ifdef NEURONTE_BACKEND_CELLS
#define COVERED_DEFAULT "latin1"
#else
#define COVERED_DEFAULT "none"
#endif

/* ---- a file's characters ---- */

/* The scalars a file has trained, in order, at most `cap`. */
static int trained(neuronte *n, uint32_t *out, int cap)
{
    uint32_t s;
    int k = 0;
    /* a file without a table of trained characters draws anything: a short list will do */
    if (neuronte_untrained(n, 0x10FFFE) == 0 && neuronte_untrained(n, 0xFFFE) == 0 && cap > 256)
        cap = 256;
    for (s = 0x20; s <= 0x10FFFF && k < cap; s++) {
        if (s == 0xD800)
            s = 0xE000;
        if (neuronte_untrained(n, s) == 0)
            out[k++] = s;
    }
    return k;
}

/* ---- measuring ---- */

#define NSIZES NB_SIZES
struct cell {
    int w, h;
};
struct work {
    char path[4096], name[256];
    struct cell cell;
    uint32_t *scalars;
    int count;
    int timed;         /* the first `timed` of them are the ones no cells file holds: only they are timed */
    float *ref;        /* count planes from the CPU library */
    int sizes[NSIZES]; /* 1, 16, 64, 256, timed */
};

static uint32_t rng_state = 2463534242u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    return n & 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

/* How often a size is timed: small calls often, the whole table a few times. */
static int repeats_of(int size_index)
{
    static const int r[NSIZES] = {16, 8, 4, 3, 3};
    return r[size_index];
}

/* A GPU backend is timed in ROUNDS rounds. In each, every size is timed on the CPU and then on
 * the GPU with the same calls, so that other work on the machine falls on both; the backend has
 * to win in every round. How often a size is timed in one round: */
#define ROUNDS 3
static int round_repeats(int size_index)
{
    static const int r[NSIZES] = {6, 3, 2, 1, 1};
    return r[size_index];
}

/* The i-th call of `size` glyphs: characters spread evenly over the timed part of the table,
 * from a start that moves with i, so that a small call mixes the file's nets as the table does
 * and no two calls are the same. The largest call is that part in order. */
static const uint32_t *window(const struct work *w, int size, int i, uint32_t *list)
{
    int j, start = (int)(((long long)i * 7919) % w->timed);
    if (size >= w->timed)
        return w->scalars;
    for (j = 0; j < size; j++)
        list[j] = w->scalars[(start + (int)((long long)j * w->timed / size)) % w->timed];
    return list;
}

/* One CPU thread: microseconds per glyph for each size, and the reference planes on the way. */
static int measure_cpu(struct work *w, double us[NSIZES])
{
    neuronte *n;
    size_t plane = (size_t)w->cell.w * w->cell.h;
    float *scratch;
    uint32_t *pick;
    int s, i, j;
    if (neuronte_open_file(w->path, &n) != NEURONTE_OK)
        return -1;
    scratch = malloc((size_t)w->count * plane * sizeof *scratch);
    pick = malloc((size_t)w->count * sizeof *pick);
    w->ref = malloc((size_t)w->count * plane * sizeof *w->ref);
    if (!scratch || !pick || !w->ref)
        return -1;
    for (s = 0; s < NSIZES; s++) {
        double t[32];
        int size = w->sizes[s], reps = s == NSIZES - 1 ? 2 : repeats_of(s);
        if (size > w->timed) {
            us[s] = -1;
            continue;
        }
        for (i = 0; i < reps; i++) {
            const uint32_t *list = window(w, size, i, pick);
            float *out = s == NSIZES - 1 && i == 0 ? w->ref : scratch;
            double t0 = nb_now();
            for (j = 0; j < size; j++)
                if (neuronte_coverage(n, list[j], w->cell.w, w->cell.h, 0, out + (size_t)j * plane) != NEURONTE_OK)
                    return -1;
            t[i] = (nb_now() - t0) * 1e6 / size;
        }
        us[s] = median(t, reps);
    }
    /* the reference planes of the characters that are compared but not timed */
    for (j = w->timed; j < w->count; j++)
        if (neuronte_coverage(n, w->scalars[j], w->cell.w, w->cell.h, 0, w->ref + (size_t)j * plane) != NEURONTE_OK)
            return -1;
    free(scratch);
    free(pick);
    neuronte_close(n);
    return 0;
}

struct slice {
    const struct work *w;
    int first, count;
    int failed;
};

static void *cpu_slice(void *arg)
{
    struct slice *sl = arg;
    neuronte *n;
    size_t plane = (size_t)sl->w->cell.w * sl->w->cell.h;
    float *out = malloc(plane * sizeof *out);
    int j;
    if (!out || neuronte_open_file(sl->w->path, &n) != NEURONTE_OK) {
        sl->failed = 1;
        return NULL;
    }
    for (j = 0; j < sl->count; j++)
        neuronte_coverage(n, sl->w->scalars[sl->first + j], sl->w->cell.w, sl->w->cell.h, 0, out);
    neuronte_close(n);
    free(out);
    return NULL;
}

/* The timed part of the table on `threads` threads, a handle each: microseconds per glyph. */
static double measure_cpu_threads(const struct work *w)
{
    pthread_t th[64];
    struct slice sl[64];
    int nt = threads < 1 ? 1 : threads > 64 ? 64 : threads, i, per = (w->timed + nt - 1) / nt;
    double t0 = nb_now();
    for (i = 0; i < nt; i++) {
        sl[i].w = w;
        sl[i].first = i * per < w->timed ? i * per : w->timed;
        sl[i].count = sl[i].first + per <= w->timed ? per : w->timed - sl[i].first;
        sl[i].failed = 0;
        pthread_create(&th[i], NULL, cpu_slice, &sl[i]);
    }
    for (i = 0; i < nt; i++)
        pthread_join(th[i], NULL);
    return (nb_now() - t0) * 1e6 / w->timed;
}

/* ---- the GPU library, loaded in a child only ---- */

struct gpu_lib {
    void *dl;
    int (*abi)(void);
    int (*open)(const struct neuronte_gpu_options *, neuronte_gpu **, char *, size_t);
    void (*close)(neuronte_gpu *);
    int (*coverage)(neuronte_gpu *, const uint32_t *, int, float *, struct neuronte_gpu_timing *);
    const char *(*device)(const neuronte_gpu *);
    const char *(*device_type)(const neuronte_gpu *);
    int (*backend)(const neuronte_gpu *);
    const char *(*backend_name)(int);
};

static int gpu_load(const char *path, struct gpu_lib *g, char *why, size_t len)
{
    memset(g, 0, sizeof *g);
    g->dl = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!g->dl) {
        snprintf(why, len, "cannot load %s: %s", path, dlerror());
        return -1;
    }
    g->abi = (int (*)(void))dlsym(g->dl, "neuronte_gpu_abi");
    g->open = (int (*)(const struct neuronte_gpu_options *, neuronte_gpu **, char *, size_t))dlsym(g->dl, "neuronte_gpu_open");
    g->close = (void (*)(neuronte_gpu *))dlsym(g->dl, "neuronte_gpu_close");
    g->coverage = (int (*)(neuronte_gpu *, const uint32_t *, int, float *, struct neuronte_gpu_timing *))dlsym(
        g->dl, "neuronte_gpu_coverage");
    g->device = (const char *(*)(const neuronte_gpu *))dlsym(g->dl, "neuronte_gpu_device");
    g->device_type = (const char *(*)(const neuronte_gpu *))dlsym(g->dl, "neuronte_gpu_device_type");
    g->backend = (int (*)(const neuronte_gpu *))dlsym(g->dl, "neuronte_gpu_backend");
    g->backend_name = (const char *(*)(int))dlsym(g->dl, "neuronte_gpu_backend_name");
    if (!g->abi || !g->open || !g->close || !g->coverage || !g->device || !g->device_type || !g->backend ||
        !g->backend_name) {
        snprintf(why, len, "%s is not a libneuronte_gpu this program knows (a call is missing)", path);
        return -1;
    }
    if (g->abi() != NEURONTE_GPU_ABI) {
        snprintf(why, len, "%s has interface %d, this program was built for %d", path, g->abi(), NEURONTE_GPU_ABI);
        return -1;
    }
    return 0;
}

static int backend_number(const char *name)
{
    return !strcmp(name, "cuda") ? NEURONTE_GPU_CUDA : !strcmp(name, "vulkan") ? NEURONTE_GPU_VULKAN
         : !strcmp(name, "opencl") ? NEURONTE_GPU_OPENCL : !strcmp(name, "opengl") ? NEURONTE_GPU_OPENGL : -1;
}

/* A call of any size, cut into calls of at most max_batch as the loader cuts them. */
static int gpu_cover(const struct gpu_lib *g, neuronte_gpu *h, const uint32_t *scalars, int count, float *out,
                     size_t plane, int max_batch)
{
    int at;
    for (at = 0; at < count; at += max_batch) {
        int n = count - at < max_batch ? count - at : max_batch;
        if (g->coverage(h, scalars + at, n, out + (size_t)at * plane, NULL))
            return -1;
    }
    return 0;
}

static void clean(char *s)
{
    for (; *s; s++)
        if (*s == '\n' || *s == '\r')
            *s = ' ';
}

/* The child's work for one backend: open every work item, compare many calls with the CPU's
 * planes, time the sizes. It reports in key=value lines on `fd`. */
static void probe_child(int fd, const char *lib_path, const char *backend, struct work *works, int nworks,
                        int max_batch)
{
    FILE *out = fdopen(fd, "w");
    struct gpu_lib g;
    char why[512] = "";
    int k;
    setvbuf(out, NULL, _IOLBF, 0);
    if (gpu_load(lib_path, &g, why, sizeof why)) {
        fprintf(out, "error=%s\n", why);
        exit(0);
    }
    for (k = 0; k < nworks; k++) {
        struct work *w = &works[k];
        struct neuronte_gpu_options opt = {0};
        neuronte_gpu *h = NULL;
        size_t plane = (size_t)w->cell.w * w->cell.h;
        float *got = malloc((size_t)w->count * plane * sizeof *got);
        uint32_t *pick = malloc((size_t)w->count * sizeof *pick);
        int calls = 0, differing = 0, s, i, j;
        double t0 = nb_now();
        opt.nrt_path = w->path;
        opt.cell_w = w->cell.w;
        opt.cell_h = w->cell.h;
        opt.backend = backend_number(backend);
        opt.flags = NEURONTE_GPU_FILE;
        opt.max_batch = max_batch;
        if (g.open(&opt, &h, why, sizeof why) || !h) {
            clean(why);
            fprintf(out, "error=%s\n", why[0] ? why : "open failed");
            exit(0);
        }
        if (k == 0) {
            snprintf(why, sizeof why, "%s", g.device(h));
            clean(why);
            fprintf(out, "device=%s\ndevice_type=%s\nbackend=%s\n", why, g.device_type(h), g.backend_name(g.backend(h)));
        }
        fprintf(out, "open_s.%d=%.3f\nopened=%d\n", k, nb_now() - t0, k);
        /* the soak: calls of 1 to 8, eight random subsets, the whole table twice */
        for (i = 0; i < 18; i++) {
            int count = i < 8 ? i + 1 : i < 16 ? 1 + (int)(rng() % (uint32_t)w->count) : w->count;
            if (count > w->count)
                count = w->count;
            for (j = 0; j < count; j++)
                pick[j] = i < 16 ? (uint32_t)(rng() % (uint32_t)w->count) : (uint32_t)j;
            {
                uint32_t *list = malloc((size_t)count * sizeof *list);
                int bad = 0;
                for (j = 0; j < count; j++)
                    list[j] = w->scalars[pick[j]];
                memset(got, 0xff, (size_t)count * plane * sizeof *got);
                if (gpu_cover(&g, h, list, count, got, plane, max_batch)) {
                    fprintf(out, "error=a call of %d glyphs failed\n", count);
                    exit(0);
                }
                for (j = 0; j < count && !bad; j++)
                    bad = memcmp(got + (size_t)j * plane, w->ref + (size_t)pick[j] * plane, plane * sizeof *got) != 0;
                free(list);
                calls++;
                differing += bad;
            }
        }
        fprintf(out, "soak.%d=%d %d\n", k, calls, differing);
        {
            double gt[NSIZES][ROUNDS], ct[NSIZES][ROUNDS];
            neuronte *cpu = NULL;
            int r;
            if (neuronte_open_file(w->path, &cpu) != NEURONTE_OK) {
                fprintf(out, "error=the library cannot open the file\n");
                exit(0);
            }
            for (r = 0; r < ROUNDS; r++)
                for (s = 0; s < NSIZES; s++) {
                    double t[32];
                    int size = w->sizes[s], reps = round_repeats(s);
                    if (size > w->timed)
                        continue;
                    for (i = 0; i < reps; i++) {
                        const uint32_t *list = window(w, size, r * reps + i, pick);
                        double c0 = nb_now();
                        for (j = 0; j < size; j++)
                            neuronte_coverage(cpu, list[j], w->cell.w, w->cell.h, 0, got + (size_t)j * plane);
                        t[i] = (nb_now() - c0) * 1e6 / size;
                    }
                    ct[s][r] = median(t, reps);
                    for (i = 0; i < reps; i++) {
                        const uint32_t *list = window(w, size, r * reps + i, pick);
                        double c0 = nb_now();
                        if (gpu_cover(&g, h, list, size, got, plane, max_batch)) {
                            fprintf(out, "error=a call of %d glyphs failed\n", size);
                            exit(0);
                        }
                        t[i] = (nb_now() - c0) * 1e6 / size;
                    }
                    gt[s][r] = median(t, reps);
                }
            neuronte_close(cpu);
            for (s = 0; s < NSIZES; s++) {
                double worst = 0;
                if (w->sizes[s] > w->timed)
                    continue;
                for (r = 0; r < ROUNDS; r++)
                    if (gt[s][r] > 0 && (worst == 0 || ct[s][r] / gt[s][r] < worst))
                        worst = ct[s][r] / gt[s][r];
                /* the worst round before the medians, which sort the rounds */
                fprintf(out, "w.%d.%d=%.4f\n", k, s, worst);
                fprintf(out, "t.%d.%d=%.3f\n", k, s, median(gt[s], ROUNDS));
                fprintf(out, "c.%d.%d=%.3f\n", k, s, median(ct[s], ROUNDS));
            }
        }
        g.close(h);
        free(got);
        free(pick);
    }
    fprintf(out, "done=1\n");
    fflush(out);
    exit(0);
}

/* Run fn in a child and read its key=value lines into kv. The child has `first_limit` seconds
 * until its first "opened" line (kernels may be compiled) and `limit` seconds after it, and in
 * no case longer than until `hard` (a time of nb_now(); 0: no such end). *timed_out says whether
 * it was killed. */
static void run_child(void (*fn)(int fd, void *arg), void *arg, double first_limit, double limit, double hard,
                      struct nb_kv *kv, int *timed_out)
{
    int p[2];
    pid_t pid;
    struct nb_str b = {0};
    double deadline = nb_now() + first_limit;
    int opened = 0, status;
    *timed_out = 0;
    memset(kv, 0, sizeof *kv);
    nb_add(&b, "%s", "");
    if (pipe(p)) {
        free(b.s);
        return;
    }
    fflush(NULL);
    pid = fork();
    if (pid == 0) {
        close(p[0]);
        fn(p[1], arg);
        _exit(0);
    }
    close(p[1]);
    for (;;) {
        struct pollfd pf = {p[0], POLLIN, 0};
        char buf[4096];
        double left;
        ssize_t n;
        if (hard > 0 && deadline > hard)
            deadline = hard;
        left = deadline - nb_now();
        if (left <= 0) {
            *timed_out = 1;
            break;
        }
        if (poll(&pf, 1, (int)(left * 1000) + 1) <= 0)
            continue;
        n = read(p[0], buf, sizeof buf - 1);
        if (n <= 0)
            break;
        buf[n] = 0;
        nb_add(&b, "%s", buf);
        if (!opened && strstr(b.s, "opened=")) {
            opened = 1;
            deadline = nb_now() + limit;
        }
    }
    close(p[0]);
    if (*timed_out)
        kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    if (!*timed_out && WIFSIGNALED(status))
        nb_add(&b, "error=the child was ended by signal %d\n", WTERMSIG(status));
    if (b.s)
        nb_kv_parse(b.s, kv);
}

struct probe_arg {
    const char *lib, *backend;
    struct work *works;
    int nworks, max_batch;
};

static void probe_entry(int fd, void *arg)
{
    struct probe_arg *a = arg;
    probe_child(fd, a->lib, a->backend, a->works, a->nworks, a->max_batch);
}

/* Whether the GPU a backend found also drives a screen: a connected output on a graphics card
 * of that maker. A heuristic from sysfs. */
static int draws_desktop(const char *device_name, const char *device_type)
{
    const char *root = nb_env("NEURONTE_BACKEND_TEST_DRM");
    const char *vendor = strstr(device_name, "NVIDIA") || strstr(device_name, "Quadro") || strstr(device_name, "GeForce") ||
                         strstr(device_name, "RTX") || strstr(device_name, "GTX") ? "0x10de"
                       : strstr(device_name, "Intel") ? "0x8086"
                       : strstr(device_name, "AMD") || strstr(device_name, "Radeon") ? "0x1002" : NULL;
    struct dirent *e;
    DIR *d;
    int yes = 0, known = 0;
    if (!root)
        root = "/sys/class/drm";
    d = opendir(root);
    while (d && (e = readdir(d))) {
        char path[1024], v[64], card[64];
        const char *dash = strchr(e->d_name, '-');
        if (strncmp(e->d_name, "card", 4) || !dash)
            continue;
        snprintf(card, sizeof card, "%.*s", (int)(dash - e->d_name), e->d_name);
        snprintf(path, sizeof path, "%s/%s/device/vendor", root, card);
        nb_first_line(path, v, sizeof v);
        if (!vendor || strcmp(v, vendor))
            continue;
        known = 1;
        snprintf(path, sizeof path, "%s/%s/status", root, e->d_name);
        nb_first_line(path, v, sizeof v);
        if (!strcmp(v, "connected"))
            yes = 1;
    }
    if (d)
        closedir(d);
    /* nothing known about its outputs: an integrated GPU is taken to draw the desktop */
    return known ? yes : !strcmp(device_type, "integrated-gpu");
}

/* ---- step 10 ---- */

static void conf_path_here(char *out, size_t len)
{
    snprintf(out, len, "%s/backend.conf", state_dir);
}

static void now_text(char *out, size_t len)
{
    time_t now = time(NULL);
    struct tm tm;
    gmtime_r(&now, &tm);
    strftime(out, len, "%Y-%m-%dT%H:%M:%SZ", &tm);
}

/* Pick the files to measure: those named in the settings, else the terminal table
 * (neuronte-split-v*.nrt, the last by name), else the largest file of one layer with 32
 * trained characters. Only the CPU library looks at them. */
static int pick_files(char picked[][256], int max)
{
    const char *named = setting("backend.files", NULL);
    char **names, path[4096];
    int n = list_weights(&names), i, k = 0, best = -1;
    off_t best_size = -1;
    if (named) {
        char *copy = strdup(named), *save = NULL, *t;
        for (t = strtok_r(copy, ", ", &save); t && k < max; t = strtok_r(NULL, ", ", &save))
            snprintf(picked[k++], 256, "%s", t);
        free(copy);
    } else {
        for (i = 0; i < n; i++)
            if (!strncmp(names[i], "neuronte-split-v", 16))
                best = i;
        if (best < 0)
            for (i = 0; i < n; i++) {
                struct stat st;
                neuronte *lib;
                uint32_t few[32];
                weight_path(names[i], path, sizeof path);
                if (stat(path, &st) || st.st_size <= best_size || neuronte_open_file(path, &lib) != NEURONTE_OK)
                    continue;
                if (neuronte_nets(lib) > 0 && neuronte_layers(lib) <= 1 && trained(lib, few, 32) == 32) {
                    best = i;
                    best_size = st.st_size;
                }
                neuronte_close(lib);
            }
        if (best >= 0)
            snprintf(picked[k++], 256, "%s", names[best]);
    }
    for (i = 0; i < n; i++)
        free(names[i]);
    free(names);
    return k;
}

/* What step 10's result depends on. */
static char *stamp_10(void)
{
    struct nb_str b = {0};
    char picked[4][256], path[4096], h[65];
    int n = pick_files(picked, 4), i;
    stamp_common(&b, "10-backend");
    nb_add(&b, "input.files=%d\n", n);
    for (i = 0; i < n; i++) {
        weight_path(picked[i], path, sizeof path);
        nb_sha256_file(path, h);
        nb_add(&b, "input.file.%s=%s\n", picked[i], h[0] ? h : "absent");
    }
    return b.s;
}

static void conf_head(struct nb_str *c)
{
    char when[64];
    now_text(when, sizeof when);
    nb_add(c, "# written by the setup step 10-backend; settings belong in setup.conf, not here\nversion=%d\nwritten=%s\n",
           NB_CONF_VERSION, when);
}

static void conf_cpu(struct nb_str *c, const char *reason)
{
    nb_add(c, "use.single=cpu\nuse.burst=cpu\nuse.bulk=cpu\nuse.bulk_vs_threads=cpu\nmin_glyphs=0\nbattery=%s\nreason=%s\n",
           setting("battery", "cpu"), reason);
}

/* "The CPU, for now": what a start leaves while the measuring waits for the run after it. */
static int write_pending(void)
{
    struct nb_str c = {0};
    char path[4096];
    int rc;
    conf_head(&c);
    conf_cpu(&c, "pending: to be measured after the start; the CPU draws meanwhile");
    conf_path_here(path, sizeof path);
    rc = nb_mkdirs(state_dir) || nb_write_atomic(path, c.s);
    free(c.s);
    return rc;
}

static int finish_10(struct nb_str *c, const char *stamp, double started)
{
    struct nb_str st = {0};
    char conf[4096], stamp_path[4096], h[65], when[64];
    int rc = 0;
    conf_path_here(conf, sizeof conf);
    snprintf(stamp_path, sizeof stamp_path, "%s/10-backend.stamp", state_dir);
    if (nb_write_atomic(conf, c->s)) {
        say("10-backend: cannot write %s", conf);
        return 1;
    }
    nb_sha256_file(conf, h);
    now_text(when, sizeof when);
    nb_add(&st, "%soutput.backend_conf=%s\nmade=%s\nseconds=%.1f\n", stamp, h, when, nb_now() - started);
    if (nb_write_atomic(stamp_path, st.s)) {
        say("10-backend: cannot write %s", stamp_path);
        rc = 1;
    }
    free(st.s);
    return rc;
}

/* 1 when the answer needs no measuring: a machine set never to use a GPU, or one with no
 * GPU library. The reason is written to why. */
static int cpu_without_measuring(char *why, size_t len)
{
    char lib[4096];
    if (!strcmp(setting("backend", "auto"), "cpu")) {
        snprintf(why, len, "backend=cpu in the settings: no GPU is used on this machine");
        return 1;
    }
    gpu_library_path(lib, sizeof lib);
    if (!lib[0]) {
        snprintf(why, len, "no GPU library (libneuronte_gpu) is installed");
        return 1;
    }
    return 0;
}

static int run_10(void)
{
    static const char *const size_name[NSIZES] = {"single", "burst", "batch", "block", "bulk"};
    static struct work works[NB_WORKS];
    static struct nb_decide_in in;
    struct nb_decide_out outd;
    char lib[4096], picked[4][256], why[512], devices[NB_BACKENDS][256];
    char *stamp = stamp_10(), *save = NULL, *t, *copy;
    struct nb_str c = {0}, m = {0}, notes = {0};
    struct cell cells[4];
    int ncells = 0, nfiles, nworks = 0, i, k, s, rc, max_batch = atoi(setting("backend.max_batch", "1024"));
    double started = nb_now(), hard;
    char load_before[128] = "";

    if (nb_mkdirs(state_dir)) {
        say("10-backend: cannot make %s", state_dir);
        return 1;
    }
    conf_head(&c);
    if (cpu_without_measuring(why, sizeof why)) {
        conf_cpu(&c, why);
        rc = finish_10(&c, stamp, started);
        free(c.s);
        free(stamp);
        return rc;
    }
    if (max_batch < 1)
        max_batch = 1024;
    gpu_library_path(lib, sizeof lib);
    nb_add(&m, "%s", "");
    nb_add(&notes, "%s", "");
    memset(&in, 0, sizeof in);
    in.margin = atof(setting("backend.margin", "1.3"));
    in.desktop_margin = atof(setting("backend.desktop_margin", "2"));
    if (in.margin < 1)
        in.margin = 1.3;
    if (in.desktop_margin < in.margin)
        in.desktop_margin = in.margin;

    /* the work: files at cell sizes, with the CPU's planes and times */
    nb_first_line("/proc/loadavg", load_before, sizeof load_before);
    copy = strdup(setting("backend.cells", "8x16,16x32"));
    for (t = strtok_r(copy, ", ", &save); t && ncells < 4; t = strtok_r(NULL, ", ", &save))
        if (sscanf(t, "%dx%d", &cells[ncells].w, &cells[ncells].h) == 2 && cells[ncells].w > 0 && cells[ncells].h > 0 &&
            cells[ncells].w <= 256 && cells[ncells].h <= 256)
            ncells++;
    free(copy);
    nfiles = pick_files(picked, 4);
    for (i = 0; i < nfiles; i++)
        for (k = 0; k < ncells && nworks < NB_WORKS; k++) {
            struct work *w = &works[nworks];
            neuronte *n;
            memset(w, 0, sizeof *w);
            snprintf(w->name, sizeof w->name, "%s", picked[i]);
            weight_path(picked[i], w->path, sizeof w->path);
            w->cell = cells[k];
            if (neuronte_open_file(w->path, &n) != NEURONTE_OK) {
                say("10-backend: the library cannot open %s", w->path);
                continue;
            }
            w->scalars = malloc(4096 * sizeof *w->scalars);
            w->count = w->scalars && neuronte_nets(n) > 0 ? trained(n, w->scalars, 4096) : 0;
            neuronte_close(n);
            if (w->count < 16)
                continue;
            w->timed = w->count;
            if (!strcmp(setting("backend.covered", COVERED_DEFAULT), "latin1")) {
                /* the characters the cells files hold are answered without a net: they are
                 * compared like the rest, but timed calls are made of the others */
                int held = 0;
                while (held < w->count && w->scalars[held] <= 0xFF)
                    held++;
                if (held > 0 && w->count - held >= 16) {
                    uint32_t *first = malloc((size_t)held * sizeof *first);
                    if (first) {
                        memcpy(first, w->scalars, (size_t)held * sizeof *first);
                        memmove(w->scalars, w->scalars + held, (size_t)(w->count - held) * sizeof *first);
                        memcpy(w->scalars + w->count - held, first, (size_t)held * sizeof *first);
                        free(first);
                        w->timed = w->count - held;
                    }
                }
            }
            w->sizes[0] = 1;
            w->sizes[1] = 16;
            w->sizes[2] = 64;
            w->sizes[3] = 256;
            w->sizes[4] = w->timed;
            memset(in.cpu1[nworks], 0, sizeof in.cpu1[nworks]);
            if (measure_cpu(w, in.cpu1[nworks])) {
                say("10-backend: the library cannot draw %s", w->name);
                continue;
            }
            in.whole[nworks] = w->timed;
            in.cpun[nworks] = measure_cpu_threads(w);
            for (s = 0; s < NSIZES; s++)
                if (in.cpu1[nworks][s] > 0)
                    nb_add(&m, "measure.cpu1.%s.%dx%d.%s=%.1f\n", w->name, w->cell.w, w->cell.h, size_name[s], in.cpu1[nworks][s]);
            nb_add(&m, "measure.cpu%d.%s.%dx%d.bulk=%.1f\n", threads, w->name, w->cell.w, w->cell.h, in.cpun[nworks]);
            if (k == 0) {
                nb_add(&m, "measure.glyphs.%s=%d\n", w->name, w->timed);
                if (w->timed < w->count)
                    nb_add(&m, "measure.not_timed.%s=%d\n", w->name, w->count - w->timed);
            }
            nworks++;
        }
    in.nworks = nworks;

    if (!nworks) {
        snprintf(why, sizeof why, "no file to measure in the weights directory");
        conf_cpu(&c, why);
    } else {
        /* the GPU work has its own clock: the CPU's part is over and cannot hang */
        hard = nb_now() + atof(setting("backend.step_seconds", "150"));
        copy = strdup(setting("backend.try", "cuda,vulkan"));
        for (t = strtok_r(copy, ", ", &save); t && in.nbackends < NB_BACKENDS; t = strtok_r(NULL, ", ", &save)) {
            struct probe_arg a = {lib, t, works, nworks, max_batch};
            struct nb_decide_backend *b = &in.backend[in.nbackends];
            struct nb_kv kv;
            const char *err;
            int timed_out, calls = 0, differing = 0;
            double t0 = nb_now();
            if (backend_number(t) < 0) {
                nb_add(&notes, "%s: not a backend this step knows; ", t);
                continue;
            }
            if (hard - t0 < 1) {
                nb_add(&notes, "%s: not tried, the step's time was used up; ", t);
                nb_add(&m, "probe.%s=not tried: the step's time was used up\n", t);
                continue;
            }
            run_child(probe_entry, &a, atof(setting("backend.compile_seconds", "60")), atof(setting("backend.probe_seconds", "60")),
                      hard, &kv, &timed_out);
            err = nb_kv_get(&kv, "error", NULL);
            if (timed_out || err || !nb_kv_get(&kv, "done", NULL)) {
                const char *what = timed_out ? "stopped: it did not finish in the time allowed" : err ? err : "its process ended early";
                nb_add(&notes, "%s: %s; ", t, what);
                nb_add(&m, "probe.%s=%s (%.1f s)\n", t, what, nb_now() - t0);
                nb_kv_free(&kv);
                continue;
            }
            memset(b, 0, sizeof *b);
            snprintf(b->name, sizeof b->name, "%s", t);
            snprintf(devices[in.nbackends], sizeof devices[0], "%s", nb_kv_get(&kv, "device", ""));
            b->exact = 1;
            for (k = 0; k < nworks; k++) {
                char key[64];
                int a1 = 0, a2 = 0;
                snprintf(key, sizeof key, "soak.%d", k);
                if (sscanf(nb_kv_get(&kv, key, "0 1"), "%d %d", &a1, &a2) != 2 || a1 < 1 || a2 != 0)
                    b->exact = 0;
                calls += a1;
                differing += a2;
                for (s = 0; s < NSIZES; s++) {
                    snprintf(key, sizeof key, "t.%d.%d", k, s);
                    b->gpu[k][s] = atof(nb_kv_get(&kv, key, "-1"));
                    snprintf(key, sizeof key, "c.%d.%d", k, s);
                    b->cpu[k][s] = atof(nb_kv_get(&kv, key, "0"));
                    snprintf(key, sizeof key, "w.%d.%d", k, s);
                    b->worst[k][s] = atof(nb_kv_get(&kv, key, "0"));
                    if (b->gpu[k][s] > 0)
                        nb_add(&m, "measure.gpu-%s.%s.%dx%d.%s=%.1f\n", t, works[k].name, works[k].cell.w, works[k].cell.h,
                               size_name[s], b->gpu[k][s]);
                    if (b->gpu[k][s] > 0 && b->cpu[k][s] > 0 && b->worst[k][s] > 0)
                        nb_add(&m, "measure.gpu-%s.%s.%dx%d.%s.times_cpu=%.2f\nmeasure.gpu-%s.%s.%dx%d.%s.worst_round=%.2f\n", t,
                               works[k].name, works[k].cell.w, works[k].cell.h, size_name[s], b->cpu[k][s] / b->gpu[k][s], t,
                               works[k].name, works[k].cell.w, works[k].cell.h, size_name[s], b->worst[k][s]);
                }
                snprintf(key, sizeof key, "open_s.%d", k);
                nb_add(&m, "measure.gpu-%s.%s.%dx%d.open_s=%s\n", t, works[k].name, works[k].cell.w, works[k].cell.h,
                       nb_kv_get(&kv, key, "?"));
            }
            b->draws_desktop = draws_desktop(devices[in.nbackends], nb_kv_get(&kv, "device_type", ""));
            nb_add(&m, "probe.%s=%s on %s; %d calls compared with the CPU, %d differing%s (%.1f s)\n", t,
                   nb_kv_get(&kv, "backend", t), devices[in.nbackends], calls, differing,
                   b->draws_desktop ? "; this GPU also draws the desktop" : "", nb_now() - t0);
            in.nbackends++;
            nb_kv_free(&kv);
        }
        free(copy);
        nb_decide(&in, &outd);
        if (outd.chosen >= 0) {
            const struct nb_decide_backend *b = &in.backend[outd.chosen];
            char drv[1024], id[64];
            nb_driver_id(b->name, drv, sizeof drv);
            nb_file_id(lib, id, sizeof id);
            nb_add(&c, "gpu.library=%s\ngpu.library.id=%s\ngpu.backend=%s\ngpu.device=%s\ngpu.driver=%s\n", lib, id, b->name,
                   devices[outd.chosen], drv);
            nb_add(&c, "gpu.identical=yes\ngpu.draws_desktop=%s\ngpu.max_batch=%d\n", b->draws_desktop ? "yes" : "no", max_batch);
            nb_add(&c, "idle_close_ms=%d\nwake_ms=%d\n", atoi(setting("backend.idle_close_ms", "1000")),
                   atoi(setting("backend.wake_ms", "50")));
        }
        nb_add(&c, "use.single=%s\nuse.burst=%s\nuse.bulk=%s\nuse.bulk_vs_threads=%s\nmin_glyphs=%d\nbattery=%s\n",
               outd.single ? "gpu" : "cpu", outd.burst ? "gpu" : "cpu", outd.bulk ? "gpu" : "cpu",
               outd.bulk_vs_threads ? "gpu" : "cpu", outd.min_glyphs, setting("battery", "cpu"));
        nb_add(&c, "reason=%s%s%s\n", outd.reason, notes.len ? " | " : "", notes.s);
    }
    nb_add(&c, "threads=%d\nmargin=%.2f\ndesktop_margin=%.2f\n", threads, in.margin, in.desktop_margin);
    {
        char now[128];
        nb_first_line("/proc/loadavg", now, sizeof now);
        nb_add(&c, "measure.load=%s\nmeasure.load_after=%s\n", load_before, now);
    }
    nb_add(&c, "# microseconds per glyph from call to answer. cpu1 is one thread. single, burst, batch, block and bulk are\n"
               "# calls of 1, 16, 64, 256 glyphs and of all the file's characters that are timed (measure.glyphs;\n"
               "# measure.not_timed are held by cells files, and are compared only). A GPU backend is timed in three\n"
               "# rounds, each with the CPU beside it on the same calls: times_cpu is the speed-up between the medians,\n"
               "# worst_round the least of the three, which is the one the choice is made on. measure.load is the\n"
               "# machine's load average before and after: a loaded machine makes the CPU look slower than it is.\n%s", m.s);
    rc = finish_10(&c, stamp, started);
    free(stamp);
    free(c.s);
    free(m.s);
    free(notes.s);
    return rc;
}

static int check_10(void)
{
    char stamp_path[4096], conf[4096], h[65];
    char *want = stamp_10();
    struct nb_kv have;
    int ok;
    snprintf(stamp_path, sizeof stamp_path, "%s/10-backend.stamp", state_dir);
    conf_path_here(conf, sizeof conf);
    ok = stamp_matches(stamp_path, want);
    free(want);
    if (ok) {
        nb_sha256_file(conf, h);
        nb_kv_read(stamp_path, &have);
        ok = h[0] && !strcmp(nb_kv_get(&have, "output.backend_conf", ""), h);
        nb_kv_free(&have);
    }
    return ok ? 0 : 1;
}

/* ---- step 40 ---- */

struct warm_arg {
    const char *lib, *backend;
    char **names;
    int n;
};

static void warm_entry(int fd, void *arg)
{
    struct warm_arg *a = arg;
    FILE *out = fdopen(fd, "w");
    struct gpu_lib g;
    neuronte_gpu *keep = NULL;
    char why[512] = "", path[4096], id[64];
    int i;
    setvbuf(out, NULL, _IOLBF, 0);
    if (gpu_load(a->lib, &g, why, sizeof why)) {
        fprintf(out, "error=%s\n", why);
        exit(0);
    }
    fprintf(out, "opened=0\n"); /* any file may compile: one limit for the whole run */
    for (i = 0; i < a->n; i++) {
        struct neuronte_gpu_options opt = {0};
        neuronte_gpu *h = NULL;
        double t0 = nb_now();
        weight_path(a->names[i], path, sizeof path);
        nb_file_id(path, id, sizeof id);
        opt.nrt_path = path;
        opt.cell_w = 8;
        opt.cell_h = 16;
        opt.backend = backend_number(a->backend);
        opt.flags = NEURONTE_GPU_FILE;
        opt.max_batch = 16;
        why[0] = 0;
        if (g.open(&opt, &h, why, sizeof why) || !h) {
            clean(why);
            /* a file this backend does not take is drawn by the CPU; it is settled all the same */
            fprintf(out, "refused.%s=%s %s\n", a->names[i], id, why[0] ? why : "open failed");
            continue;
        }
        fprintf(out, "warm.%s=%s\nseconds.%s=%.3f\n", a->names[i], id, a->names[i], nb_now() - t0);
        /* keep one handle so that the device stays open for the files that follow */
        if (!keep)
            keep = h;
        else
            g.close(h);
    }
    if (keep)
        g.close(keep);
    fprintf(out, "done=1\n");
    fflush(out);
    exit(0);
}

/* backend.conf is the machine's: in a user's run it is read from the system's state. */
static void conf_path_40(char *out, size_t len)
{
    if (user_mode)
        nb_backend_conf_path(out, len);
    else
        conf_path_here(out, len);
}

static int run_40(void)
{
    struct nb_kv conf, kv;
    struct nb_str st = {0};
    char conf_path[4096], stamp_path[4096], when[64], **names;
    const char *backend, *lib;
    int n, i, timed_out = 0, rc = 0;
    double started = nb_now();

    now_text(when, sizeof when);
    conf_path_40(conf_path, sizeof conf_path);
    snprintf(stamp_path, sizeof stamp_path, "%s/40-backend-warm.stamp", state_dir);
    if (nb_mkdirs(state_dir)) {
        say("40-backend-warm: cannot make %s", state_dir);
        return 1;
    }
    if (nb_kv_read(conf_path, &conf)) {
        say("40-backend-warm: no %s: the step 10-backend has not run", conf_path);
        return 1;
    }
    if (!strncmp(nb_kv_get(&conf, "reason", ""), "pending", 7)) {
        say("40-backend-warm: the backend is still to be measured; nothing to warm yet");
        nb_kv_free(&conf);
        return boot_mode ? EX_DEFERRED : 1;
    }
    backend = nb_kv_get(&conf, "gpu.backend", NULL);
    lib = nb_kv_get(&conf, "gpu.library", NULL);
    nb_add(&st, "step=40-backend-warm\nversion=%s\nscope=%s\nmade=%s\n", NB_STEP_VERSION, user_mode ? "user" : "system", when);
    if (!backend || !lib || atoi(nb_kv_get(&conf, "min_glyphs", "0")) <= 0) {
        nb_add(&st, "gpu.backend=none\nnote=the CPU was chosen: nothing to warm\n");
    } else if (!strcmp(setting("backend", "auto"), "cpu")) {
        nb_add(&st, "gpu.backend=none\nnote=backend=cpu in the settings: nothing is warmed\n");
    } else {
        struct warm_arg a = {lib, backend, NULL, 0};
        char drv[1024], id[64];
        double limit = atof(setting("backend.warm_seconds", "300"));
        n = list_weights(&names);
        a.names = names;
        a.n = n;
        run_child(warm_entry, &a, limit, limit, nb_now() + limit, &kv, &timed_out);
        nb_driver_id(backend, drv, sizeof drv);
        nb_file_id(lib, id, sizeof id);
        nb_add(&st, "gpu.backend=%s\ngpu.library=%s\ngpu.library.id=%s\ngpu.driver=%s\n", backend, lib, id, drv);
        for (i = 0; i < kv.n; i++)
            if (!strncmp(kv.key[i], "warm.", 5) || !strncmp(kv.key[i], "refused.", 8) || !strncmp(kv.key[i], "seconds.", 8))
                nb_add(&st, "%s=%s\n", kv.key[i], kv.val[i]);
        if (nb_kv_get(&kv, "error", NULL)) {
            say("40-backend-warm: %s", nb_kv_get(&kv, "error", ""));
            nb_add(&st, "error=%s\n", nb_kv_get(&kv, "error", ""));
            rc = 1;
        } else if (timed_out || !nb_kv_get(&kv, "done", NULL)) {
            const char *what = timed_out ? "out of time" : "its process ended early";
            say("40-backend-warm: %s; what was compiled is kept, run it again for the rest", what);
            nb_add(&st, "error=%s\n", what);
            rc = 1;
        }
        for (i = 0; i < n; i++)
            free(names[i]);
        free(names);
        nb_kv_free(&kv);
    }
    nb_add(&st, "seconds=%.1f\n", nb_now() - started);
    /* the stamp is written even after a failure: it names the files that are warm, and the
     * check finds the rest missing */
    if (nb_write_atomic(stamp_path, st.s)) {
        say("40-backend-warm: cannot write %s", stamp_path);
        rc = 1;
    }
    free(st.s);
    nb_kv_free(&conf);
    return rc;
}

static int check_40(void)
{
    struct nb_kv conf;
    char conf_path[4096], stamp_path[4096], **names, path[4096];
    int n, i, ok, gpu;
    conf_path_40(conf_path, sizeof conf_path);
    snprintf(stamp_path, sizeof stamp_path, "%s/40-backend-warm.stamp", state_dir);
    if (nb_kv_read(conf_path, &conf))
        return 1;
    if (!strncmp(nb_kv_get(&conf, "reason", ""), "pending", 7)) {
        nb_kv_free(&conf);
        return 1;
    }
    gpu = nb_kv_get(&conf, "gpu.backend", NULL) && atoi(nb_kv_get(&conf, "min_glyphs", "0")) > 0 &&
          strcmp(setting("backend", "auto"), "cpu");
    if (!gpu) {
        struct nb_kv none = {0};
        ok = nb_warm_current(stamp_path, &none, NULL);
    } else {
        ok = nb_warm_current(stamp_path, &conf, NULL);
        n = list_weights(&names);
        for (i = 0; i < n; i++) {
            weight_path(names[i], path, sizeof path);
            if (ok)
                ok = nb_warm_current(stamp_path, &conf, path);
            free(names[i]);
        }
        free(names);
    }
    nb_kv_free(&conf);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *step = NULL, *verb = NULL, *v;
    char base[4096], why[512];
    int i, current;

    if (readlink("/proc/self/exe", self_path, sizeof self_path - 1) < 0)
        snprintf(self_path, sizeof self_path, "%s", argv[0]);
    snprintf(base, sizeof base, "%s", argv[0]);
    step = basename(base);
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--step") && i + 1 < argc)
            step = argv[++i];
        else
            verb = argv[i];
    }
    if (!verb || (strcmp(verb, "check") && strcmp(verb, "run")) || (strcmp(step, "10-backend") && strcmp(step, "40-backend-warm"))) {
        fprintf(stderr, "usage: 10-backend check|run\n       40-backend-warm check|run\n"
                        "       neuronte-backend-step --step 10-backend|40-backend-warm check|run\n");
        return 2;
    }
    v = nb_env("NEURONTE_SETUP_SCOPE");
    user_mode = v && !strcmp(v, "user");
    boot_mode = nb_env("NEURONTE_SETUP_BOOT") != NULL;
    force = nb_env("NEURONTE_FORCE") != NULL;
    nb_kv_read(nb_setup_conf_path(), &setup);
    if (user_mode) {
        char path[4096];
        const char *x = nb_env("XDG_CONFIG_HOME"), *h = nb_env("HOME");
        if (x)
            snprintf(path, sizeof path, "%s/neuronte/setup.conf", x);
        else
            snprintf(path, sizeof path, "%s/.config/neuronte/setup.conf", h ? h : "/nonexistent");
        nb_kv_read(path, &user_setup);
    }
    v = nb_env("NEURONTE_WEIGHTS");
    snprintf(weights_dir, sizeof weights_dir, "%s", v ? v : setting("weights", "/usr/share/neuronte/weights"));
    v = nb_env("NEURONTE_STATE");
    if (v)
        snprintf(state_dir, sizeof state_dir, "%s", v);
    else if (user_mode)
        nb_user_state(state_dir, sizeof state_dir);
    else
        nb_system_state(state_dir, sizeof state_dir);
    v = nb_env("NEURONTE_THREADS");
    threads = atoi(v ? v : setting("threads", "2"));
    if (threads < 1)
        threads = 1;

    if (!strcmp(step, "10-backend")) {
        if (user_mode)
            return 0; /* the choice is the machine's: the system's run makes it */
        current = check_10() == 0;
        if (!strcmp(verb, "check"))
            return current ? 0 : 1;
        if (current && !force)
            return 0;
        if (boot_mode && !cpu_without_measuring(why, sizeof why)) {
            /* a login screen is waiting: say "the CPU, for now", write no stamp, and leave
             * the measuring to the run after the start */
            if (write_pending()) {
                say("10-backend: cannot write backend.conf in %s", state_dir);
                return 1;
            }
            return EX_DEFERRED;
        }
        return run_10();
    }
    current = check_40() == 0;
    if (!strcmp(verb, "check"))
        return current ? 0 : 1;
    if (current && !force)
        return 0;
    if (boot_mode) {
        /* compiling may take minutes on a cold cache. Until the stamp is there programs are
         * told "warming" and draw on the CPU. Nothing to warm is settled at once. */
        struct nb_kv conf;
        char path[4096];
        int gpu;
        conf_path_40(path, sizeof path);
        if (nb_kv_read(path, &conf))
            return EX_DEFERRED;
        gpu = nb_kv_get(&conf, "gpu.backend", NULL) && atoi(nb_kv_get(&conf, "min_glyphs", "0")) > 0 &&
              strncmp(nb_kv_get(&conf, "reason", ""), "pending", 7) && strcmp(setting("backend", "auto"), "cpu");
        if (!strncmp(nb_kv_get(&conf, "reason", ""), "pending", 7))
            gpu = 1;
        nb_kv_free(&conf);
        if (gpu)
            return EX_DEFERRED;
    }
    return run_40();
}
