/* Shared by the loader (neuronte_backend.c) and the setup steps
 * (neuronte-backend-step.c): key=value files, file identities, SHA-256,
 * where things are, and what the machine is. Everything here reads files
 * only; nothing wakes a GPU. C11 and POSIX. */
#ifndef NEURONTE_CHOOSER_COMMON_H
#define NEURONTE_CHOOSER_COMMON_H

#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#define NB_CONF_VERSION 1
#define NB_STEP_VERSION "1"

/* ---- key=value files ---- */

struct nb_kv {
    char *text; /* the file, cut into strings */
    char **key, **val;
    int n;
};

static void nb_kv_free(struct nb_kv *kv)
{
    free(kv->text);
    free(kv->key);
    free(kv->val);
    memset(kv, 0, sizeof *kv);
}

/* Cut text (which kv then owns) into lines "key=value"; '#' starts a comment line; spaces
 * around key and value are dropped. */
static int nb_kv_parse(char *text, struct nb_kv *kv)
{
    char *p, *line;
    int cap = 0;
    memset(kv, 0, sizeof *kv);
    kv->text = text;
    for (line = kv->text; line && *line; line = p) {
        char *eq, *e;
        p = strchr(line, '\n');
        if (p)
            *p++ = 0;
        while (*line == ' ' || *line == '\t')
            line++;
        if (!*line || *line == '#' || !(eq = strchr(line, '=')))
            continue;
        *eq = 0;
        for (e = eq; e > line && (e[-1] == ' ' || e[-1] == '\t'); e--)
            e[-1] = 0;
        for (eq++; *eq == ' ' || *eq == '\t'; eq++)
            ;
        for (e = eq + strlen(eq); e > eq && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r'); e--)
            e[-1] = 0;
        if (kv->n == cap) {
            char **k, **v;
            cap = cap ? cap * 2 : 64;
            k = realloc(kv->key, (size_t)cap * sizeof *kv->key);
            v = realloc(kv->val, (size_t)cap * sizeof *kv->val);
            if (k)
                kv->key = k;
            if (v)
                kv->val = v;
            if (!k || !v)
                return -1;
        }
        kv->key[kv->n] = line;
        kv->val[kv->n++] = eq;
    }
    return 0;
}

/* Read a key=value file. Returns 0, or -1 when the file cannot be read (kv is then empty).
 * A file larger than 4 MiB is not one of ours. */
static int nb_kv_read(const char *path, struct nb_kv *kv)
{
    FILE *f = fopen(path, "r");
    long len;
    char *text;
    memset(kv, 0, sizeof *kv);
    if (!f)
        return -1;
    if (fseek(f, 0, SEEK_END) || (len = ftell(f)) < 0 || len > (1 << 22) || fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return -1;
    }
    text = malloc((size_t)len + 1);
    if (!text || fread(text, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(text);
        return -1;
    }
    fclose(f);
    text[len] = 0;
    if (nb_kv_parse(text, kv)) {
        nb_kv_free(kv);
        return -1;
    }
    return 0;
}

static const char *nb_kv_get(const struct nb_kv *kv, const char *key, const char *dflt)
{
    int i;
    for (i = kv->n - 1; i >= 0; i--) /* the last setting of a key counts */
        if (!strcmp(kv->key[i], key))
            return kv->val[i];
    return dflt;
}

/* ---- a growing string ---- */

struct nb_str {
    char *s;
    size_t len, cap;
};

static void nb_add(struct nb_str *b, const char *fmt, ...)
{
    va_list ap;
    int n;
    for (;;) {
        va_start(ap, fmt);
        n = vsnprintf(b->s ? b->s + b->len : NULL, b->s ? b->cap - b->len : 0, fmt, ap);
        va_end(ap);
        if (n < 0)
            return;
        if (b->s && (size_t)n < b->cap - b->len) {
            b->len += (size_t)n;
            return;
        }
        b->cap = (b->cap + (size_t)n + 1) * 2;
        b->s = realloc(b->s, b->cap);
        if (!b->s) {
            b->cap = b->len = 0;
            return;
        }
        b->s[b->len] = 0;
    }
}

/* Write a whole file so that a reader sees the old one or the new one: a temporary file in the
 * same directory, flushed, then renamed over it. */
static int nb_write_atomic(const char *path, const char *text)
{
    char tmp[4096];
    int fd, ok;
    size_t len = strlen(text);
    if (snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid()) >= (int)sizeof tmp)
        return -1;
    fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        return -1;
    ok = write(fd, text, len) == (ssize_t)len && fsync(fd) == 0;
    ok = close(fd) == 0 && ok;
    if (!ok || rename(tmp, path)) {
        unlink(tmp);
        return -1;
    }
    return 0;
}

static int nb_mkdirs(const char *dir)
{
    char buf[4096], *p;
    if (snprintf(buf, sizeof buf, "%s", dir) >= (int)sizeof buf)
        return -1;
    for (p = buf + 1; *p; p++)
        if (*p == '/') {
            *p = 0;
            if (mkdir(buf, 0755) && errno != EEXIST)
                return -1;
            *p = '/';
        }
    return mkdir(buf, 0755) && errno != EEXIST ? -1 : 0;
}

static const char *nb_base(const char *path)
{
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

/* ---- identities of files ---- */

/* "size:mtime in nanoseconds", or "" for a file that is not there. Cheap: for the loader. */
static void nb_file_id(const char *path, char *out, size_t len)
{
    struct stat st;
    if (!path || stat(path, &st))
        snprintf(out, len, "%s", "");
    else
        snprintf(out, len, "%lld:%lld%09ld", (long long)st.st_size, (long long)st.st_mtim.tv_sec, (long)st.st_mtim.tv_nsec);
}

/* SHA-256 (FIPS 180-4) of a file as 64 hex digits; "" when it cannot be read. For the stamps. */
static void nb_sha256_file(const char *path, char out[65])
{
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char block[64];
    uint64_t total = 0;
    size_t got;
    int done = 0, i;
    FILE *f = fopen(path, "rb");
    out[0] = 0;
    if (!f)
        return;
    while (!done) {
        got = fread(block, 1, 64, f);
        total += got;
        if (got < 64) {
            /* padding: 0x80, zeros, the length in bits; it may need a block of its own */
            if (ferror(f)) {
                fclose(f);
                return;
            }
            block[got++] = 0x80;
            if (got > 56) {
                memset(block + got, 0, 64 - got);
                done = -1; /* this block, then one more with the length */
            } else {
                memset(block + got, 0, 56 - got);
                for (i = 0; i < 8; i++)
                    block[56 + i] = (unsigned char)((total * 8) >> (56 - 8 * i));
                done = 1;
            }
        }
        for (;;) {
            uint32_t w[64], a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], g = h[5], hh = h[6], k = h[7], t1, t2;
            uint32_t *v = h;
            for (i = 0; i < 16; i++)
                w[i] = (uint32_t)block[4 * i] << 24 | (uint32_t)block[4 * i + 1] << 16 | (uint32_t)block[4 * i + 2] << 8 |
                       block[4 * i + 3];
            for (; i < 64; i++) {
                uint32_t s0 = (w[i - 15] >> 7 | w[i - 15] << 25) ^ (w[i - 15] >> 18 | w[i - 15] << 14) ^ (w[i - 15] >> 3);
                uint32_t s1 = (w[i - 2] >> 17 | w[i - 2] << 15) ^ (w[i - 2] >> 19 | w[i - 2] << 13) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }
            for (i = 0; i < 64; i++) {
                t1 = k + ((e >> 6 | e << 26) ^ (e >> 11 | e << 21) ^ (e >> 25 | e << 7)) + ((e & g) ^ (~e & hh)) + K[i] + w[i];
                t2 = ((a >> 2 | a << 30) ^ (a >> 13 | a << 19) ^ (a >> 22 | a << 10)) + ((a & b) ^ (a & c) ^ (b & c));
                k = hh; hh = g; g = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
            }
            v[0] += a; v[1] += b; v[2] += c; v[3] += d; v[4] += e; v[5] += g; v[6] += hh; v[7] += k;
            if (done != -1)
                break;
            memset(block, 0, 56);
            for (i = 0; i < 8; i++)
                block[56 + i] = (unsigned char)((total * 8) >> (56 - 8 * i));
            done = 1;
        }
    }
    fclose(f);
    for (i = 0; i < 8; i++)
        snprintf(out + 8 * i, 9, "%08x", h[i]);
}

/* ---- where things are ---- */

static const char *nb_env(const char *name)
{
    const char *v = getenv(name);
    return v && *v ? v : NULL;
}

static const char *nb_setup_conf_path(void)
{
    const char *v = nb_env("NEURONTE_SETUP_CONF");
    return v ? v : "/etc/neuronte/setup.conf";
}

/* The machine's state directory: where setup writes backend.conf. */
static void nb_system_state(char *out, size_t len)
{
    const char *v = nb_env("NEURONTE_SYSTEM_STATE");
    struct nb_kv kv;
    if (v) {
        snprintf(out, len, "%s", v);
        return;
    }
    nb_kv_read(nb_setup_conf_path(), &kv);
    snprintf(out, len, "%s", nb_kv_get(&kv, "state", "/var/lib/neuronte"));
    nb_kv_free(&kv);
}

/* A user's own state directory: where that user's warm stamp is. */
static void nb_user_state(char *out, size_t len)
{
    const char *x = nb_env("XDG_STATE_HOME"), *h = nb_env("HOME");
    if (x)
        snprintf(out, len, "%s/neuronte", x);
    else
        snprintf(out, len, "%s/.local/state/neuronte", h ? h : "/nonexistent");
}

static void nb_backend_conf_path(char *out, size_t len)
{
    const char *v = nb_env("NEURONTE_BACKEND_CONF");
    char state[2048];
    if (v) {
        snprintf(out, len, "%s", v);
        return;
    }
    nb_system_state(state, sizeof state);
    snprintf(out, len, "%s/backend.conf", state);
}

/* ---- what the machine is ---- */

static void nb_first_line(const char *path, char *out, size_t len)
{
    FILE *f = fopen(path, "r");
    out[0] = 0;
    if (f) {
        if (fgets(out, (int)len, f))
            out[strcspn(out, "\r\n")] = 0;
        fclose(f);
    }
}

/* The driver a backend ("cuda", "vulkan") runs on, as a string that changes when the driver
 * does: NVIDIA's kernel module version, and for other drivers the size and time of the Mesa
 * driver files found. Read from files; no GPU is touched. NEURONTE_BACKEND_TEST_DRIVER
 * replaces it (tests). */
static void nb_driver_id(const char *backend, char *out, size_t len)
{
    static const char *const mesa[] = {
        "/usr/lib/x86_64-linux-gnu/libvulkan_intel.so",   "/usr/lib/x86_64-linux-gnu/libvulkan_radeon.so",
        "/usr/lib/x86_64-linux-gnu/libvulkan_nouveau.so", "/usr/lib/aarch64-linux-gnu/libvulkan_panfrost.so",
        "/usr/lib/aarch64-linux-gnu/libvulkan_freedreno.so", "/usr/lib/aarch64-linux-gnu/libvulkan_broadcom.so",
        "/usr/lib64/libvulkan_intel.so", "/usr/lib64/libvulkan_radeon.so", "/usr/lib/libvulkan_intel.so",
        "/usr/lib/libvulkan_radeon.so", NULL};
    const char *t = nb_env("NEURONTE_BACKEND_TEST_DRIVER");
    char nv[256], id[64];
    size_t at = 0;
    int i;
    out[0] = 0;
    if (t) {
        snprintf(out, len, "%s", t);
        return;
    }
    nb_first_line("/sys/module/nvidia/version", nv, sizeof nv);
    if (!strcmp(backend, "cuda") || (nv[0] && strcmp(backend, "vulkan"))) {
        snprintf(out, len, "nvidia %s", nv[0] ? nv : "absent");
        return;
    }
    if (nv[0])
        at = (size_t)snprintf(out, len, "nvidia %s;", nv);
    for (i = 0; mesa[i] && at < len; i++) {
        nb_file_id(mesa[i], id, sizeof id);
        if (id[0])
            at += (size_t)snprintf(out + at, len - at, "%s %s;", strrchr(mesa[i], '/') + 1, id);
    }
    if (!out[0])
        snprintf(out, len, "%s", "unknown");
}

/* Display devices on the PCI bus as "vendor:device" pairs, sorted by address. */
static void nb_gpu_list(char *out, size_t len)
{
    const char *root = nb_env("NEURONTE_BACKEND_TEST_PCI");
    struct dirent **names = NULL;
    size_t at = 0;
    int n, i;
    out[0] = 0;
    if (!root)
        root = "/sys/bus/pci/devices";
    n = scandir(root, &names, NULL, alphasort);
    for (i = 0; i < n; i++) {
        char path[1024], cls[32], ven[32], dev[32];
        snprintf(path, sizeof path, "%s/%s/class", root, names[i]->d_name);
        nb_first_line(path, cls, sizeof cls);
        if (!strncmp(cls, "0x03", 4)) {
            snprintf(path, sizeof path, "%s/%s/vendor", root, names[i]->d_name);
            nb_first_line(path, ven, sizeof ven);
            snprintf(path, sizeof path, "%s/%s/device", root, names[i]->d_name);
            nb_first_line(path, dev, sizeof dev);
            if (at < len)
                at += (size_t)snprintf(out + at, len - at, "%s%s:%s", at ? " " : "", ven + (ven[1] == 'x' ? 2 : 0),
                                       dev + (dev[1] == 'x' ? 2 : 0));
        }
        free(names[i]);
    }
    free(names);
    if (!out[0])
        snprintf(out, len, "%s", "none");
}

static void nb_kernel_release(char *out, size_t len)
{
    struct utsname u;
    snprintf(out, len, "%s", uname(&u) ? "unknown" : u.release);
}

/* Whether the machine runs from its battery, and how full that is. A mouse's or a headset's
 * battery (scope "Device") does not count. Returns 1 when discharging; *percent is the lowest
 * capacity among discharging batteries, or -1. NEURONTE_BACKEND_POWER_DIR replaces the
 * directory read (tests). */
static int nb_discharging(int *percent)
{
    const char *root = nb_env("NEURONTE_BACKEND_POWER_DIR");
    struct dirent *e;
    DIR *d;
    int yes = 0;
    *percent = -1;
    if (!root)
        root = "/sys/class/power_supply";
    d = opendir(root);
    if (!d)
        return 0;
    while ((e = readdir(d))) {
        char path[1024], v[64];
        int cap;
        if (e->d_name[0] == '.')
            continue;
        snprintf(path, sizeof path, "%s/%s/type", root, e->d_name);
        nb_first_line(path, v, sizeof v);
        if (strcmp(v, "Battery"))
            continue;
        snprintf(path, sizeof path, "%s/%s/scope", root, e->d_name);
        nb_first_line(path, v, sizeof v);
        if (!strcmp(v, "Device"))
            continue;
        snprintf(path, sizeof path, "%s/%s/status", root, e->d_name);
        nb_first_line(path, v, sizeof v);
        if (strcmp(v, "Discharging"))
            continue;
        snprintf(path, sizeof path, "%s/%s/capacity", root, e->d_name);
        nb_first_line(path, v, sizeof v);
        cap = v[0] ? atoi(v) : 100;
        if (!yes || cap < *percent)
            *percent = cap;
        yes = 1;
    }
    closedir(d);
    return yes;
}

static double nb_now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

#endif
