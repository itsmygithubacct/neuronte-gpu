/* neuronte_gpu_soak: many calls of changing size, every one compared with the CPU library.
 *
 *   neuronte_gpu_soak --nrt FILE.nrt --cell WxH --scalars LIST [--backend cuda|vulkan|opengl]
 *                     [--device NAME] [--calls N] [--seed S] [--max-batch N]
 *
 * tools/exact.py compares one call. This draws the list's characters in
 * calls of 1, 2, 3, ... glyphs, in random subsets and whole, N calls in all,
 * and compares each call's floats with the library's, bit for bit. It
 * prints the number of calls and glyphs that differed, the largest
 * difference, and the first differing call. Exit status 1 if any differed.
 */
#define _GNU_SOURCE
#include "neuronte.h"
#include "neuronte_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static uint32_t rng;
static uint32_t next(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

int main(int argc, char **argv)
{
    struct neuronte_gpu_options opt = {0};
    const char *backend = "cuda", *list = NULL;
    uint32_t *all = NULL, *pick;
    neuronte *lib;
    neuronte_gpu *g;
    char why[512] = "";
    int n, calls = 200, i, k, bad_calls = 0, bad_glyphs = 0, first_bad = -1, first_bad_count = 0, glyphs = 0;
    size_t plane;
    float *ref, *got;
    double worst = 0;

    rng = 1;
    opt.max_batch = 2048;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--nrt") && i + 1 < argc) opt.nrt_path = argv[++i];
        else if (!strcmp(argv[i], "--scalars") && i + 1 < argc) list = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) opt.device = argv[++i];
        else if (!strcmp(argv[i], "--calls") && i + 1 < argc) calls = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) rng = (uint32_t)strtoul(argv[++i], NULL, 0) | 1;
        else if (!strcmp(argv[i], "--max-batch") && i + 1 < argc) opt.max_batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cell") && i + 1 < argc && sscanf(argv[i + 1], "%dx%d", &opt.cell_w, &opt.cell_h) == 2) i++;
        else return 2;
    }
    if (!opt.nrt_path || !list || opt.cell_w < 1 || calls < 1 || (n = read_scalars(list, &all)) < 1)
        return 2;
    if (n > opt.max_batch)
        n = opt.max_batch;
    opt.backend = !strcmp(backend, "vulkan") ? NEURONTE_GPU_VULKAN : !strcmp(backend, "opengl") ? NEURONTE_GPU_OPENGL
                : !strcmp(backend, "cpu") ? NEURONTE_GPU_CPU : NEURONTE_GPU_CUDA;
    opt.flags = NEURONTE_GPU_FILE;
    if (neuronte_open_file(opt.nrt_path, &lib) != NEURONTE_OK || neuronte_gpu_open(&opt, &g, why, sizeof why)) {
        printf("{\"error\":\"open failed: %s\"}\n", why);
        return 1;
    }
    plane = (size_t)opt.cell_w * opt.cell_h;
    ref = malloc((size_t)n * plane * sizeof *ref);
    got = malloc((size_t)n * plane * sizeof *got);
    pick = malloc((size_t)n * sizeof *pick);
    for (k = 0; k < calls; k++) {
        /* sizes 1, 2, 3, ... for the first calls, then random sizes, every eighth call the whole list */
        int count = k < 8 ? k + 1 : k % 8 == 7 ? n : 1 + (int)(next() % (uint32_t)n), differ = 0;
        if (count > n)
            count = n;
        for (i = 0; i < count; i++)
            pick[i] = count == n ? all[i] : all[next() % (uint32_t)n];
        for (i = 0; i < count; i++)
            if (neuronte_coverage(lib, pick[i], opt.cell_w, opt.cell_h, 0, ref + (size_t)i * plane) != NEURONTE_OK)
                return 1;
        memset(got, 0xff, (size_t)count * plane * sizeof *got);
        if (neuronte_gpu_coverage(g, pick, count, got, NULL)) {
            printf("{\"error\":\"call %d of %d glyphs failed\"}\n", k, count);
            return 1;
        }
        for (i = 0; i < count; i++)
            if (memcmp(ref + (size_t)i * plane, got + (size_t)i * plane, plane * sizeof *ref)) {
                differ++;
                for (size_t p = 0; p < plane; p++) {
                    double d = fabs((double)ref[(size_t)i * plane + p] - (double)got[(size_t)i * plane + p]);
                    if (!(d <= worst)) /* also catches a NaN */
                        worst = d;
                }
            }
        glyphs += count;
        if (differ) {
            bad_calls++;
            bad_glyphs += differ;
            if (first_bad < 0) {
                first_bad = k;
                first_bad_count = count;
            }
        }
    }
    printf("{\"tool\":\"neuronte_gpu_soak\",\"backend\":\"%s\",\"device\":\"%s\",\"cell\":[%d,%d],\"calls\":%d,"
           "\"glyphs\":%d,\"calls_differing\":%d,\"glyphs_differing\":%d,\"largest_difference\":%g,"
           "\"first_differing_call\":%d,\"its_glyphs\":%d}\n",
           neuronte_gpu_backend_name(neuronte_gpu_backend(g)), neuronte_gpu_device(g), opt.cell_w, opt.cell_h, calls,
           glyphs, bad_calls, bad_glyphs, worst, first_bad, first_bad_count);
    neuronte_gpu_close(g);
    neuronte_close(lib);
    return bad_calls != 0;
}
