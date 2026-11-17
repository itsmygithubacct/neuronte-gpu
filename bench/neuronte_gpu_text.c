/* neuronte_gpu_text: draw a line of text with neuronte_gpu_draw_text() and
 * with the reference library's neuronte_draw_text(), compare the pixels and
 * time both.
 *
 *   neuronte_gpu_text --nrt FACE.nrt --em PX [--backend auto|cuda|vulkan|cpu] [--device NAME]
 *                     [--text STRING] [--bold] [--stroke PX] [--repeats N]
 *
 * Prints one JSON object: whether the two buffers are identical, the time of
 * the first line (every letter new) and of a repeated line (every letter
 * kept), for both.
 */
#define _GNU_SOURCE
#include "neuronte.h"
#include "neuronte_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    struct neuronte_gpu_options opt = {0};
    const char *text = "The quick brown fox jumps over the lazy dog. 0123456789 (Neuronte) [v2] {GPU}; 100% @ #1!";
    const char *backend = "auto";
    int em = 14, bold = 0, stroke = -1, repeats = 20, i;
    char reason[512] = "";
    neuronte_gpu *g = NULL;
    neuronte *n = NULL;

    for (i = 1; i < argc; i++) {
#define ARG() (i + 1 < argc ? argv[++i] : (exit(2), (char *)0))
        if (!strcmp(argv[i], "--nrt")) opt.nrt_path = ARG();
        else if (!strcmp(argv[i], "--em")) em = atoi(ARG());
        else if (!strcmp(argv[i], "--backend")) backend = ARG();
        else if (!strcmp(argv[i], "--device")) opt.device = ARG();
        else if (!strcmp(argv[i], "--text")) text = ARG();
        else if (!strcmp(argv[i], "--bold")) bold = 1;
        else if (!strcmp(argv[i], "--stroke")) stroke = atoi(ARG());
        else if (!strcmp(argv[i], "--repeats")) repeats = atoi(ARG());
        else return 2;
#undef ARG
    }
    if (!opt.nrt_path || em < 1 || repeats < 1 || repeats > 1000)
        return 2;
    opt.cell_w = em;
    opt.cell_h = 2 * em;
    opt.backend = !strcmp(backend, "cuda") ? NEURONTE_GPU_CUDA : !strcmp(backend, "vulkan") ? NEURONTE_GPU_VULKAN
                : !strcmp(backend, "cpu") ? NEURONTE_GPU_CPU : NEURONTE_GPU_AUTO;
    opt.flags = NEURONTE_GPU_FILE;
    if (neuronte_open_file(opt.nrt_path, &n) != NEURONTE_OK) {
        printf("{\"error\":\"the library cannot open the file\"}\n");
        return 1;
    }
    uint64_t t0 = now();
    if (neuronte_gpu_open(&opt, &g, reason, sizeof reason)) {
        printf("{\"error\":\"open failed: %s\"}\n", reason);
        return 1;
    }
    uint64_t open_ns = now() - t0;
    size_t len = strlen(text);
    int width = neuronte_text_width(n, text, len, bold, em) + 4 * em, height = 2 * em + 8;
    if (width < 1) {
        printf("{\"error\":\"the file has no advances\"}\n");
        return 1;
    }
    size_t px = (size_t)width * height;
    uint32_t *a = malloc(px * sizeof *a), *b = malloc(px * sizeof *b);
    uint64_t *ta = malloc((size_t)repeats * sizeof *ta), *tb = malloc((size_t)repeats * sizeof *tb);
    struct neuronte_gpu_timing tm, first_tm;
    int ra = 0, rb = 0, same = 1;
    if (!a || !b || !ta || !tb)
        return 1;
    /* a background that is not uniform, so blending is compared too */
    for (size_t k = 0; k < px; k++)
        a[k] = b[k] = 0xFF000000u | (uint32_t)(k * 2654435761u >> 8);
    t0 = now();
    ra = stroke < 0 ? neuronte_draw_text(n, a, width, width, height, 2, 4, em, bold, text, len, 0xE0F0C040u)
                    : neuronte_draw_text_stroke(n, a, width, width, height, 2, 4, em, stroke, text, len, 0xE0F0C040u);
    uint64_t cpu_first = now() - t0;
    t0 = now();
    rb = neuronte_gpu_draw_text(g, b, width, width, height, 2, 4, bold, stroke, text, len, 0xE0F0C040u, &first_tm);
    uint64_t gpu_first = now() - t0;
    same = ra == rb && memcmp(a, b, px * sizeof *a) == 0;
    for (i = 0; i < repeats; i++) {
        t0 = now();
        ra = stroke < 0 ? neuronte_draw_text(n, a, width, width, height, 2, 4, em, bold, text, len, 0xE0F0C040u)
                        : neuronte_draw_text_stroke(n, a, width, width, height, 2, 4, em, stroke, text, len,
                                                    0xE0F0C040u);
        ta[i] = now() - t0;
        t0 = now();
        rb = neuronte_gpu_draw_text(g, b, width, width, height, 2, 4, bold, stroke, text, len, 0xE0F0C040u, &tm);
        tb[i] = now() - t0;
        same = same && ra == rb && memcmp(a, b, px * sizeof *a) == 0;
    }
    qsort(ta, (size_t)repeats, sizeof *ta, cmp_u64);
    qsort(tb, (size_t)repeats, sizeof *tb, cmp_u64);
    printf("{\"tool\":\"neuronte_gpu_text\",\"backend\":\"%s\",\"device\":\"%s\",\"em\":%d,\"characters\":%zu,"
           "\"pen\":%d,\"identical\":%s,\"open_ns\":%llu,"
           "\"first_line\":{\"library_ns\":%llu,\"gpu_ns\":%llu,\"gpu_kernel_ns\":%llu,\"gpu_plan_ns\":%llu,"
           "\"launches\":%d,\"library_glyphs\":%d,\"warped\":%d},"
           "\"kept_line\":{\"library_ns\":%llu,\"gpu_ns\":%llu}}\n",
           neuronte_gpu_backend_name(neuronte_gpu_backend(g)), neuronte_gpu_device(g), em, len, rb,
           same ? "true" : "false", (unsigned long long)open_ns, (unsigned long long)cpu_first,
           (unsigned long long)gpu_first, (unsigned long long)first_tm.run, (unsigned long long)first_tm.plan,
           first_tm.launches, first_tm.cpu_glyphs, first_tm.warped, (unsigned long long)ta[repeats / 2],
           (unsigned long long)tb[repeats / 2]);
    neuronte_gpu_close(g);
    neuronte_close(n);
    return same ? 0 : 1;
}
