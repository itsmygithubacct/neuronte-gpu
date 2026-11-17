/* neuronte_colour_cost: what the CPU library pays to draw each character of a file, colour files included.
 *
 *   neuronte_colour_cost FILE.nrt SCALARS WxH[,WxH...]
 *
 * For each cell size and each listed character, on one thread:
 *   layers   neuronte_coverage_layers (every layer's mask and its palette entry; never cached)
 *   first    neuronte_draw_scalar into a pixel buffer, the first time (what a program pays on first use)
 *   again    the same call a second time (the library's draw cache)
 * It prints the median, mean and largest time per character and the total for the list. No GPU is used.
 */
#define _GNU_SOURCE
#include "neuronte.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static int cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static void stats(const char *name, double *v, int n)
{
    double sum = 0;
    int i;
    for (i = 0; i < n; i++)
        sum += v[i];
    qsort(v, (size_t)n, sizeof *v, cmp);
    printf(",\"%s\":{\"median_us\":%.1f,\"mean_us\":%.1f,\"max_us\":%.1f,\"total_ms\":%.2f}", name, v[n / 2], sum / n,
           v[n - 1], sum / 1e3);
}

int main(int argc, char **argv)
{
    uint32_t list[4096];
    char line[128], *sizes, *tok;
    int n = 0, i;
    FILE *f;
    if (argc != 4 || !(f = fopen(argv[2], "r")))
        return 2;
    while (n < 4096 && fgets(line, sizeof line, f))
        if (line[0] != '\n' && line[0] != '#')
            list[n++] = (line[0] == 'U' && line[1] == '+') ? (uint32_t)strtoul(line + 2, NULL, 16)
                                                           : (uint32_t)strtoul(line, NULL, 0);
    fclose(f);
    if (!n)
        return 2;
    sizes = strdup(argv[3]);
    for (tok = strtok(sizes, ","); tok; tok = strtok(NULL, ",")) {
        int w, h, k;
        neuronte *lib;
        float *masks;
        uint8_t entries[64];
        uint32_t *px;
        double *tl = malloc((size_t)n * sizeof *tl), *t1 = malloc((size_t)n * sizeof *t1),
               *t2 = malloc((size_t)n * sizeof *t2), t;
        if (sscanf(tok, "%dx%d", &w, &h) != 2 || neuronte_open_file(argv[1], &lib) != NEURONTE_OK)
            return 2;
        k = neuronte_layers(lib);
        masks = malloc((size_t)k * w * h * sizeof *masks);
        px = calloc((size_t)w * h, sizeof *px);
        for (i = 0; i < n; i++) {
            t = now_us();
            if (neuronte_coverage_layers(lib, list[i], w, h, 0, masks, entries) != NEURONTE_OK)
                return 1;
            tl[i] = now_us() - t;
        }
        for (i = 0; i < n; i++) {
            t = now_us();
            neuronte_draw_scalar(lib, px, w, w, h, 0, 0, w, h, list[i], 0xffffffffu);
            t1[i] = now_us() - t;
            t = now_us();
            neuronte_draw_scalar(lib, px, w, w, h, 0, 0, w, h, list[i], 0xffffffffu);
            t2[i] = now_us() - t;
        }
        printf("{\"tool\":\"neuronte_colour_cost\",\"file\":\"%s\",\"cell\":[%d,%d],\"characters\":%d,\"layer_count\":%d,\"nets\":%d",
               strrchr(argv[1], '/') ? strrchr(argv[1], '/') + 1 : argv[1], w, h, n, k, neuronte_nets(lib));
        stats("layers", tl, n);
        stats("first_draw", t1, n);
        stats("draw_again", t2, n);
        printf("}\n");
        neuronte_close(lib);
        free(masks); free(px); free(tl); free(t1); free(t2);
    }
    return 0;
}
