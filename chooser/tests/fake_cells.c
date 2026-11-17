/* A stand-in for the cells file of setup step 30-cells, for the loader's
 * test: it "holds" the scalars U+0041 to U+005A at the size given in
 * FAKE_CELLS_SIZE (WxH), drawn by the reference library when the stand-in
 * is opened from the weights file it is opened for. Without FAKE_CELLS_SIZE
 * there is no cells file. */
#include "neuronte.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct neuronte_cells {
    int cw, ch;
    float *planes;
} neuronte_cells;

int fake_cells_finds;

int neuronte_cells_open(const char *path, const char *nrt, unsigned flags, neuronte_cells **out)
{
    const char *size = getenv("FAKE_CELLS_SIZE");
    neuronte_cells *c;
    neuronte *n;
    int i;
    (void)path;
    (void)flags;
    *out = NULL;
    if (!size || !nrt)
        return -1;
    c = calloc(1, sizeof *c);
    if (sscanf(size, "%dx%d", &c->cw, &c->ch) != 2 || neuronte_open_file(nrt, &n) != NEURONTE_OK) {
        free(c);
        return -1;
    }
    c->planes = malloc(26 * (size_t)c->cw * c->ch * sizeof *c->planes);
    for (i = 0; i < 26; i++)
        neuronte_coverage(n, 0x41 + (uint32_t)i, c->cw, c->ch, 0, c->planes + (size_t)i * c->cw * c->ch);
    neuronte_close(n);
    *out = c;
    return 0;
}

const float *neuronte_cells_find(const neuronte_cells *c, uint32_t scalar, int cell_w, int cell_h, int samples,
                                 unsigned style, int snap_below)
{
    if (samples != 0 || style != 0 || snap_below != -1)
        return NULL;
    if (cell_w != c->cw || cell_h != c->ch || scalar < 0x41 || scalar > 0x5A)
        return NULL;
    fake_cells_finds++;
    return c->planes + (size_t)(scalar - 0x41) * c->cw * c->ch;
}

void neuronte_cells_close(neuronte_cells *c)
{
    free(c->planes);
    free(c);
}
