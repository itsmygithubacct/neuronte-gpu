/* neuronte-backend: say what the loader would do on this machine, and try it.
 *
 *   neuronte-backend                 the choice, and why
 *   neuronte-backend FILE.nrt WxH    open the file through the loader, draw its characters in
 *                                    calls of 1, 16 and all of them, compare every glyph with
 *                                    the reference library and time both
 *   neuronte-backend FILE.nrt WxH SECONDS   the same, then keep the handle open and idle for
 *                                    that long (to watch what the GPU does after a call)
 *
 * Exit 0, or 1 when a glyph differed from the reference library, or 2 for a mistake in the
 * command or a file that cannot be opened. */
#include "common.h"

#include "neuronte.h"
#include "neuronte_backend.h"

static void show(const struct neuronte_backend_info *i)
{
    printf("gpu: %s\nbackend: %s\ndevice: %s\nmin_glyphs: %d\nreason: %s\n", i->gpu ? "yes" : "no", i->backend, i->device,
           i->min_glyphs, i->reason);
}

int main(int argc, char **argv)
{
    struct neuronte_backend_info info;
    neuronte_backend *b = NULL;
    neuronte *n = NULL;
    uint32_t *scalars, s;
    float *got, *want;
    size_t plane;
    double t0, cpu_s, first_s, again_s, small_s = 0;
    int cw, ch, count = 0, i, differing = 0, small = 0;

    if (argc == 1) {
        neuronte_backend_choice(&info);
        show(&info);
        return 0;
    }
    if (argc < 3 || argc > 4 || sscanf(argv[2], "%dx%d", &cw, &ch) != 2) {
        fprintf(stderr, "usage: neuronte-backend [FILE.nrt WxH [SECONDS]]\n");
        return 2;
    }
    if (neuronte_backend_open(argv[1], cw, ch, &b) != 0 || neuronte_open_file(argv[1], &n) != NEURONTE_OK) {
        fprintf(stderr, "neuronte-backend: cannot open %s at %dx%d\n", argv[1], cw, ch);
        return 2;
    }
    plane = (size_t)cw * ch;
    scalars = malloc(4096 * sizeof *scalars);
    for (s = 0x20; s <= 0x10FFFF && count < 4096; s++) {
        if (s == 0xD800)
            s = 0xE000;
        if (neuronte_untrained(n, s) == 0)
            scalars[count++] = s;
        if (count == 256 && neuronte_untrained(n, 0x10FFFE) == 0 && neuronte_untrained(n, 0xFFFE) == 0)
            break; /* a file without a table of its characters: 256 will do */
    }
    got = malloc((size_t)count * plane * sizeof *got);
    want = malloc((size_t)count * plane * sizeof *want);
    if (!got || !want || count == 0)
        return 2;
    t0 = nb_now();
    for (i = 0; i < count; i++)
        neuronte_coverage(n, scalars[i], cw, ch, 0, want + (size_t)i * plane);
    cpu_s = nb_now() - t0;

    /* all of them: the first call may find the GPU still opening, the second not */
    t0 = nb_now();
    if (neuronte_backend_coverage(b, scalars, count, got) != 0)
        return 2;
    first_s = nb_now() - t0;
    differing += memcmp(got, want, (size_t)count * plane * sizeof *got) != 0;
    {
        struct timespec nap = {1, 0};
        neuronte_backend_info(b, &info);
        if (info.gpu && info.from_gpu == 0)
            nanosleep(&nap, NULL);
    }
    memset(got, 0, (size_t)count * plane * sizeof *got);
    t0 = nb_now();
    if (neuronte_backend_coverage(b, scalars, count, got) != 0)
        return 2;
    again_s = nb_now() - t0;
    differing += memcmp(got, want, (size_t)count * plane * sizeof *got) != 0;
    /* calls of one glyph and of 16 */
    t0 = nb_now();
    for (i = 0; i < count && i < 64; i++, small++) {
        neuronte_backend_coverage(b, scalars + i, 1, got);
        differing += memcmp(got, want + (size_t)i * plane, plane * sizeof *got) != 0;
    }
    for (i = 0; i + 16 <= count && i < 256; i += 16, small += 16) {
        neuronte_backend_coverage(b, scalars + i, 16, got);
        differing += memcmp(got, want + (size_t)i * plane, 16 * plane * sizeof *got) != 0;
    }
    small_s = nb_now() - t0;
    neuronte_backend_info(b, &info);
    show(&info);
    printf("file: %s\ncell: %dx%d\nglyphs: %d\n", argv[1], cw, ch, count);
    printf("reference library, one thread: %.1f us/glyph\n", cpu_s * 1e6 / count);
    printf("loader, all in one call, first: %.1f us/glyph (%.3f s)\n", first_s * 1e6 / count, first_s);
    printf("loader, all in one call, again: %.1f us/glyph (%.3f s)\n", again_s * 1e6 / count, again_s);
    printf("loader, calls of 1 and of 16: %.1f us/glyph\n", small ? small_s * 1e6 / small : 0.0);
    printf("answered from: cells file %llu, cpu %llu, gpu %llu\n", (unsigned long long)info.from_cells,
           (unsigned long long)info.from_cpu, (unsigned long long)info.from_gpu);
    printf("calls differing from the reference library: %d\n", differing);
    if (argc == 4) {
        struct timespec nap = {atoi(argv[3]), 0};
        fflush(stdout);
        nanosleep(&nap, NULL);
    }
    neuronte_backend_close(b);
    neuronte_close(n);
    return differing ? 1 : 0;
}
