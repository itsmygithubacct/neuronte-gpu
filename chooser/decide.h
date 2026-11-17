/* The chooser's rules, apart from any measuring, so that they can be tested
 * with made-up numbers: which backend is fastest (nb_decide), whether a
 * backend.conf still describes this machine (nb_conf_stale), and whether the
 * kernels of a file are warm for this user (nb_warm_current). Included after
 * common.h. */
#ifndef NEURONTE_CHOOSER_DECIDE_H
#define NEURONTE_CHOOSER_DECIDE_H

#define NB_SIZES 5 /* glyphs per call: 1, 16, 64, 256, a whole table */
#define NB_BACKENDS 4
#define NB_WORKS 16 /* a file at a cell size */

static const int nb_size_glyphs[NB_SIZES] = {1, 16, 64, 256, 0};

struct nb_decide_backend {
    char name[32];
    int exact;         /* every compared call was byte for byte the CPU's */
    int draws_desktop; /* the GPU also draws the screen */
    double gpu[NB_WORKS][NB_SIZES]; /* microseconds per glyph; 0 or less: not measured */
    /* When the backend was timed in rounds, each with the CPU beside it: the CPU's microseconds
     * per glyph there, and the least speed-up over it among the rounds. 0: not timed so, and
     * the speed-up is cpu1 over gpu. */
    double cpu[NB_WORKS][NB_SIZES];
    double worst[NB_WORKS][NB_SIZES];
};

struct nb_decide_in {
    int nworks, nbackends;
    int whole[NB_WORKS];            /* glyphs in the work's whole table */
    double cpu1[NB_WORKS][NB_SIZES]; /* one CPU thread */
    double cpun[NB_WORKS];          /* the whole table on the threads setup may use */
    struct nb_decide_backend backend[NB_BACKENDS];
    double margin, desktop_margin;
};

struct nb_decide_out {
    int chosen; /* index of the backend, or -1: the CPU */
    int single, burst, bulk;
    int bulk_vs_threads; /* the GPU also beats the threaded CPU on a whole table */
    int min_glyphs;      /* calls of this many glyphs or more go to the GPU; 0: none */
    char reason[512];
};

/* The least speed-up over one CPU thread among the works measured at a size, and among the
 * rounds of each where it was timed in rounds; 0 when none was. */
static double nb_least_ratio(const struct nb_decide_in *in, const struct nb_decide_backend *b, int s, int *at)
{
    double least = 0;
    int k;
    for (k = 0; k < in->nworks; k++)
        if (b->gpu[k][s] > 0 && (b->worst[k][s] > 0 || in->cpu1[k][s] > 0)) {
            double r = b->worst[k][s] > 0 ? b->worst[k][s] : in->cpu1[k][s] / b->gpu[k][s];
            if (least == 0 || r < least) {
                least = r;
                if (at)
                    *at = k;
            }
        }
    return least;
}

/* A backend is used from the smallest call size at which, and at every larger size, it is
 * `margin` times faster than one CPU thread on every work measured, in every round it was
 * timed in: a size that is in doubt stays with the CPU. A GPU that also draws the
 * desktop is used for whole tables only, and only at `desktop_margin`. A backend that differed
 * from the CPU by one byte is never used. Among backends, the one used from the smallest call;
 * then the one faster on a whole table. */
static void nb_decide(const struct nb_decide_in *in, struct nb_decide_out *out)
{
    int b, s, k, best_min = 0, min_whole = 0;
    double best_bulk = 0;
    char note[400] = "";
    size_t at = 0;
    memset(out, 0, sizeof *out);
    out->chosen = -1;
    for (k = 0; k < in->nworks; k++)
        if (!min_whole || in->whole[k] < min_whole)
            min_whole = in->whole[k];
    for (b = 0; b < in->nbackends; b++) {
        const struct nb_decide_backend *g = &in->backend[b];
        int from = NB_SIZES, mg;
        double bulk = nb_least_ratio(in, g, NB_SIZES - 1, NULL);
        if (!g->exact) {
            if (at < sizeof note)
                at += (size_t)snprintf(note + at, sizeof note - at, "%s differs from the CPU; ", g->name);
            continue;
        }
        for (s = NB_SIZES - 1; s >= 0; s--) {
            double need = !g->draws_desktop ? in->margin : s == NB_SIZES - 1 ? in->desktop_margin : 0;
            double r = nb_least_ratio(in, g, s, NULL);
            if (need <= 0 || r < need)
                break;
            from = s;
        }
        if (from == NB_SIZES) {
            if (at < sizeof note)
                at += (size_t)snprintf(note + at, sizeof note - at, "%s is %.2f times one CPU thread on a whole table, under the %.2f asked%s; ",
                                       g->name, bulk, g->draws_desktop ? in->desktop_margin : in->margin,
                                       g->draws_desktop ? " of a GPU that draws the desktop" : "");
            continue;
        }
        mg = from < NB_SIZES - 1 ? nb_size_glyphs[from] : min_whole > 257 ? min_whole : 257;
        if (out->chosen < 0 || mg < best_min || (mg == best_min && bulk > best_bulk)) {
            out->chosen = b;
            best_min = mg;
            best_bulk = bulk;
        }
    }
    if (at >= 2 && at < sizeof note)
        note[at - 2] = 0; /* the last "; " */
    if (out->chosen < 0) {
        snprintf(out->reason, sizeof out->reason, "the CPU is fastest here: %s",
                 in->nbackends ? note : "no GPU backend answered");
        return;
    }
    {
        const struct nb_decide_backend *g = &in->backend[out->chosen];
        int w = 0, w1 = 0;
        double r1;
        best_bulk = nb_least_ratio(in, g, NB_SIZES - 1, &w);
        r1 = nb_least_ratio(in, g, 0, &w1);
        out->min_glyphs = best_min;
        out->single = best_min <= 1;
        out->burst = best_min <= 16;
        out->bulk = 1;
        out->bulk_vs_threads = 1;
        for (k = 0; k < in->nworks; k++)
            if (g->gpu[k][NB_SIZES - 1] > 0 && in->cpun[k] > 0 && in->cpun[k] / g->gpu[k][NB_SIZES - 1] < in->margin)
                out->bulk_vs_threads = 0;
        at = (size_t)snprintf(out->reason, sizeof out->reason,
                              "%s is at least %.1f times one CPU thread on a whole table%s (%.1f against %.1f us/glyph)", g->name,
                              best_bulk, g->worst[w][NB_SIZES - 1] > 0 ? " in every round" : "", g->gpu[w][NB_SIZES - 1],
                              g->cpu[w][NB_SIZES - 1] > 0 ? g->cpu[w][NB_SIZES - 1] : in->cpu1[w][NB_SIZES - 1]);
        if (r1 > 0 && at < sizeof out->reason)
            at += (size_t)snprintf(out->reason + at, sizeof out->reason - at, "; one glyph %.0f against %.0f", g->gpu[w1][0],
                                   g->cpu[w1][0] > 0 ? g->cpu[w1][0] : in->cpu1[w1][0]);
        if (at < sizeof out->reason && best_min <= 1)
            at += (size_t)snprintf(out->reason + at, sizeof out->reason - at, "; used for every call");
        else if (at < sizeof out->reason)
            at += (size_t)snprintf(out->reason + at, sizeof out->reason - at, "; used for calls of %d glyphs or more%s", best_min,
                                   g->draws_desktop ? " (this GPU also draws the desktop)" : "");
        if (note[0] && at < sizeof out->reason)
            snprintf(out->reason + at, sizeof out->reason - at, " | %s", note);
    }
}

/* Whether a backend.conf no longer describes this machine. Files are read; no GPU is touched.
 * Returns 0 when it does, else 1 with the reason in why. A conf that chose the CPU is never stale
 * here: setup's own stamp decides when to measure again. */
static int nb_conf_stale(const struct nb_kv *conf, char *why, size_t len)
{
    const char *backend = nb_kv_get(conf, "gpu.backend", NULL), *lib = nb_kv_get(conf, "gpu.library", NULL);
    char id[64], drv[1024];
    why[0] = 0;
    if (atoi(nb_kv_get(conf, "version", "0")) != NB_CONF_VERSION) {
        snprintf(why, len, "backend.conf is of another version");
        return 1;
    }
    if (!backend || atoi(nb_kv_get(conf, "min_glyphs", "0")) <= 0)
        return 0;
    if (!lib || strcmp(nb_kv_get(conf, "gpu.identical", ""), "yes")) {
        snprintf(why, len, "backend.conf names a GPU without a library that passed the byte check");
        return 1;
    }
    nb_file_id(lib, id, sizeof id);
    if (!id[0] || strcmp(id, nb_kv_get(conf, "gpu.library.id", ""))) {
        snprintf(why, len, "the GPU library changed since it was measured");
        return 1;
    }
    nb_driver_id(backend, drv, sizeof drv);
    if (strcmp(drv, nb_kv_get(conf, "gpu.driver", ""))) {
        snprintf(why, len, "the GPU driver changed since it was measured");
        return 1;
    }
    return 0;
}

/* Whether the warm stamp at stamp_path covers the backend conf chose, with the present library
 * and driver, and (when file is not NULL) that weights file as it is now. 1 when it does. */
static int nb_warm_current(const char *stamp_path, const struct nb_kv *conf, const char *file)
{
    const char *backend = nb_kv_get(conf, "gpu.backend", NULL), *lib = nb_kv_get(conf, "gpu.library", NULL);
    struct nb_kv st;
    char id[64], drv[1024], key[512];
    int ok = 0, gpu = backend && lib && atoi(nb_kv_get(conf, "min_glyphs", "0")) > 0;
    if (nb_kv_read(stamp_path, &st))
        return 0;
    if (strcmp(nb_kv_get(&st, "version", ""), NB_STEP_VERSION))
        goto out;
    if (!gpu) {
        ok = !strcmp(nb_kv_get(&st, "gpu.backend", ""), "none");
        goto out;
    }
    if (strcmp(nb_kv_get(&st, "gpu.backend", ""), backend) || strcmp(nb_kv_get(&st, "gpu.library", ""), lib))
        goto out;
    nb_file_id(lib, id, sizeof id);
    nb_driver_id(backend, drv, sizeof drv);
    if (!id[0] || strcmp(nb_kv_get(&st, "gpu.library.id", ""), id) || strcmp(nb_kv_get(&st, "gpu.driver", ""), drv))
        goto out;
    ok = 1;
    if (file) {
        const char *v;
        size_t n;
        nb_file_id(file, id, sizeof id);
        n = strlen(id);
        snprintf(key, sizeof key, "warm.%s", nb_base(file));
        v = nb_kv_get(&st, key, NULL);
        if (!v) {
            snprintf(key, sizeof key, "refused.%s", nb_base(file));
            v = nb_kv_get(&st, key, NULL);
        }
        ok = v && n && !strncmp(v, id, n) && (v[n] == 0 || v[n] == ' ');
    }
out:
    nb_kv_free(&st);
    return ok;
}

#endif
