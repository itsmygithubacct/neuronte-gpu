/* The chooser's rules on made-up and on measured numbers. */
#include "../common.h"
#include "../decide.h"

static int fails, checks;
#define OK(cond) \
    do { \
        checks++; \
        if (!(cond)) { \
            fprintf(stderr, "    FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            fails++; \
        } \
    } while (0)

static void one(struct nb_decide_in *in, int k, int whole, const double cpu[5], double cpun)
{
    in->whole[k] = whole;
    memcpy(in->cpu1[k], cpu, sizeof in->cpu1[k]);
    in->cpun[k] = cpun;
    if (in->nworks <= k)
        in->nworks = k + 1;
}

static struct nb_decide_backend *gpu(struct nb_decide_in *in, const char *name, int exact, int desktop, int k, const double t[5])
{
    struct nb_decide_backend *b = &in->backend[in->nbackends];
    int i;
    for (i = 0; i < in->nbackends; i++)
        if (!strcmp(in->backend[i].name, name))
            b = &in->backend[i];
    if (b == &in->backend[in->nbackends])
        in->nbackends++;
    snprintf(b->name, sizeof b->name, "%s", name);
    b->exact = exact;
    b->draws_desktop = desktop;
    memcpy(b->gpu[k], t, sizeof b->gpu[k]);
    return b;
}

int main(void)
{
    struct nb_decide_in in;
    struct nb_decide_out out;

    /* a discrete GPU that wins every call size (the Quadro RTX 3000's figures of 17 November
     * against a loaded core): used from one glyph */
    memset(&in, 0, sizeof in);
    in.margin = 1.3;
    in.desktop_margin = 2;
    one(&in, 0, 1341, (double[]){392, 400, 410, 420, 617.7}, 80);
    gpu(&in, "cuda", 1, 0, 0, (double[]){144, 77, 74, 63, 34.9});
    nb_decide(&in, &out);
    OK(out.chosen == 0 && out.min_glyphs == 1 && out.single && out.burst && out.bulk && out.bulk_vs_threads);
    OK(strstr(out.reason, "cuda") && strstr(out.reason, "17.7"));

    /* a slower discrete GPU (the Quadro M2000M's figures): wins whole tables and blocks of 256,
     * loses at 64, so it is used from 256 although it also won at 16; loses to eight threads */
    memset(&in, 0, sizeof in);
    in.margin = 1.3;
    in.desktop_margin = 2;
    one(&in, 0, 1341, (double[]){354, 354, 354, 354, 354}, 88);
    gpu(&in, "cuda", 1, 0, 0, (double[]){293, 228, 300, 258, 226});
    nb_decide(&in, &out);
    OK(out.chosen == 0 && out.min_glyphs == 256 && !out.single && !out.burst && out.bulk && !out.bulk_vs_threads);

    /* timed in rounds, a GPU near the line (the Quadro M2000M's figures of 17 November): by the
     * medians it wins from 16 glyphs, but one round in three was under the margin at 16, 64 and
     * 256, so those sizes stay with the CPU and only whole tables go to it. The choice is made on
     * the worst round, whatever cpu1 says. */
    memset(&in, 0, sizeof in);
    in.margin = 1.3;
    in.desktop_margin = 2;
    one(&in, 0, 1152, (double[]){208, 335, 416, 397, 410}, 185);
    {
        struct nb_decide_backend *b = gpu(&in, "cuda", 1, 0, 0, (double[]){297, 241, 252, 257, 222});
        memcpy(b->cpu[0], (double[]){208, 335, 416, 397, 410}, sizeof b->cpu[0]);
        memcpy(b->worst[0], (double[]){0.70, 1.39, 1.65, 1.55, 1.85}, sizeof b->worst[0]);
        nb_decide(&in, &out);
        OK(out.chosen == 0 && out.min_glyphs == 16 && !out.single && out.burst);
        memcpy(b->worst[0], (double[]){0.70, 1.22, 1.28, 1.31, 1.62}, sizeof b->worst[0]);
        nb_decide(&in, &out);
        OK(out.chosen == 0 && out.min_glyphs == 256 && !out.burst && out.bulk);
        OK(strstr(out.reason, "at least 1.6 times") && strstr(out.reason, "in every round"));
        memcpy(b->worst[0], (double[]){0.70, 1.22, 1.28, 1.29, 1.62}, sizeof b->worst[0]);
        nb_decide(&in, &out);
        OK(out.chosen == 0 && out.min_glyphs == 1152 && !out.burst && out.bulk);
        memcpy(b->worst[0], (double[]){0.70, 1.22, 1.28, 1.29, 1.29}, sizeof b->worst[0]);
        nb_decide(&in, &out);
        OK(out.chosen == -1 && out.min_glyphs == 0);
    }

    /* a backend that wins every size by twice the margin in every round makes timing the next
     * one pointless; one size short of that, a size not timed in rounds, a GPU that draws the
     * desktop, or one differing byte, and the next one is timed */
    memset(&in, 0, sizeof in);
    in.margin = 1.3;
    in.desktop_margin = 2;
    OK(nb_clear_winner(&in) == NULL);
    one(&in, 0, 1152, (double[]){269, 370, 418, 442, 569}, 263);
    {
        struct nb_decide_backend *b = gpu(&in, "cuda", 1, 0, 0, (double[]){102, 42, 37, 33, 32});
        OK(nb_clear_winner(&in) == NULL); /* not timed in rounds */
        memcpy(b->worst[0], (double[]){2.64, 7.2, 7.9, 8.9, 10.3}, sizeof b->worst[0]);
        OK(nb_clear_winner(&in) && !strcmp(nb_clear_winner(&in), "cuda"));
        b->worst[0][0] = 2.55;
        OK(nb_clear_winner(&in) == NULL);
        b->worst[0][0] = 2.64;
        b->draws_desktop = 1;
        OK(nb_clear_winner(&in) == NULL);
        b->draws_desktop = 0;
        b->exact = 0;
        OK(nb_clear_winner(&in) == NULL);
        b->exact = 1;
        in.cpu1[0][3] = -1; /* a size the file is too small for is not asked of it */
        b->worst[0][3] = 0;
        OK(nb_clear_winner(&in) != NULL);
    }

    /* under the margin by a hair: the CPU */
    memset(&in, 0, sizeof in);
    in.margin = 1.3;
    in.desktop_margin = 2;
    one(&in, 0, 1000, (double[]){100, 100, 100, 100, 129}, 30);
    gpu(&in, "vulkan", 1, 0, 0, (double[]){500, 300, 200, 150, 100});
    nb_decide(&in, &out);
    OK(out.chosen == -1 && out.min_glyphs == 0 && !out.bulk && strstr(out.reason, "the CPU is fastest"));
    in.cpu1[0][4] = 130;
    nb_decide(&in, &out);
    OK(out.chosen == 0 && out.min_glyphs == 1000 && out.bulk && !out.burst);

    /* a backend that differed from the CPU by one byte is never used, whatever its speed */
    memset(&in, 0, sizeof in);
    in.margin = 1.3;
    in.desktop_margin = 2;
    one(&in, 0, 1000, (double[]){400, 400, 400, 400, 400}, 100);
    gpu(&in, "cuda", 0, 0, 0, (double[]){10, 5, 4, 3, 2});
    nb_decide(&in, &out);
    OK(out.chosen == -1 && strstr(out.reason, "differs from the CPU"));
    /* and an exact, slower one beside it is */
    gpu(&in, "vulkan", 1, 0, 0, (double[]){900, 300, 200, 150, 100});
    nb_decide(&in, &out);
    OK(out.chosen == 1 && out.min_glyphs == 16 && !out.single && out.burst);

    /* a GPU that also draws the desktop: whole tables only, and only at twice one thread */
    memset(&in, 0, sizeof in);
    in.margin = 1.3;
    in.desktop_margin = 2;
    one(&in, 0, 1341, (double[]){300, 300, 300, 300, 300}, 80);
    gpu(&in, "vulkan", 1, 1, 0, (double[]){100, 100, 100, 100, 160});
    nb_decide(&in, &out);
    OK(out.chosen == -1);
    in.backend[0].gpu[0][4] = 145;
    nb_decide(&in, &out);
    OK(out.chosen == 0 && out.min_glyphs == 1341 && !out.single && !out.burst && out.bulk);

    /* two works: the GPU must win on both, and the smaller whole table sets the threshold */
    memset(&in, 0, sizeof in);
    in.margin = 1.3;
    in.desktop_margin = 2;
    one(&in, 0, 1341, (double[]){300, 300, 300, 300, 300}, 80);
    one(&in, 1, 600, (double[]){900, 900, 900, 900, 900}, 240);
    gpu(&in, "cuda", 1, 0, 0, (double[]){400, 400, 400, 400, 100});
    gpu(&in, "cuda", 1, 0, 1, (double[]){800, 800, 800, 800, 800});
    nb_decide(&in, &out);
    OK(out.chosen == -1);
    in.backend[0].gpu[1][4] = 300;
    nb_decide(&in, &out);
    OK(out.chosen == 0 && out.min_glyphs == 600);

    /* of two that win, the one used from the smaller call; at a tie, the faster on a table */
    memset(&in, 0, sizeof in);
    in.margin = 1.3;
    in.desktop_margin = 2;
    one(&in, 0, 1000, (double[]){400, 400, 400, 400, 400}, 100);
    gpu(&in, "cuda", 1, 0, 0, (double[]){500, 100, 100, 100, 20});
    gpu(&in, "vulkan", 1, 0, 0, (double[]){200, 150, 150, 150, 50});
    nb_decide(&in, &out);
    OK(out.chosen == 1 && out.min_glyphs == 1);
    in.backend[1].gpu[0][0] = 500;
    nb_decide(&in, &out);
    OK(out.chosen == 0 && out.min_glyphs == 16);

    /* nothing answered */
    memset(&in, 0, sizeof in);
    in.margin = 1.3;
    in.desktop_margin = 2;
    one(&in, 0, 1000, (double[]){400, 400, 400, 400, 400}, 100);
    nb_decide(&in, &out);
    OK(out.chosen == -1 && strstr(out.reason, "no GPU backend answered"));

    printf("test_decide: %d checks, %d failed\n", checks, fails);
    return fails != 0;
}
