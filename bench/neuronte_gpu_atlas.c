/* neuronte_gpu_atlas: ways to get Neuronte glyphs into a terminal's glyph atlas, compared.
 *
 *   neuronte_gpu_atlas --nrt FILE.nrt --cell WxH --scalars LIST [--device NAME]
 *                      [--paths cpu,device,gl,gl-direct] [--repeats N] [--ask MAJOR.MINOR]
 *                      [--device-backend auto|cuda|vulkan] [--device-name NAME]
 *
 * The program stands in for a terminal that draws with OpenGL: it makes a
 * context the way such a terminal does (OpenGL 3.3 core asked for, or the
 * version given with --ask; whatever the driver gives is taken), on the
 * device named, and a glyph atlas like the
 * terminal's: an array texture of GL_SRGB8_ALPHA8 in which a glyph is a
 * sprite of W by H+1 texels, white with coverage in alpha. It then fills
 * the atlas with the listed characters by each path:
 *
 *   cpu        the CPU library draws each glyph; glTexSubImage3D uploads it
 *   device     neuronte-gpu on its own device context (CUDA, else Vulkan)
 *              draws the batch; the pixels come back and are uploaded
 *   gl         neuronte-gpu draws the batch in THIS OpenGL context
 *              (a compute shader, needs 4.3); pixels come back, are uploaded
 *   gl-direct  the same, but the pixels never leave the device: a second
 *              compute shader writes them to a scratch texture, which is
 *              copied into the atlas
 *
 * For each path: the time to open, the time from nothing to the first
 * glyph in the atlas, the time for the whole list, the time of a frame that
 * needs the whole list new (glyphs, then the screen drawn), and whether the
 * atlas ends up byte for byte what the cpu path made. A frame is a
 * 1920x1080 grid of cells drawn from the atlas into a framebuffer.
 * One JSON object per path on stdout.
 */
#define _GNU_SOURCE
#include "neuronte.h"
#include "neuronte_gpu.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/glcorearb.h>
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

static uint64_t median(uint64_t *v, int n)
{
    qsort(v, (size_t)n, sizeof *v, cmp_u64);
    return v[n / 2];
}

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

/* ---- the stand-in terminal ---- */

static int cw, ch, xnum, ynum, count, grid_cols, grid_rows;
static uint32_t *scalars;
static GLuint atlas, scratch, frame_fbo, frame_tex, frame_prog, blit_prog, vao;
static int gl_major, gl_minor, ask_major = 3, ask_minor = 3;

static EGLDisplay dpy;
static EGLContext ctx;

static int make_context(const char *want, char *why, size_t len)
{
    /* what the terminal asks for: 3.3 core, forward compatible */
    const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, ask_major, EGL_CONTEXT_MINOR_VERSION, ask_minor,
                                      EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                                      EGL_CONTEXT_OPENGL_FORWARD_COMPATIBLE, EGL_TRUE, EGL_NONE};
    static const EGLint cfg_attr[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    EGLDeviceEXT devs[16];
    EGLint ndev = 0, i;
    PFNEGLQUERYDEVICESEXTPROC query = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    char seen[512] = "";
    if (!query || !query(16, devs, &ndev))
        ndev = 0;
    for (i = 0; i < ndev; i++) {
        EGLint major, minor, n = 0;
        EGLConfig cfg = NULL;
        const char *r, *v;
        size_t used;
        dpy = eglGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devs[i], NULL);
        if (!dpy || !eglInitialize(dpy, &major, &minor))
            continue;
        eglBindAPI(EGL_OPENGL_API);
        ctx = eglCreateContext(dpy, NULL, EGL_NO_CONTEXT, ctx_attr);
        if (!ctx && eglChooseConfig(dpy, cfg_attr, &cfg, 1, &n) && n > 0)
            ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
        if (!ctx || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
            eglTerminate(dpy);
            continue;
        }
        r = (const char *)glGetString(GL_RENDERER);
        v = (const char *)glGetString(GL_VENDOR);
        used = strlen(seen);
        snprintf(seen + used, sizeof seen - used, "%s%s, %s", used ? "; " : "", v ? v : "?", r ? r : "?");
        if (r && v && (!want || strcasestr(r, want) || strcasestr(v, want)))
            return 0;
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(dpy, ctx);
        eglTerminate(dpy);
    }
    snprintf(why, len, "no OpenGL device matching '%s' (saw: %s)", want ? want : "", seen);
    return -1;
}

static GLuint compile(GLenum kind, const char *src)
{
    GLuint sh = glCreateShader(kind);
    GLint ok = 0;
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(sh, sizeof log, NULL, log);
        fprintf(stderr, "shader: %s\n", log);
        exit(1);
    }
    return sh;
}

static GLuint link2(GLuint a, GLuint b)
{
    GLuint p = glCreateProgram();
    GLint ok = 0;
    glAttachShader(p, a);
    if (b)
        glAttachShader(p, b);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        fprintf(stderr, "link: %s\n", log);
        exit(1);
    }
    return p;
}

static GLuint new_atlas(GLenum format)
{
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D_ARRAY, t);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, format, xnum * cw, ynum * (ch + 1), 1);
    return t;
}

static void setup_frame(void)
{
    static const char *vs = "#version 330 core\nvoid main() {\n"
                            "    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
                            "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n}\n";
    static const char *fs =
        "#version 330 core\nuniform sampler2DArray sprites;\nuniform ivec4 lay; // cell w, cell h, xnum, count\n"
        "uniform int cols;\nout vec4 color;\nvoid main() {\n"
        "    ivec2 px = ivec2(gl_FragCoord.xy);\n    ivec2 cell = px / lay.xy, in_cell = px - cell * lay.xy;\n"
        "    int s = (cell.x + cell.y * cols) % lay.w;\n"
        "    ivec2 at = ivec2(s % lay.z, s / lay.z) * ivec2(lay.x, lay.y + 1) + in_cell;\n"
        "    float a = texelFetch(sprites, ivec3(at, 0), 0).a;\n    color = vec4(vec3(a), 1.0);\n}\n";
    frame_prog = link2(compile(GL_VERTEX_SHADER, vs), compile(GL_FRAGMENT_SHADER, fs));
    glGenVertexArrays(1, &vao);
    glGenTextures(1, &frame_tex);
    glBindTexture(GL_TEXTURE_2D, frame_tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1920, 1080);
    glGenFramebuffers(1, &frame_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, frame_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frame_tex, 0);
    glViewport(0, 0, 1920, 1080);
}

/* One frame: every cell of the screen drawn from the atlas. */
static void draw_frame(void)
{
    glUseProgram(frame_prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, atlas);
    glUniform1i(glGetUniformLocation(frame_prog, "sprites"), 0);
    glUniform4i(glGetUniformLocation(frame_prog, "lay"), cw, ch, xnum, count);
    glUniform1i(glGetUniformLocation(frame_prog, "cols"), grid_cols);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* What the terminal does with a glyph's coverage: white, alpha as a byte, one upload per sprite. */
static void upload(int first, int n, const float *cov, uint32_t *px)
{
    size_t plane = (size_t)cw * ch;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, atlas);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    for (int i = 0; i < n; i++) {
        const float *c = cov + (size_t)i * plane;
        int s = first + i;
        for (size_t k = 0; k < plane; k++) {
            int a = (int)((double)c[k] * 255.0 + 0.5);
            px[k] = 0xffffff00u | (uint32_t)(a < 0 ? 0 : a > 255 ? 255 : a);
        }
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, (s % xnum) * cw, (s / xnum) * (ch + 1), 0, cw, ch, 1, GL_RGBA,
                        GL_UNSIGNED_INT_8_8_8_8, px);
    }
}

/* The device-only route: coverage in a shader storage buffer to the scratch texture, then into the atlas. */
static void setup_blit(void)
{
    static const char *cs =
        "#version 430\nlayout(local_size_x = 8, local_size_y = 8) in;\n"
        "layout(std430, binding = 0) readonly buffer Cov { float cov[]; };\n"
        "layout(rgba8, binding = 0) uniform writeonly image2DArray dst;\n"
        "uniform ivec4 lay; // cell w, cell h, xnum, glyphs\nuniform int first;\nvoid main() {\n"
        "    ivec2 at = ivec2(gl_GlobalInvocationID.xy);\n    ivec2 sp = at / ivec2(lay.x, lay.y + 1);\n"
        "    ivec2 in_sp = at - sp * ivec2(lay.x, lay.y + 1);\n    int s = sp.x + sp.y * lay.z - first;\n"
        "    if (sp.x >= lay.z || s < 0 || s >= lay.w || in_sp.y >= lay.y) return;\n"
        "    float v = cov[s * lay.x * lay.y + in_sp.y * lay.x + in_sp.x];\n"
        "    precise double r = double(v) * 255.0lf + 0.5lf;\n    int a = clamp(int(r), 0, 255);\n"
        "    imageStore(dst, ivec3(at, 0), vec4(1.0, 1.0, 1.0, float(a) / 255.0));\n}\n";
    blit_prog = link2(compile(GL_COMPUTE_SHADER, cs), 0);
    scratch = new_atlas(GL_RGBA8);
}

static void blit(GLuint cov, int first, int n)
{
    int y0 = first / xnum, y1 = (first + n - 1) / xnum + 1;
    glUseProgram(blit_prog);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cov);
    glBindImageTexture(0, scratch, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glUniform4i(glGetUniformLocation(blit_prog, "lay"), cw, ch, xnum, n);
    glUniform1i(glGetUniformLocation(blit_prog, "first"), first);
    /* the rows of sprites that hold these glyphs */
    glDispatchCompute((GLuint)(xnum * cw + 7) / 8, (GLuint)(y1 * (ch + 1) + 7) / 8, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
    glCopyImageSubData(scratch, GL_TEXTURE_2D_ARRAY, 0, 0, y0 * (ch + 1), 0, atlas, GL_TEXTURE_2D_ARRAY, 0, 0,
                       y0 * (ch + 1), 0, xnum * cw, (y1 - y0) * (ch + 1), 1);
}

enum { P_CPU, P_DEVICE, P_GL, P_GL_DIRECT, P_COUNT };
static const char *path_name[P_COUNT] = {"cpu", "device", "gl", "gl-direct"};

struct path {
    int kind;
    neuronte *lib;
    neuronte_gpu *gpu;
    float *cov;
    uint32_t *px;
};

/* Glyphs first .. first+n-1 of the list into the atlas by this path; 0 on success. */
static int fill(struct path *p, int first, int n)
{
    if (p->kind == P_CPU) {
        for (int i = 0; i < n; i++) {
            if (neuronte_coverage(p->lib, scalars[first + i], cw, ch, 0, p->cov) != NEURONTE_OK)
                return -1;
            upload(first + i, 1, p->cov, p->px);
        }
    } else if (p->kind == P_GL_DIRECT) {
        if (neuronte_gpu_coverage_device(p->gpu, scalars + first, n, NULL))
            return -1;
        blit(neuronte_gpu_gl_output(p->gpu), first, n);
    } else {
        if (neuronte_gpu_coverage(p->gpu, scalars + first, n, p->cov, NULL))
            return -1;
        upload(first, n, p->cov, p->px);
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *nrt = NULL, *list = NULL, *device = NULL, *paths = "cpu,device,gl,gl-direct";
    const char *dev_backend = "auto", *dev_name = NULL;
    char why[512] = "";
    int repeats = 5, i, k;
    unsigned char *reference = NULL, *got;
    size_t atlas_bytes;
    uint64_t t0, idle[64];

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--nrt") && i + 1 < argc) nrt = argv[++i];
        else if (!strcmp(argv[i], "--scalars") && i + 1 < argc) list = argv[++i];
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = argv[++i];
        else if (!strcmp(argv[i], "--paths") && i + 1 < argc) paths = argv[++i];
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cell") && i + 1 < argc && sscanf(argv[i + 1], "%dx%d", &cw, &ch) == 2) i++;
        else if (!strcmp(argv[i], "--ask") && i + 1 < argc && sscanf(argv[i + 1], "%d.%d", &ask_major, &ask_minor) == 2) i++;
        else if (!strcmp(argv[i], "--device-backend") && i + 1 < argc) dev_backend = argv[++i];
        else if (!strcmp(argv[i], "--device-name") && i + 1 < argc) dev_name = argv[++i];
        else return 2;
    }
    if (!nrt || !list || cw < 1 || ch < 1 || repeats < 1 || repeats > 64 || (count = read_scalars(list, &scalars)) < 1)
        return 2;
    if (make_context(device, why, sizeof why)) {
        printf("{\"error\":\"%s\"}\n", why);
        return 1;
    }
    glGetIntegerv(GL_MAJOR_VERSION, &gl_major);
    glGetIntegerv(GL_MINOR_VERSION, &gl_minor);
    grid_cols = 1920 / cw;
    grid_rows = 1080 / ch;
    xnum = 2048 / cw;
    ynum = (count + xnum - 1) / xnum;
    atlas_bytes = (size_t)xnum * cw * ynum * (ch + 1) * 4;
    got = malloc(atlas_bytes);
    setup_frame();
    if (gl_major > 4 || (gl_major == 4 && gl_minor >= 3))
        setup_blit();
    atlas = new_atlas(GL_SRGB8_ALPHA8);
    for (k = 0; k < 20; k++) { /* an idle frame's time: nothing new to draw */
        t0 = now();
        draw_frame();
        glFinish();
        idle[k] = now() - t0;
    }
    printf("{\"context\":{\"asked\":\"%d.%d core\",\"got\":\"%d.%d\",\"version\":\"%s\",\"renderer\":\"%s\","
           "\"compute_shaders\":%s},\"cell\":[%d,%d],\"glyphs\":%d,\"atlas\":[%d,%d],\"atlas_bytes\":%zu,"
           "\"screen_cells\":%d,\"idle_frame_ns\":%llu}\n",
           ask_major, ask_minor, gl_major, gl_minor, (const char *)glGetString(GL_VERSION),
           (const char *)glGetString(GL_RENDERER),
           blit_prog ? "true" : "false", cw, ch, count, xnum * cw, ynum * (ch + 1), atlas_bytes,
           grid_cols * grid_rows, (unsigned long long)median(idle, 20));

    for (int kind = 0; kind < P_COUNT; kind++) {
        struct path p = {0};
        struct neuronte_gpu_options opt = {0};
        struct neuronte_gpu_memory mem = {0};
        uint64_t open_ns, first_ns, all[64], frame[64];
        int same = -1, worst = 0;
        size_t differing = 0, first_differing = 0;
        {
            /* the path is run when its name is one of the comma-separated words */
            const char *q = paths;
            size_t ln = strlen(path_name[kind]);
            int listed = 0;
            while (q && *q) {
                const char *e = strchr(q, ',');
                size_t n = e ? (size_t)(e - q) : strlen(q);
                if (n == ln && !strncmp(q, path_name[kind], n))
                    listed = 1;
                q = e ? e + 1 : NULL;
            }
            if (!listed || ((kind == P_GL || kind == P_GL_DIRECT) && !blit_prog))
                continue;
        }
        p.kind = kind;
        p.cov = malloc((size_t)count * cw * ch * sizeof *p.cov);
        p.px = malloc((size_t)cw * ch * sizeof *p.px);
        glDeleteTextures(1, &atlas);
        atlas = new_atlas(GL_SRGB8_ALPHA8);
        glFinish();
        t0 = now();
        if (kind == P_CPU) {
            if (neuronte_open_file(nrt, &p.lib) != NEURONTE_OK) {
                printf("{\"path\":\"cpu\",\"error\":\"the library cannot open the file\"}\n");
                return 1;
            }
        } else {
            opt.nrt_path = nrt;
            opt.cell_w = cw;
            opt.cell_h = ch;
            opt.max_batch = count;
            opt.flags = NEURONTE_GPU_FILE | (kind == P_DEVICE ? 0 : NEURONTE_GPU_GL_CURRENT);
            opt.backend = kind != P_DEVICE ? NEURONTE_GPU_OPENGL
                          : !strcmp(dev_backend, "cuda") ? NEURONTE_GPU_CUDA
                          : !strcmp(dev_backend, "vulkan") ? NEURONTE_GPU_VULKAN : NEURONTE_GPU_AUTO;
            if (kind == P_DEVICE)
                opt.device = dev_name;
            if (neuronte_gpu_open(&opt, &p.gpu, why, sizeof why)) {
                for (char *q = why; *q; q++)
                    if (*q == '"' || *q == '\\' || *q == '\n')
                        *q = ' ';
                printf("{\"path\":\"%s\",\"error\":\"%s\"}\n", path_name[kind], why);
                continue;
            }
            neuronte_gpu_memory(p.gpu, &mem);
        }
        open_ns = now() - t0;
        /* nothing to the first glyph in the atlas */
        t0 = now();
        if (fill(&p, 0, 1))
            return 1;
        glFinish();
        first_ns = now() - t0;
        for (k = 0; k < repeats; k++) {
            /* the whole list new */
            t0 = now();
            if (fill(&p, 0, count))
                return 1;
            glFinish();
            all[k] = now() - t0;
        }
        for (k = 0; k < repeats; k++) {
            /* a frame that needs the whole list new: the glyphs, then the screen */
            t0 = now();
            if (fill(&p, 0, count))
                return 1;
            draw_frame();
            glFinish();
            frame[k] = now() - t0;
        }
        glBindTexture(GL_TEXTURE_2D_ARRAY, atlas);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, GL_UNSIGNED_BYTE, got);
        if (kind == P_CPU) {
            reference = malloc(atlas_bytes);
            memcpy(reference, got, atlas_bytes);
        } else if (reference) {
            size_t b;
            same = memcmp(reference, got, atlas_bytes) == 0;
            for (b = 0; b < atlas_bytes; b++)
                if (reference[b] != got[b]) {
                    int d = abs((int)reference[b] - (int)got[b]);
                    if (!differing++)
                        first_differing = b;
                    if (d > worst)
                        worst = d;
                }
        }
        printf("{\"path\":\"%s\",\"backend\":\"%s\",\"device\":\"%s\",\"open_ns\":%llu,\"first_glyph_ns\":%llu,"
               "\"all_glyphs_ns\":%llu,\"per_glyph_ns\":%llu,\"frame_with_all_new_ns\":%llu,"
               "\"atlas_equals_cpu\":%s,\"differing_bytes\":%zu,\"largest_difference\":%d,\"first_differing_byte\":%zu,"
               "\"atlas_bytes\":%zu,\"buffers\":%zu,\"weights\":%zu,\"gl_error\":%u}\n",
               path_name[kind], p.gpu ? neuronte_gpu_backend_name(neuronte_gpu_backend(p.gpu)) : "cpu-library",
               p.gpu ? neuronte_gpu_device(p.gpu) : "", (unsigned long long)open_ns, (unsigned long long)first_ns,
               (unsigned long long)median(all, repeats), (unsigned long long)(median(all, repeats) / (uint64_t)count),
               (unsigned long long)median(frame, repeats), same < 0 ? "null" : same ? "true" : "false", differing, worst,
               first_differing, (size_t)atlas_bytes, mem.buffers,
               mem.weights, glGetError());
        fflush(stdout);
        if (p.gpu)
            neuronte_gpu_close(p.gpu);
        if (p.lib)
            neuronte_close(p.lib);
        free(p.cov);
        free(p.px);
    }
    return 0;
}
