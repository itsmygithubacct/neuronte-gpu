/* neuronte-gpu OpenGL backend for file handles: the compute shader of the
 * Vulkan backend (kernels/cover2.comp) in an OpenGL 4.3 context.
 *
 * Two ways to get the context:
 *   - its own: a surfaceless EGL context on a device chosen by name, for
 *     programs that have no OpenGL of their own (the benchmark);
 *   - the caller's (NEURONTE_GPU_GL_CURRENT): whatever context is current
 *     on the calling thread at open and at every call, for a program that
 *     draws with OpenGL and wants the glyphs evaluated where it draws.
 *     The program bound and nothing else of the caller's state is changed.
 *
 * OpenGL and EGL are loaded at run time and declared here, so the build
 * needs no headers. In exact mode every value is `precise`; whether a
 * driver then rounds each multiply and add separately is its own matter,
 * so exactness is measured per driver (tools/exact.py), not assumed.
 */
#define _GNU_SOURCE
#include "internal.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned GLenum, GLuint, GLbitfield;
typedef int GLint, GLsizei;
typedef char GLchar;
typedef unsigned char GLubyte;
typedef ptrdiff_t GLsizeiptr, GLintptr;
typedef uint64_t GLuint64;

#define GL_VERSION 0x1F02
#define GL_RENDERER 0x1F01
#define GL_VENDOR 0x1F00
#define GL_MAJOR_VERSION 0x821B
#define GL_MINOR_VERSION 0x821C
#define GL_COMPUTE_SHADER 0x91B9
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_DYNAMIC_READ 0x88E9
#define GL_DYNAMIC_COPY 0x88EA
#define GL_SHADER_STORAGE_BARRIER_BIT 0x2000
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x0200
#define GL_CURRENT_PROGRAM 0x8B8D
#define GL_TIME_ELAPSED 0x88BF
#define GL_QUERY_RESULT 0x8866
#define GL_MAX_COMPUTE_WORK_GROUP_COUNT 0x91BE
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 0x90EB
#define GL_MAX_COMPUTE_SHARED_MEMORY_SIZE 0x8262
#define GL_SHADER_BINARY_FORMAT_SPIR_V 0x9551

typedef void *EGLDisplay, *EGLConfig, *EGLContext, *EGLSurface, *EGLDeviceEXT;
typedef int EGLint;
typedef unsigned EGLBoolean, EGLenum;
#define EGL_OPENGL_API 0x30A2
#define EGL_NONE 0x3038
#define EGL_RENDERABLE_TYPE 0x3040
#define EGL_OPENGL_BIT 0x0008
#define EGL_CONTEXT_MAJOR_VERSION 0x3098
#define EGL_CONTEXT_MINOR_VERSION 0x30FB
#define EGL_CONTEXT_OPENGL_PROFILE_MASK 0x30FD
#define EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT 0x00000001
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD

#define NBUF2 8
enum { F_PARAMS, F_SEGS, F_MASKS, F_PAIRS, F_OUT, F_BLOCKS, F_SAMP, F_LIST };

struct gl {
    void *dl, *dl46, *egl;
    const GLubyte *(*GetString)(GLenum);
    void (*GetIntegerv)(GLenum, GLint *);
    void (*GetIntegeri_v)(GLenum, GLuint, GLint *);
    GLenum (*GetError)(void);
    GLuint (*CreateShader)(GLenum);
    void (*ShaderSource)(GLuint, GLsizei, const GLchar *const *, const GLint *);
    void (*CompileShader)(GLuint);
    void (*GetShaderiv)(GLuint, GLenum, GLint *);
    void (*GetShaderInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
    GLuint (*CreateProgram)(void);
    void (*AttachShader)(GLuint, GLuint);
    void (*LinkProgram)(GLuint);
    void (*GetProgramiv)(GLuint, GLenum, GLint *);
    void (*GetProgramInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
    void (*DeleteShader)(GLuint);
    void (*DeleteProgram)(GLuint);
    void (*UseProgram)(GLuint);
    void (*Uniform1i)(GLint, GLint);
    void (*GenBuffers)(GLsizei, GLuint *);
    void (*DeleteBuffers)(GLsizei, const GLuint *);
    void (*BindBuffer)(GLenum, GLuint);
    void (*BufferData)(GLenum, GLsizeiptr, const void *, GLenum);
    void (*BufferSubData)(GLenum, GLintptr, GLsizeiptr, const void *);
    void (*GetBufferSubData)(GLenum, GLintptr, GLsizeiptr, void *);
    void (*BindBufferBase)(GLenum, GLuint, GLuint);
    void (*DispatchCompute)(GLuint, GLuint, GLuint);
    void (*MemoryBarrier)(GLbitfield);
    void (*Finish)(void);
    void (*GenQueries)(GLsizei, GLuint *);
    void (*DeleteQueries)(GLsizei, const GLuint *);
    void (*BeginQuery)(GLenum, GLuint);
    void (*EndQuery)(GLenum);
    void (*GetQueryObjectui64v)(GLuint, GLenum, GLuint64 *);
    /* OpenGL 4.6, absent before */
    void (*ShaderBinary)(GLsizei, const GLuint *, GLenum, const void *, GLsizei);
    void (*SpecializeShader)(GLuint, const GLchar *, GLuint, const GLuint *, const GLuint *);
    /* EGL, for a context of our own */
    EGLDisplay (*eGetDisplay)(void *);
    EGLDisplay (*eGetPlatformDisplay)(EGLenum, void *, const intptr_t *);
    EGLBoolean (*eQueryDevices)(EGLint, EGLDeviceEXT *, EGLint *);
    EGLBoolean (*eInitialize)(EGLDisplay, EGLint *, EGLint *);
    EGLBoolean (*eBindAPI)(EGLenum);
    EGLBoolean (*eChooseConfig)(EGLDisplay, const EGLint *, EGLConfig *, EGLint, EGLint *);
    EGLContext (*eCreateContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint *);
    EGLBoolean (*eMakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
    EGLBoolean (*eDestroyContext)(EGLDisplay, EGLContext);
    EGLBoolean (*eTerminate)(EGLDisplay);
};

struct impl {
    struct gl gl;
    neuronte_gpu *g;
    int own; /* the context is ours */
    EGLDisplay dpy;
    EGLContext ctx;
    GLuint buf[NBUF2];
    size_t size[NBUF2];
    GLuint *prog, reduce, query;
    GLint max_groups;
    double *rate, *work, slice_ns;
};

static int load_gl(struct gl *gl, int own, char *why, size_t len)
{
    static const char *names[] = {"libOpenGL.so.0", "libGL.so.1"};
    size_t i;
    for (i = 0; i < 2 && !gl->dl; i++)
        gl->dl = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!gl->dl) {
        neuronte_gpu_say(why, len, "no OpenGL library (libOpenGL.so.0)");
        return -1;
    }
#define SYM(field, name)                                             \
    do {                                                             \
        *(void **)&gl->field = dlsym(gl->dl, name);                  \
        if (!gl->field) {                                            \
            neuronte_gpu_say(why, len, "OpenGL lacks %s", name);     \
            return -1;                                               \
        }                                                            \
    } while (0)
    SYM(GetString, "glGetString");
    SYM(GetIntegerv, "glGetIntegerv");
    SYM(GetIntegeri_v, "glGetIntegeri_v");
    SYM(GetError, "glGetError");
    SYM(CreateShader, "glCreateShader");
    SYM(ShaderSource, "glShaderSource");
    SYM(CompileShader, "glCompileShader");
    SYM(GetShaderiv, "glGetShaderiv");
    SYM(GetShaderInfoLog, "glGetShaderInfoLog");
    SYM(CreateProgram, "glCreateProgram");
    SYM(AttachShader, "glAttachShader");
    SYM(LinkProgram, "glLinkProgram");
    SYM(GetProgramiv, "glGetProgramiv");
    SYM(GetProgramInfoLog, "glGetProgramInfoLog");
    SYM(DeleteShader, "glDeleteShader");
    SYM(DeleteProgram, "glDeleteProgram");
    SYM(UseProgram, "glUseProgram");
    SYM(Uniform1i, "glUniform1i");
    SYM(GenBuffers, "glGenBuffers");
    SYM(DeleteBuffers, "glDeleteBuffers");
    SYM(BindBuffer, "glBindBuffer");
    SYM(BufferData, "glBufferData");
    SYM(BufferSubData, "glBufferSubData");
    SYM(GetBufferSubData, "glGetBufferSubData");
    SYM(BindBufferBase, "glBindBufferBase");
    SYM(DispatchCompute, "glDispatchCompute");
    SYM(MemoryBarrier, "glMemoryBarrier");
    SYM(Finish, "glFinish");
    SYM(GenQueries, "glGenQueries");
    SYM(DeleteQueries, "glDeleteQueries");
    SYM(BeginQuery, "glBeginQuery");
    SYM(EndQuery, "glEndQuery");
    SYM(GetQueryObjectui64v, "glGetQueryObjectui64v");
#undef SYM
    *(void **)&gl->ShaderBinary = dlsym(gl->dl, "glShaderBinary");
    *(void **)&gl->SpecializeShader = dlsym(gl->dl, "glSpecializeShader");
    if (!gl->SpecializeShader && (gl->dl46 = dlopen("libGL.so.1", RTLD_NOW | RTLD_LOCAL)))
        /* the vendor-neutral libOpenGL stops at 4.5; libGL of the same dispatch has the 4.6 entry */
        *(void **)&gl->SpecializeShader = dlsym(gl->dl46, "glSpecializeShader");
    if (!own)
        return 0;
    gl->egl = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!gl->egl) {
        neuronte_gpu_say(why, len, "no EGL library (libEGL.so.1)");
        return -1;
    }
#define SYM(field, name) *(void **)&gl->field = dlsym(gl->egl, name)
    SYM(eGetDisplay, "eglGetDisplay");
    SYM(eGetPlatformDisplay, "eglGetPlatformDisplay");
    SYM(eInitialize, "eglInitialize");
    SYM(eBindAPI, "eglBindAPI");
    SYM(eChooseConfig, "eglChooseConfig");
    SYM(eCreateContext, "eglCreateContext");
    SYM(eMakeCurrent, "eglMakeCurrent");
    SYM(eDestroyContext, "eglDestroyContext");
    SYM(eTerminate, "eglTerminate");
#undef SYM
    {
        /* extension functions are not exported by name: the library hands them out */
        void *(*proc)(const char *);
        *(void **)&proc = dlsym(gl->egl, "eglGetProcAddress");
        if (proc)
            *(void **)&gl->eQueryDevices = proc("eglQueryDevicesEXT");
    }
    if (!gl->eGetDisplay || !gl->eInitialize || !gl->eBindAPI || !gl->eChooseConfig || !gl->eCreateContext ||
        !gl->eMakeCurrent || !gl->eDestroyContext || !gl->eTerminate) {
        neuronte_gpu_say(why, len, "EGL lacks a function this backend needs");
        return -1;
    }
    return 0;
}

/* A surfaceless OpenGL 4.3 core context on `dpy`, made current; 0 when it has one, else the step
 * that failed. */
static int try_display(struct impl *m, EGLDisplay dpy)
{
    struct gl *gl = &m->gl;
    static const EGLint cfg_attr[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    static const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3,
                                      EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
    EGLConfig cfg = NULL;
    EGLint n = 0, major, minor;
    EGLContext ctx;
    if (!dpy)
        return 1;
    if (!gl->eInitialize(dpy, &major, &minor))
        return 2;
    if (!gl->eBindAPI(EGL_OPENGL_API)) {
        gl->eTerminate(dpy);
        return 3;
    }
    /* a context without a config where the driver allows it, else any config that can do OpenGL */
    ctx = gl->eCreateContext(dpy, NULL, NULL, ctx_attr);
    if (!ctx && gl->eChooseConfig(dpy, cfg_attr, &cfg, 1, &n) && n > 0)
        ctx = gl->eCreateContext(dpy, cfg, NULL, ctx_attr);
    if (!ctx) {
        gl->eTerminate(dpy);
        return 4;
    }
    if (!gl->eMakeCurrent(dpy, NULL, NULL, ctx)) {
        gl->eDestroyContext(dpy, ctx);
        gl->eTerminate(dpy);
        return 5;
    }
    m->dpy = dpy;
    m->ctx = ctx;
    return 0;
}

static void drop_context(struct impl *m)
{
    if (m->ctx) {
        m->gl.eMakeCurrent(m->dpy, NULL, NULL, NULL);
        m->gl.eDestroyContext(m->dpy, m->ctx);
        m->gl.eTerminate(m->dpy);
    }
    m->ctx = NULL;
    m->dpy = NULL;
}

/* Whether the current context is the device asked for; its name is added to `seen`. */
static int wanted(struct impl *m, const char *want, int allow_cpu, char *seen, size_t len)
{
    struct gl *gl = &m->gl;
    const char *r = (const char *)gl->GetString(GL_RENDERER), *v = (const char *)gl->GetString(GL_VENDOR);
    size_t used = strlen(seen);
    snprintf(seen + used, len - used, "%s%s, %s", used ? "; " : "", v ? v : "?", r ? r : "?");
    if (!r || !v)
        return 0;
    if (!allow_cpu && (strcasestr(r, "llvmpipe") || strcasestr(r, "softpipe")))
        return 0;
    return !want || strcasestr(r, want) || strcasestr(v, want);
}

/* A context of our own on the device whose vendor or renderer name contains `want` (any when NULL). */
static int own_context(struct impl *m, const char *want, int allow_cpu, char *why, size_t len)
{
    struct gl *gl = &m->gl;
    EGLDeviceEXT devs[16];
    EGLint ndev = 0, i;
    char seen[512] = "";
    /* one display per device, where EGL can list devices; then the default surfaceless display */
    if (gl->eQueryDevices && gl->eGetPlatformDisplay && gl->eQueryDevices(16, devs, &ndev)) {
        for (i = 0; i < ndev; i++) {
            int step = try_display(m, gl->eGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devs[i], NULL));
            if (step) {
                size_t used = strlen(seen);
                snprintf(seen + used, sizeof seen - used, "%sdevice %d: no context (step %d)", used ? "; " : "", i, step);
                continue;
            }
            if (wanted(m, want, allow_cpu, seen, sizeof seen))
                return 0;
            drop_context(m);
        }
    }
    if (gl->eGetPlatformDisplay) {
        int step = try_display(m, gl->eGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL));
        if (step) {
            size_t used = strlen(seen);
            snprintf(seen + used, sizeof seen - used, "%ssurfaceless: no context (step %d)", used ? "; " : "", step);
        } else {
            if (wanted(m, want, allow_cpu, seen, sizeof seen))
                return 0;
            drop_context(m);
        }
    }
    neuronte_gpu_say(why, len, "no OpenGL 4.3 device%s%s%s through EGL (%d listed; saw: %s)",
                     want ? " matching '" : "", want ? want : "", want ? "'" : "", (int)ndev, seen[0] ? seen : "none");
    return -1;
}

static void impl_close(void *p)
{
    struct impl *m = p;
    struct gl *gl;
    int i;
    if (!m)
        return;
    gl = &m->gl;
    if (m->own && m->ctx)
        gl->eMakeCurrent(m->dpy, NULL, NULL, m->ctx);
    if (!m->own || m->ctx) {
        for (i = 0; i < NBUF2; i++)
            if (m->buf[i])
                gl->DeleteBuffers(1, &m->buf[i]);
        for (i = 0; m->prog && i < m->g->plan.nkern; i++)
            if (m->prog[i])
                gl->DeleteProgram(m->prog[i]);
        if (m->reduce)
            gl->DeleteProgram(m->reduce);
        if (m->query)
            gl->DeleteQueries(1, &m->query);
    }
    if (m->own)
        drop_context(m);
    if (gl->egl)
        dlclose(gl->egl);
    if (gl->dl46)
        dlclose(gl->dl46);
    if (gl->dl)
        dlclose(gl->dl);
    free(m->prog);
    free(m->rate);
    free(m->work);
    free(m);
}

static GLuint linked(struct impl *m, GLuint sh, char *why, size_t len)
{
    struct gl *gl = &m->gl;
    GLuint prog = gl->CreateProgram();
    GLint ok = 0;
    char log[1024] = "";
    gl->AttachShader(prog, sh);
    gl->LinkProgram(prog);
    gl->DeleteShader(sh);
    gl->GetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        gl->GetProgramInfoLog(prog, sizeof log, NULL, log);
        neuronte_gpu_say(why, len, "linking the compute shader failed: %.400s", log);
        gl->DeleteProgram(prog);
        return 0;
    }
    return prog;
}

static GLuint build(struct impl *m, const char *head, const char *body, char *why, size_t len)
{
    struct gl *gl = &m->gl;
    const GLchar *src[2] = {head, body};
    GLuint sh = gl->CreateShader(GL_COMPUTE_SHADER);
    GLint ok = 0;
    char log[1024] = "";
    gl->ShaderSource(sh, 2, src, NULL);
    gl->CompileShader(sh);
    gl->GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        gl->GetShaderInfoLog(sh, sizeof log, NULL, log);
        neuronte_gpu_say(why, len, "compiling the compute shader failed: %.400s", log);
        gl->DeleteShader(sh);
        return 0;
    }
    return linked(m, sh, why, len);
}

/* The kernel from SPIR-V, specialised as the Vulkan backend specialises it; 0 when the driver
 * will not take it (the caller then compiles the text). */
static GLuint build_spirv(struct impl *m, const struct embedded_spirv *spv, int threads, int h, int gp_max, int wide)
{
    struct gl *gl = &m->gl;
    const GLuint idx[4] = {0, 1, 2, 3}, val[4] = {(GLuint)threads, (GLuint)h, (GLuint)gp_max, (GLuint)wide};
    GLuint sh = gl->CreateShader(GL_COMPUTE_SHADER), prog;
    GLint ok = 0;
    char why[512] = "";
    gl->ShaderBinary(1, &sh, GL_SHADER_BINARY_FORMAT_SPIR_V, spv->code, (GLsizei)(spv->words * sizeof(uint32_t)));
    gl->SpecializeShader(sh, "main", 4, idx, val);
    gl->GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok || gl->GetError() != 0) {
        if (getenv("NEURONTE_GPU_DEBUG")) {
            char log[1024] = "";
            gl->GetShaderInfoLog(sh, sizeof log, NULL, log);
            fprintf(stderr, "neuronte-gpu: the driver refused the SPIR-V kernel (%s); compiling the text\n", log);
        }
        while (gl->GetError() != 0)
            ;
        gl->DeleteShader(sh);
        return 0;
    }
    prog = linked(m, sh, why, sizeof why);
    if (!prog && getenv("NEURONTE_GPU_DEBUG"))
        fprintf(stderr, "neuronte-gpu: the SPIR-V kernel did not link (%s); compiling the text\n", why);
    return prog;
}

/* (Re)make buffer `which` with `bytes` of `data` (NULL: uninitialised). */
static void store(struct impl *m, int which, size_t bytes, const void *data, GLenum usage)
{
    struct gl *gl = &m->gl;
    gl->BindBuffer(GL_SHADER_STORAGE_BUFFER, m->buf[which]);
    gl->BufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(bytes ? bytes : 16), data, usage);
    m->g->device_bytes += (bytes ? bytes : 16) - m->size[which];
    m->size[which] = bytes ? bytes : 16;
}

/* This call's records into buffer `which`, which grows when it must. */
static void fill(struct impl *m, int which, size_t bytes, const void *data)
{
    struct gl *gl = &m->gl;
    if (bytes > m->size[which]) {
        size_t want = m->size[which] < 16384 ? 16384 : m->size[which];
        while (want < bytes)
            want *= 2;
        store(m, which, want, NULL, GL_DYNAMIC_DRAW);
    }
    gl->BindBuffer(GL_SHADER_STORAGE_BUFFER, m->buf[which]);
    if (bytes)
        gl->BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)bytes, data);
}

static int impl_run_plan(void *p, int count, float *out, struct neuronte_gpu_timing *t)
{
    struct impl *m = p;
    struct gl *gl = &m->gl;
    neuronte_gpu *g = m->g;
    struct v2plan *pl = &g->plan;
    size_t plane = (size_t)g->cell_w * g->cell_h, bytes = (size_t)count * plane * sizeof(float);
    GLint before = 0;
    GLuint64 ns = 0;
    uint64_t t0, t1, t2;
    int k, i;

    t0 = neuronte_gpu_now();
    if (m->own && !gl->eMakeCurrent(m->dpy, NULL, NULL, m->ctx))
        return -1;
    gl->GetIntegerv(GL_CURRENT_PROGRAM, &before);
    fill(m, F_PAIRS, (size_t)pl->npairs * 2 * sizeof(uint32_t), pl->pairs);
    fill(m, F_BLOCKS, (size_t)pl->nblocks * B_INTS * sizeof(int32_t), pl->blocks);
    if (pl->nreduce)
        fill(m, F_LIST, (size_t)pl->nreduce * sizeof(uint32_t), pl->reduce);
    if (pl->dynamic) {
        gl->BindBuffer(GL_SHADER_STORAGE_BUFFER, m->buf[F_SEGS]);
        gl->BufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)((size_t)pl->nstatic_segs * S_INTS * sizeof(int32_t)),
                          (GLsizeiptr)((size_t)(pl->nsegs - pl->nstatic_segs) * S_INTS * sizeof(int32_t)),
                          pl->segs + (size_t)pl->nstatic_segs * S_INTS);
        gl->BindBuffer(GL_SHADER_STORAGE_BUFFER, m->buf[F_PARAMS]);
        gl->BufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)(pl->nstatic * sizeof(float)),
                          (GLsizeiptr)((pl->nparams - pl->nstatic) * sizeof(float)), pl->params + pl->nstatic);
    }
    for (i = 0; i < NBUF2; i++)
        gl->BindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)i, m->buf[i]);
    t1 = neuronte_gpu_now();
    if (m->query)
        gl->BeginQuery(GL_TIME_ELAPSED, m->query);
    for (k = 0; k < pl->nkern; k++) {
        const struct v2kern *kn = &pl->kern[k];
        uint64_t k0;
        int done;
        m->work[k] = 0;
        if (!kn->nblocks)
            continue;
        gl->Finish();
        k0 = neuronte_gpu_now();
        gl->UseProgram(m->prog[k]);
        gl->Uniform1i(1, g->cell_w);
        gl->Uniform1i(2, g->cell_h);
        gl->Uniform1i(3, g->samples);
        for (done = 0; done < kn->nblocks;) {
            /* a dispatch takes only as many blocks as should finish within the slice (see the
             * Vulkan backend): the display's own frames wait for a dispatch on the same device */
            double budget = m->rate[k] * m->slice_ns, cost = 0;
            GLuint n = 0;
            while (done + (int)n < kn->nblocks && (GLint)n < m->max_groups) {
                const int32_t *b = pl->blocks + (size_t)(kn->block0 + done + (int)n) * B_INTS;
                double hid = pl->segs[(size_t)b[B_SEG] * S_INTS + S_HID];
                double c = (double)(b[B_HI] - b[B_LO]) * (b[B_MODE] < BM_SAMPLE ? g->samples * g->samples : 1) *
                           (2 * hid * hid + 3 * hid);
                if (n && cost + c > budget)
                    break;
                cost += c;
                n++;
            }
            gl->Uniform1i(0, kn->block0 + done);
            gl->DispatchCompute(n, 1, 1);
            done += (int)n;
            m->work[k] += cost;
            t->launches++;
        }
        gl->Finish();
        {
            double took = (double)(neuronte_gpu_now() - k0);
            if (took > 200000.0)
                m->rate[k] = m->work[k] / took;
        }
    }
    if (pl->nreduce) {
        size_t per = (size_t)m->max_groups * 64 / plane;
        int done;
        gl->MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        gl->UseProgram(m->reduce);
        gl->Uniform1i(1, g->cell_w);
        gl->Uniform1i(2, g->cell_h);
        gl->Uniform1i(3, g->samples);
        for (done = 0; per && done < pl->nreduce;) {
            int n = (size_t)(pl->nreduce - done) < per ? pl->nreduce - done : (int)per;
            gl->Uniform1i(0, n);
            gl->Uniform1i(4, done);
            gl->DispatchCompute((GLuint)(((size_t)n * plane + 63) / 64), 1, 1);
            done += n;
            t->launches++;
        }
    }
    if (m->query)
        gl->EndQuery(GL_TIME_ELAPSED);
    /* what follows may read the output with a shader, or as buffer data */
    gl->MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    if (out)
        gl->Finish();
    t2 = neuronte_gpu_now();
    if (out) {
        gl->BindBuffer(GL_SHADER_STORAGE_BUFFER, m->buf[F_OUT]);
        gl->GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)bytes, out);
    }
    if (m->query)
        gl->GetQueryObjectui64v(m->query, GL_QUERY_RESULT, &ns);
    else
        ns = t2 - t1;
    gl->UseProgram((GLuint)before);
    if (gl->GetError() != 0) {
        while (gl->GetError() != 0)
            ;
        return -1;
    }
    t->upload += t1 - t0;
    t->wait += t2 - t1;
    t->run += ns;
    t->download += neuronte_gpu_now() - t2;
    return 0;
}

static int impl_put(void *p, int glyph, const float *plane_data)
{
    struct impl *m = p;
    struct gl *gl = &m->gl;
    size_t plane = (size_t)m->g->cell_w * m->g->cell_h * sizeof(float);
    gl->BindBuffer(GL_SHADER_STORAGE_BUFFER, m->buf[F_OUT]);
    gl->BufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)((size_t)glyph * plane), (GLsizeiptr)plane, plane_data);
    return gl->GetError() != 0;
}

static unsigned impl_output(void *p)
{
    return ((struct impl *)p)->buf[F_OUT];
}

static const struct backend_ops ops = {impl_close, NULL, impl_run_plan, impl_put, impl_output};

int neuronte_gpu_gl_open2(neuronte_gpu *g, const struct neuronte_gpu_options *opt, char *why, size_t len)
{
    struct impl *m = calloc(1, sizeof *m);
    struct gl *gl;
    struct v2plan *pl = &g->plan;
    size_t plane = (size_t)g->cell_w * g->cell_h, splane = plane * g->samples * g->samples;
    GLint major = 0, minor = 0, invocations = 0, shared = 0, before = 0;
    int k, nt = opt->block > 0 ? opt->block : 64;
    const char *renderer, *version, *tune;
    const struct embedded_spirv *spv = NULL;
    char head[512], kinds[256] = "";

    if (!m)
        return -1;
    m->g = g;
    m->own = !(opt->flags & NEURONTE_GPU_GL_CURRENT);
    gl = &m->gl;
    if (load_gl(gl, m->own, why, len))
        goto fail;
    if (m->own && own_context(m, opt->device, (opt->flags & NEURONTE_GPU_ALLOW_CPU_DEVICE) != 0, why, len))
        goto fail;
    renderer = (const char *)gl->GetString(GL_RENDERER);
    version = (const char *)gl->GetString(GL_VERSION);
    if (!renderer || !version) {
        neuronte_gpu_say(why, len, "no OpenGL context is current on this thread");
        goto fail;
    }
    if (!(opt->flags & NEURONTE_GPU_ALLOW_CPU_DEVICE) &&
        (strcasestr(renderer, "llvmpipe") || strcasestr(renderer, "softpipe"))) {
        /* a context drawn by the CPU would run the shader on the CPU, far slower than the library */
        neuronte_gpu_say(why, len, "the OpenGL context is drawn by the CPU (%s)", renderer);
        goto fail;
    }
    gl->GetIntegerv(GL_MAJOR_VERSION, &major);
    gl->GetIntegerv(GL_MINOR_VERSION, &minor);
    if (major < 4 || (major == 4 && minor < 3)) {
        neuronte_gpu_say(why, len, "compute shaders need OpenGL 4.3; this context is %d.%d (%s)", major, minor,
                         renderer);
        goto fail;
    }
    gl->GetIntegerv(GL_CURRENT_PROGRAM, &before);
    gl->GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &m->max_groups);
    gl->GetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &invocations);
    gl->GetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &shared);
    if (nt > invocations) {
        neuronte_gpu_say(why, len, "%s allows %d invocations per workgroup, not %d", renderer, invocations, nt);
        goto fail;
    }
    g->fill_blocks = g->fill_wide = 32;
    if (neuronte_gpu_v2_size(g, nt, nt, nt, 0, (size_t)shared / sizeof(float), why, len))
        goto fail;
    m->prog = calloc((size_t)pl->nkern + 1, sizeof *m->prog);
    m->rate = calloc((size_t)pl->nkern + 1, sizeof *m->rate);
    m->work = calloc((size_t)pl->nkern + 1, sizeof *m->work);
    if (!m->prog || !m->rate || !m->work)
        goto fail;
    /* The kernel can also be loaded as SPIR-V where OpenGL 4.6 allows, with every loop unrolled as in
     * the Vulkan backend. Measured: about four times faster than the text on Mesa's Intel driver,
     * about four times slower on NVIDIA's, and Mesa compiles it afresh at every open (many seconds)
     * where it keeps compiled text in its cache. So it is used only when asked for. */
    tune = getenv("NEURONTE_GPU_GL_SPIRV");
    if (tune && atoi(tune) && (major > 4 || minor >= 6) && gl->ShaderBinary && gl->SpecializeShader)
        for (k = 0; k < neuronte_gpu_spirv_count; k++)
            if (neuronte_gpu_spirv[k].exact == 30 + g->exact)
                spv = &neuronte_gpu_spirv[k];
    if (getenv("NEURONTE_GPU_DEBUG"))
        fprintf(stderr, "neuronte-gpu: opengl %d.%d, SPIR-V loader %s, kernel %s\n", major, minor, gl->ShaderBinary && gl->SpecializeShader ? "present" : "absent", spv ? "found" : "not chosen");
    tune = getenv("NEURONTE_GPU_SLICE_MS");
    m->slice_ns = (tune && atof(tune) > 0 ? atof(tune) : 25.0) * 1e6;
    for (k = 0; k < pl->nkern; k++) {
        const struct v2kern *kn = &pl->kern[k];
        int h = kn->wide ? kn->n * LANES : kn->n;
        size_t used = strlen(kinds);
        m->rate[k] = 2.0; /* until a launch has been timed, assume a slow device */
        snprintf(head, sizeof head,
                 "#version 430\n#define GL 1\n#define NT %d\n#define H %d\n#define GPMAX %d\n#define WIDE %d\n"
                 "#define EXACT %d\n",
                 kn->threads, h, kn->gp_max, kn->wide, g->exact);
        if (spv)
            m->prog[k] = build_spirv(m, spv, kn->threads, h, kn->gp_max, kn->wide);
        if (!m->prog[k]) {
            spv = NULL; /* this driver compiles the text */
            m->prog[k] = build(m, head, neuronte_gpu_glsl_cover2, why, len);
        }
        if (!m->prog[k])
            goto fail;
        snprintf(kinds + used, sizeof kinds - used, "%s%s%d", used ? " " : "", kn->wide ? "wide" : "H", h);
    }
    m->reduce = build(m, "#version 430\n#define GL 1\n", neuronte_gpu_glsl_reduce, why, len);
    if (!m->reduce)
        goto fail;
    gl->GenBuffers(NBUF2, m->buf);
    if (m->own) /* a query belongs to one context; the caller's may be one of several that share objects */
        gl->GenQueries(1, &m->query);
    store(m, F_PARAMS, pl->params_cap * sizeof(float), NULL, GL_STATIC_DRAW);
    gl->BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(pl->nstatic * sizeof(float)), pl->params);
    store(m, F_SEGS, (size_t)pl->segs_cap * S_INTS * sizeof(int32_t), NULL, GL_STATIC_DRAW);
    if (pl->nstatic_segs)
        gl->BufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)((size_t)pl->nstatic_segs * S_INTS * sizeof(int32_t)),
                          pl->segs);
    store(m, F_MASKS, (size_t)pl->nmasks * sizeof(uint32_t), pl->nmasks ? pl->masks : NULL, GL_STATIC_DRAW);
    store(m, F_PAIRS, 16384, NULL, GL_DYNAMIC_DRAW);
    store(m, F_BLOCKS, 16384, NULL, GL_DYNAMIC_DRAW);
    store(m, F_OUT, (size_t)g->max_batch * plane * sizeof(float), NULL, GL_DYNAMIC_READ);
    store(m, F_SAMP, pl->need_samples ? (size_t)g->max_batch * splane * sizeof(float) : 16, NULL, GL_DYNAMIC_COPY);
    store(m, F_LIST, (size_t)g->max_batch * sizeof(uint32_t), NULL, GL_DYNAMIC_DRAW);
    gl->UseProgram((GLuint)before);
    if (gl->GetError() != 0) {
        while (gl->GetError() != 0)
            ;
        neuronte_gpu_say(why, len, "OpenGL reported an error while setting up on %s", renderer);
        goto fail;
    }
    snprintf(g->device, sizeof g->device, "%s", renderer);
    snprintf(g->versions, sizeof g->versions,
             "opengl %s; %s context; %s; file handle: %d nets, %d segments, kernels %s", version,
             m->own ? "own surfaceless EGL" : "the caller's", spv ? "SPIR-V" : "GLSL text", g->nmembers,
             pl->nstatic_segs, kinds);
    g->device_type = strcasestr(renderer, "llvmpipe") || strcasestr(renderer, "softpipe") ? "cpu"
                     : strcasestr(renderer, "Intel") ? "integrated-gpu" : "gpu";
    g->ops = &ops;
    g->impl = m;
    return 0;
fail:
    impl_close(m);
    return -1;
}
