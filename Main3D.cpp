// =============================================================================
// 3D Field Simulation — GPU Compute Edition
// Requires OpenGL 4.3
//
// Key design decisions:
//   - Initial condition: Gaussian hot-sphere at centre, rest = 0.
//     This means most of the volume has field ≈ 0 and is transparent;
//     only the high-value diffusing shell is visible.
//   - Opacity = smoothstep on |fieldX|: zero-valued cells are invisible,
//     high-value cells glow fully.  Low regions literally do not exist
//     visually until diffusion reaches them.
//   - Additive blending: no depth sort needed; dense regions accumulate glow.
//   - True 3D toroidal topology: all 6 neighbours wrap at every boundary.
//   - CFL-limited substeps: dt ≤ h²/(6·D) enforced at any resolution.
// =============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

static constexpr int   RES3D = 64;
static constexpr float DIFFUSION = 0.01f;
static constexpr float CAM_DIST = 3.5f;
static constexpr float FOV_Y = 60.0f * 3.14159265f / 180.0f;

// =============================================================================
// Bitmap font (4×6 px per glyph packed into uint32_t)
// bit index = row*4 + col,  col 0 = leftmost,  row 0 = top
// =============================================================================
static const uint32_t FONT[] = {
    0x6 | (0x9 << 4) | (0x9 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20), // 0
    0x2 | (0x3 << 4) | (0x2 << 8) | (0x2 << 12) | (0x2 << 16) | (0xE << 20), // 1
    0x7 | (0x8 << 4) | (0x6 << 8) | (0x1 << 12) | (0x1 << 16) | (0xF << 20), // 2
    0x7 | (0x8 << 4) | (0x6 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20), // 3
    0x9 | (0x9 << 4) | (0xF << 8) | (0x8 << 12) | (0x8 << 16) | (0x8 << 20), // 4
    0xF | (0x1 << 4) | (0x7 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20), // 5
    0x6 | (0x1 << 4) | (0x7 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20), // 6
    0xF | (0x8 << 4) | (0x4 << 8) | (0x2 << 12) | (0x2 << 16) | (0x2 << 20), // 7
    0x6 | (0x9 << 4) | (0x6 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20), // 8
    0x6 | (0x9 << 4) | (0xE << 8) | (0x8 << 12) | (0x8 << 16) | (0x6 << 20), // 9
    0x0 | (0x0 << 4) | (0x0 << 8) | (0x0 << 12) | (0x2 << 16) | (0x0 << 20), // 10 '.'
    0xE | (0x1 << 4) | (0x6 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20), // 11 's'
    0x0,                                                   // 12 ' '
};
static constexpr int FONT_COUNT = 13;

static int formatSimTime(float t, uint32_t* out, int maxOut)
{
    if (t < 0.0f) t = 0.0f;
    int intPart = static_cast<int>(t);
    int frac = static_cast<int>(t * 10.0f) % 10;
    uint32_t digits[8]; int nd = 0;
    if (intPart == 0) { digits[nd++] = 0; }
    else {
        int tmp = intPart; while (tmp > 0 && nd < 7) { digits[nd++] = tmp % 10; tmp /= 10; }
        for (int i = 0, j = nd - 1; i < j; ++i, --j)std::swap(digits[i], digits[j]);
    }
    int n = 0;
    for (int i = 0; i < nd && n < maxOut - 3; ++i) out[n++] = digits[i];
    if (n < maxOut) out[n++] = 10;
    if (n < maxOut) out[n++] = static_cast<uint32_t>(frac);
    if (n < maxOut) out[n++] = 11;
    return n;
}

// =============================================================================
// Compute shader
//
// SSBO layout per cell (STRIDE=10 floats, std430):
//   [0] fieldX  [1] fieldY  [2] fieldZ
//   [3] dxdt    [4] dydt    [5] dzdt
//   [6] d2xdt2  [7] d2ydt2  [8] d2zdt2
//   [9] padding
//
// Toroidal wrapping: all six face-neighbours wrap with modulo arithmetic.
// This means the grid has NO boundaries — the opposite face of a cube is
// directly adjacent, exactly like the 2D toroidal case.
//
// 6-point Laplacian: ∇²f = (R+L+U+D+F+B - 6·self) / h²
// CFL limit for stability: dt ≤ h²/(6·D)
// =============================================================================
static const char* computeSrc = R"GLSL(
#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

#define STRIDE 10
layout(std430, binding=0) readonly  buffer BufIn  { float inData[];  };
layout(std430, binding=1) writeonly buffer BufOut { float outData[]; };

uniform int   uRes;
uniform float uInvH2;
uniform float uDiffusion;
uniform float uDt;

// Returns the flat float-array base index for cell (x,y,z),
// with toroidal wrapping on all three axes.
int cellBase(int x, int y, int z)
{
    x = (x % uRes + uRes) % uRes;   // handles both overflow AND underflow
    y = (y % uRes + uRes) % uRes;
    z = (z % uRes + uRes) % uRes;
    return (z * uRes * uRes + y * uRes + x) * STRIDE;
}

float get(int base, int comp) { return inData[base + comp]; }

// 6-point 3D Laplacian for one component
float laplacian3D(int xp, int xn, int yp, int yn, int zp, int zn, int c, int comp)
{
    return ( get(xp,comp) + get(xn,comp)
           + get(yp,comp) + get(yn,comp)
           + get(zp,comp) + get(zn,comp)
           - 6.0 * get(c, comp) ) * uInvH2;
}

void main()
{
    ivec3 id = ivec3(gl_GlobalInvocationID);
    if (id.x >= uRes || id.y >= uRes || id.z >= uRes) return;

    // Toroidal neighbours — all six directions wrap correctly at edges
    int c  = cellBase(id.x,   id.y,   id.z  );
    int xp = cellBase(id.x+1, id.y,   id.z  );   // +X wraps at x=N-1 → x=0
    int xn = cellBase(id.x-1, id.y,   id.z  );   // -X wraps at x=0   → x=N-1
    int yp = cellBase(id.x,   id.y+1, id.z  );
    int yn = cellBase(id.x,   id.y-1, id.z  );
    int zp = cellBase(id.x,   id.y,   id.z+1);
    int zn = cellBase(id.x,   id.y,   id.z-1);

    float ax = get(c,6), ay = get(c,7), az = get(c,8);
    float dt2 = 0.5 * uDt * uDt;

    // calculateddt(): diffusion drives velocity from Laplacian
    float vx = uDiffusion * laplacian3D(xp,xn,yp,yn,zp,zn,c,0);
    float vy = uDiffusion * laplacian3D(xp,xn,yp,yn,zp,zn,c,1);
    float vz = uDiffusion * laplacian3D(xp,xn,yp,yn,zp,zn,c,2);

    // updateField(): Verlet integration
    int base = (id.z*uRes*uRes + id.y*uRes + id.x) * STRIDE;
    outData[base+0] = get(c,0) + vx*uDt + ax*dt2;
    outData[base+1] = get(c,1) + vy*uDt + ay*dt2;
    outData[base+2] = get(c,2) + vz*uDt + az*dt2;
    outData[base+3] = vx;   outData[base+4] = vy;   outData[base+5] = vz;
    outData[base+6] = ax;   outData[base+7] = ay;   outData[base+8] = az;
    outData[base+9] = 0.0;
}
)GLSL";

// =============================================================================
// Field vertex shader
//
// Opacity: smoothstep on |fieldX| — cells near zero are fully transparent,
// high-value cells are fully opaque.  The threshold is scaled to the actual
// field magnitude so the sphere boundary is visually meaningful.
//
// gl_PointSize: perspective-correct voxel sizing.
// =============================================================================
static const char* fieldVertSrc = R"GLSL(
#version 430 core
layout(location=0) in vec3 aPos;
layout(location=1) in uint aIdx;

#define STRIDE 10
layout(std430, binding=0) readonly buffer Field { float field[]; };

uniform mat4  uMVP;
uniform float uPointScale;
uniform float uAlphaLow;    // |field| below this = fully transparent
uniform float uAlphaHigh;   // |field| above this = fully opaque

out vec4 vColor;

vec3 hsvColor(float f)
{
    float hn = atan(f * 0.02) / 3.14159265 + 0.5;
    float hp = hn * 6.0;
    float xc = 1.0 - abs(mod(hp, 2.0) - 1.0);
    int   s  = int(hp) % 6;
    if (s < 0) s += 6;
    if      (s==0) return vec3(1, xc, 0);
    else if (s==1) return vec3(xc,1,  0);
    else if (s==2) return vec3(0, 1,  xc);
    else if (s==3) return vec3(0, xc, 1);
    else if (s==4) return vec3(xc,0,  1);
    else           return vec3(1, 0,  xc);
}

void main()
{
    float fx    = field[int(aIdx) * STRIDE];
    float alpha = smoothstep(uAlphaLow, uAlphaHigh, abs(fx));

    vColor      = vec4(hsvColor(fx), alpha);
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointScale / gl_Position.w;
}
)GLSL";

static const char* fieldFragSrc = R"GLSL(
#version 430 core
in  vec4 vColor;
out vec4 FragColor;
void main()
{
    // Circular soft-edged point sprite
    vec2  uv   = gl_PointCoord - 0.5;
    float dist = dot(uv, uv);
    if (dist > 0.25) discard;
    float edge = 1.0 - smoothstep(0.15, 0.25, dist);
    FragColor  = vec4(vColor.rgb, vColor.a * edge);
}
)GLSL";

// =============================================================================
// Text overlay shaders (instanced bitmap font)
// =============================================================================
static const char* textVertSrc = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;
uniform vec2  uOrigin;
uniform vec2  uCharSize;
uniform float uAdvance;
uniform uint  uFont[13];
uniform uint  uChars[16];
out vec2 vUV;
flat out uint vGlyph;
void main()
{
    vec2 cell = uOrigin + vec2(float(gl_InstanceID) * uAdvance, 0.0);
    gl_Position = vec4(cell + vec2(aPos.x*uCharSize.x, -aPos.y*uCharSize.y), 0.0, 1.0);
    vUV    = aPos;
    vGlyph = uFont[uChars[gl_InstanceID]];
}
)GLSL";
static const char* textFragSrc = R"GLSL(
#version 430 core
in  vec2 vUV;
flat in uint vGlyph;
uniform vec4 uTextColor;
out vec4 FragColor;
void main()
{
    int col = clamp(int(vUV.x*4.0),0,3);
    int row = clamp(int(vUV.y*6.0),0,5);
    if (((vGlyph >> uint(row*4+col)) & 1u) == 0u) discard;
    FragColor = uTextColor;
}
)GLSL";
static const char* rectVertSrc = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;
uniform vec2 uRectOrigin;
uniform vec2 uRectSize;
void main() {
    gl_Position = vec4(uRectOrigin + vec2(aPos.x*uRectSize.x, -aPos.y*uRectSize.y), 0.0, 1.0);
}
)GLSL";
static const char* rectFragSrc = R"GLSL(
#version 430 core
uniform vec4 uRectColor;
out vec4 FragColor;
void main() { FragColor = uRectColor; }
)GLSL";

// =============================================================================
// Camera orbit state
// =============================================================================
struct CamState {
    bool   dragging = false;
    double lastX = 0, lastY = 0;
    float  yaw = 0.5f, pitch = 0.4f, sensitivity = 0.005f;
};
static CamState cam;

static void mouseButtonCallback(GLFWwindow* w, int btn, int action, int)
{
    if (btn == GLFW_MOUSE_BUTTON_RIGHT) {
        cam.dragging = (action == GLFW_PRESS);
        if (cam.dragging) glfwGetCursorPos(w, &cam.lastX, &cam.lastY);
    }
}
static void cursorPosCallback(GLFWwindow*, double x, double y)
{
    if (!cam.dragging) return;
    cam.yaw += float(x - cam.lastX) * cam.sensitivity;
    cam.pitch += float(y - cam.lastY) * cam.sensitivity;
    cam.pitch = std::clamp(cam.pitch, -1.5707963f, 1.5707963f);
    cam.lastX = x; cam.lastY = y;
}

// =============================================================================
// Matrix math (column-major for OpenGL)
// =============================================================================
static void mat4Mul(const float* a, const float* b, float* c)
{
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            c[col * 4 + row] = 0;
            for (int k = 0; k < 4; k++) c[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
        }
}
static void mat4Persp(float fovY, float aspect, float near, float far, float* m)
{
    float f = 1.0f / tanf(fovY * 0.5f), r = 1.0f / (near - far);
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0] = f / aspect; m[5] = f;
    m[10] = (near + far) * r; m[11] = -1.0f;
    m[14] = 2.0f * near * far * r;
}
static void buildMVP(float yaw, float pitch, float dist, float* mvp)
{
    float cy = cosf(yaw), sy = sinf(yaw), cx = cosf(pitch), sx = sinf(pitch);
    float ry[16] = { cy,0,-sy,0, 0,1,0,0, sy,0,cy,0, 0,0,0,1 };
    float rx[16] = { 1,0,0,0,    0,cx,sx,0, 0,-sx,cx,0, 0,0,0,1 };
    float t[16] = { 1,0,0,0,    0,1,0,0,   0,0,1,0,    0,0,-dist,1 };
    float p[16]; mat4Persp(FOV_Y, 1.0f, 0.1f, 20.0f, p);
    float tmp1[16], tmp2[16], tmp3[16];
    mat4Mul(rx, ry, tmp1);
    mat4Mul(t, tmp1, tmp2);
    mat4Mul(p, tmp2, mvp);
}

// =============================================================================
// Shader helpers
// =============================================================================
static unsigned int compileShader(GLenum type, const char* src)
{
    unsigned int s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr); glCompileShader(s);
    int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(s, 2048, nullptr, log); std::cerr << log << "\n"; }
    return s;
}
static unsigned int makeProgram(std::initializer_list<unsigned int> shaders)
{
    unsigned int p = glCreateProgram();
    for (auto s : shaders) glAttachShader(p, s);
    glLinkProgram(p);
    int ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, 2048, nullptr, log); std::cerr << log << "\n"; }
    for (auto s : shaders) glDeleteShader(s);
    return p;
}

// =============================================================================
// main
// =============================================================================
int main()
{
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 800, "3D Field Simulation", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n"; return -1;
    }
    glViewport(0, 0, 800, 800);

    // ── Programs ──────────────────────────────────────────────────────────────
    unsigned int computeProg = makeProgram({ compileShader(GL_COMPUTE_SHADER,  computeSrc) });
    unsigned int fieldProg = makeProgram({ compileShader(GL_VERTEX_SHADER,   fieldVertSrc),
                                            compileShader(GL_FRAGMENT_SHADER, fieldFragSrc) });
    unsigned int textProg = makeProgram({ compileShader(GL_VERTEX_SHADER,   textVertSrc),
                                            compileShader(GL_FRAGMENT_SHADER, textFragSrc) });
    unsigned int rectProg = makeProgram({ compileShader(GL_VERTEX_SHADER,   rectVertSrc),
                                            compileShader(GL_FRAGMENT_SHADER, rectFragSrc) });

    // Uniform locations
    int uRes = glGetUniformLocation(computeProg, "uRes");
    int uInvH2u = glGetUniformLocation(computeProg, "uInvH2");
    int uDiffusion = glGetUniformLocation(computeProg, "uDiffusion");
    int uDt = glGetUniformLocation(computeProg, "uDt");
    int uMVP = glGetUniformLocation(fieldProg, "uMVP");
    int uPointScale = glGetUniformLocation(fieldProg, "uPointScale");
    int uAlphaLow = glGetUniformLocation(fieldProg, "uAlphaLow");
    int uAlphaHigh = glGetUniformLocation(fieldProg, "uAlphaHigh");
    int uTxtOrigin = glGetUniformLocation(textProg, "uOrigin");
    int uTxtSize = glGetUniformLocation(textProg, "uCharSize");
    int uTxtAdvance = glGetUniformLocation(textProg, "uAdvance");
    int uTxtFont = glGetUniformLocation(textProg, "uFont");
    int uTxtChars = glGetUniformLocation(textProg, "uChars");
    int uTxtColor = glGetUniformLocation(textProg, "uTextColor");
    int uRectOrigin = glGetUniformLocation(rectProg, "uRectOrigin");
    int uRectSize = glGetUniformLocation(rectProg, "uRectSize");
    int uRectColor = glGetUniformLocation(rectProg, "uRectColor");

    // ── CFL stability for 3D: dt ≤ h²/(6·D) ──────────────────────────────────
    const int   N = RES3D;
    const float h = (N > 1) ? (2.0f / (N - 1)) : 1.0f;
    const float invH2 = 1.0f / (h * h);
    const float dtStable = (h * h) / (6.0f * DIFFUSION);
    const int   substeps = std::max(1, (int)std::ceil(0.016f / dtStable));
    const float subDt = dtStable * 0.9f;

    std::cout << "3D grid: " << N << "³  substeps/frame: " << substeps
        << "  subDt: " << subDt << "\n";

    // ── SSBOs ─────────────────────────────────────────────────────────────────
    const int STRIDE = 10;
    std::vector<float> initBuf(N * N * N * STRIDE, 0.0f);

    // Initial condition: Gaussian hot-sphere at the grid centre.
    // Outside the sphere, field = 0 → transparent.
    // Inside the sphere, field peaks at +100 → fully visible.
    // This gives a clear visual from frame 1 and lets diffusion spread
    // outward into transparent space — much more readable than ±100 everywhere.
    const float centre = (N - 1) * 0.5f;
    const float sigma = N * 0.15f;   // sphere radius ≈ 15% of grid width
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                float dx = x - centre, dy = y - centre, dz = z - centre;
                float r2 = dx * dx + dy * dy + dz * dz;
                float val = 100.0f * expf(-r2 / (2.0f * sigma * sigma));
                initBuf[(z * N * N + y * N + x) * STRIDE + 0] = val;   // fieldX
            }

    unsigned int ssbo[2];
    glGenBuffers(2, ssbo);
    for (int b = 0; b < 2; b++) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[b]);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
            initBuf.size() * sizeof(float),
            b == 0 ? initBuf.data() : nullptr,
            GL_DYNAMIC_COPY);
    }

    // ── Point cloud VBO (one point per cell, positions fixed) ─────────────────
    struct Vert3D { float x, y, z; uint32_t idx; };
    std::vector<Vert3D> cloud;
    cloud.reserve(N * N * N);
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                cloud.push_back({ -1.0f + x * h, -1.0f + y * h, -1.0f + z * h,
                                 static_cast<uint32_t>(z * N * N + y * N + x) });

    unsigned int cloudVAO, cloudVBO;
    glGenVertexArrays(1, &cloudVAO);
    glGenBuffers(1, &cloudVBO);
    glBindVertexArray(cloudVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cloudVBO);
    glBufferData(GL_ARRAY_BUFFER, cloud.size() * sizeof(Vert3D), cloud.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vert3D), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, sizeof(Vert3D), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // ── Unit quad VAO (shared by text + rect) ─────────────────────────────────
    float quadV[] = { 0,0, 1,0, 0,1, 1,0, 1,1, 0,1 };
    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadV), quadV, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // ── Perspective-correct point scale ───────────────────────────────────────
    float focalLen = 1.0f / tanf(FOV_Y * 0.5f);
    float pointScale = h * focalLen * 400.0f;

    // Opacity thresholds — tune these to taste.
    // With a Gaussian sphere peaking at 100, these settings make:
    //   |fx| < 3   → fully transparent (the empty space)
    //   |fx| > 25  → fully opaque      (core of the sphere)
    //   3–25       → smooth S-curve fade (the diffusing shell)
    const float ALPHA_LOW = 3.0f;
    const float ALPHA_HIGH = 25.0f;

    // ── Timer layout ──────────────────────────────────────────────────────────
    const float PX = 2.0f / 800.0f;
    const float CHAR_W = 20.0f * PX, CHAR_H = 30.0f * PX;
    const float CHAR_ADV = 22.0f * PX, MARGIN = 12.0f * PX, PAD = 6.0f * PX;
    const float TEXT_X = -1.0f + MARGIN, TEXT_Y = 1.0f - MARGIN;

    const int groups = (N + 7) / 8;
    int       current = 0;
    float     simTime = 0.0f;
    float     mvp[16];

    glEnable(GL_PROGRAM_POINT_SIZE);

    while (!glfwWindowShouldClose(window))
    {
        // ── Compute: substeps ─────────────────────────────────────────────────
        glUseProgram(computeProg);
        glUniform1i(uRes, N);
        glUniform1f(uInvH2u, invH2);
        glUniform1f(uDiffusion, DIFFUSION);
        glUniform1f(uDt, subDt);

        for (int s = 0; s < substeps; s++) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
            glDispatchCompute(groups, groups, groups);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            current = 1 - current;
        }
        simTime += subDt * static_cast<float>(substeps);

        // ── Clear ─────────────────────────────────────────────────────────────
        glClearColor(0.03f, 0.03f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ── Field pass: additive blending ─────────────────────────────────────
        // No depth writes — transparent points must not occlude each other.
        // Additive blend: colour accumulates; zero-alpha points add nothing.
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        glUseProgram(fieldProg);
        buildMVP(cam.yaw, cam.pitch, CAM_DIST, mvp);
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, mvp);
        glUniform1f(uPointScale, pointScale);
        glUniform1f(uAlphaLow, ALPHA_LOW);
        glUniform1f(uAlphaHigh, ALPHA_HIGH);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
        glBindVertexArray(cloudVAO);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(cloud.size()));

        // ── UI pass: normal alpha blend, no depth ─────────────────────────────
        glDepthMask(GL_TRUE);
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        uint32_t charBuf[16];
        int numChars = formatSimTime(simTime, charBuf, 16);
        float textW = CHAR_W + (numChars - 1) * CHAR_ADV;

        glUseProgram(rectProg);
        glUniform2f(uRectOrigin, TEXT_X - PAD, TEXT_Y + PAD);
        glUniform2f(uRectSize, textW + 2.0f * PAD, CHAR_H + 2.0f * PAD);
        glUniform4f(uRectColor, 0, 0, 0, 0.65f);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glUseProgram(textProg);
        glUniform2f(uTxtOrigin, TEXT_X, TEXT_Y);
        glUniform2f(uTxtSize, CHAR_W, CHAR_H);
        glUniform1f(uTxtAdvance, CHAR_ADV);
        glUniform1uiv(uTxtFont, FONT_COUNT, FONT);
        glUniform1uiv(uTxtChars, numChars, charBuf);
        glUniform4f(uTxtColor, 1, 1, 1, 1);
        glBindVertexArray(quadVAO);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, numChars);

        glDisable(GL_BLEND);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteBuffers(2, ssbo);
    glDeleteBuffers(1, &cloudVBO);  glDeleteVertexArrays(1, &cloudVAO);
    glDeleteBuffers(1, &quadVBO);   glDeleteVertexArrays(1, &quadVAO);
    glDeleteProgram(computeProg);   glDeleteProgram(fieldProg);
    glDeleteProgram(textProg);      glDeleteProgram(rectProg);
    glfwTerminate();
    return 0;
}