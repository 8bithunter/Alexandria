// =============================================================================
// Field Simulation — GPU Compute Edition
// Requires OpenGL 4.3 (compute shaders + SSBOs)
//
// Simulation modes (keys 1 / 2 / 3):
//   1 – Diffusion   : ∂u/∂t = D ∇²u
//   2 – Wave        : ∂²u/∂t² = c² ∇²u
//   3 – Schrödinger : i ∂ψ/∂t = –½ ∇²ψ   (ħ = m = 1)
//       FX = Re(ψ),  FY = Im(ψ)
//       height = |ψ|,  hue = arg(ψ)
//
// Other controls:
//   Left-click+drag  – paint excitation (Re in Schrödinger mode)
//   Scroll           – change paint value
//   Shift+Scroll     – change brush radius
//   Right-click+drag – rotate view
//   Space            – pause / resume
//   R                – reset field
// =============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

static constexpr int   RES = 200;
static constexpr float DIFFUSION = 0.01f;

// =============================================================================
// Simulation mode  0=diffusion  1=wave  2=schrodinger
// =============================================================================
static int simulationMode = 0;

// =============================================================================
// Bitmap font  (4 wide × 6 tall, packed as 6 nibbles into uint32_t)
// Bit index = row*4 + col;  row 0 = top,  col 0 = left.
// =============================================================================
// clang-format off
static const uint32_t FONT[] = {
    // ── digits ──────────────────────────────────────────────────────────────
    0x6 | (0x9 << 4) | (0x9 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20),  //  0
    0x2 | (0x3 << 4) | (0x2 << 8) | (0x2 << 12) | (0x2 << 16) | (0xE << 20),  //  1
    0x7 | (0x8 << 4) | (0x6 << 8) | (0x1 << 12) | (0x1 << 16) | (0xF << 20),  //  2
    0x7 | (0x8 << 4) | (0x6 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20),  //  3
    0x9 | (0x9 << 4) | (0xF << 8) | (0x8 << 12) | (0x8 << 16) | (0x8 << 20),  //  4
    0xF | (0x1 << 4) | (0x7 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20),  //  5
    0x6 | (0x1 << 4) | (0x7 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20),  //  6
    0xF | (0x8 << 4) | (0x4 << 8) | (0x2 << 12) | (0x2 << 16) | (0x2 << 20),  //  7
    0x6 | (0x9 << 4) | (0x6 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20),  //  8
    0x6 | (0x9 << 4) | (0xE << 8) | (0x8 << 12) | (0x8 << 16) | (0x6 << 20),  //  9
    // ── punctuation / symbols ───────────────────────────────────────────────
    0x0 | (0x0 << 4) | (0x0 << 8) | (0x0 << 12) | (0x2 << 16) | (0x0 << 20),  // 10  '.'
    0xE | (0x1 << 4) | (0x6 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20),  // 11  'S'
    0x0,                                                    // 12  ' '
    // ── mode-label glyphs ────────────────────────────────────────────────────
    // D  (13)
    0x7 | (0x9 << 4) | (0x9 << 8) | (0x9 << 12) | (0x9 << 16) | (0x7 << 20),
    // I  (14)
    0xF | (0x6 << 4) | (0x6 << 8) | (0x6 << 12) | (0x6 << 16) | (0xF << 20),
    // F  (15)
    0xF | (0x1 << 4) | (0x7 << 8) | (0x1 << 12) | (0x1 << 16) | (0x1 << 20),
    // W  (16) — approximated in 4 px
    0x9 | (0x9 << 4) | (0xF << 8) | (0x6 << 12) | (0x9 << 16) | (0x9 << 20),
    // A  (17)
    0x6 | (0x9 << 4) | (0xF << 8) | (0x9 << 12) | (0x9 << 16) | (0x9 << 20),
    // V  (18)
    0x9 | (0x9 << 4) | (0x9 << 8) | (0x9 << 12) | (0x6 << 16) | (0x6 << 20),
    // C  (19)
    0xE | (0x1 << 4) | (0x1 << 8) | (0x1 << 12) | (0x1 << 16) | (0xE << 20),
    // H  (20)
    0x9 | (0x9 << 4) | (0xF << 8) | (0x9 << 12) | (0x9 << 16) | (0x9 << 20),
};
// clang-format on
static constexpr int GLYPH_DOT = 10;
static constexpr int GLYPH_S = 11;
static constexpr int GLYPH_SPACE = 12;
static constexpr int GLYPH_D = 13;
static constexpr int GLYPH_I = 14;
static constexpr int GLYPH_F = 15;
static constexpr int GLYPH_W = 16;
static constexpr int GLYPH_A = 17;
static constexpr int GLYPH_V = 18;
static constexpr int GLYPH_C = 19;
static constexpr int GLYPH_H = 20;
static constexpr int FONT_COUNT = 21;

// ── helpers ──────────────────────────────────────────────────────────────────
static int formatSimTime(float t, uint32_t* out, int maxOut)
{
    if (t < 0.0f) t = 0.0f;
    int intPart = static_cast<int>(t);
    int frac = static_cast<int>(t * 10.0f) % 10;

    uint32_t digits[8]; int nd = 0;
    if (intPart == 0) { digits[nd++] = 0; }
    else {
        int tmp = intPart;
        while (tmp > 0 && nd < 7) { digits[nd++] = tmp % 10; tmp /= 10; }
        for (int i = 0, j = nd - 1; i < j; ++i, --j) std::swap(digits[i], digits[j]);
    }
    int n = 0;
    for (int i = 0; i < nd && n < maxOut - 3; ++i) out[n++] = digits[i];
    if (n < maxOut) out[n++] = GLYPH_DOT;
    if (n < maxOut) out[n++] = static_cast<uint32_t>(frac);
    if (n < maxOut) out[n++] = GLYPH_S;
    return n;
}

static int formatModeLabel(int mode, uint32_t* out)
{
    if (mode == 0) { out[0] = GLYPH_D; out[1] = GLYPH_I; out[2] = GLYPH_F; }
    else if (mode == 1) { out[0] = GLYPH_W; out[1] = GLYPH_A; out[2] = GLYPH_V; }
    else { out[0] = GLYPH_S; out[1] = GLYPH_C; out[2] = GLYPH_H; }
    return 3;
}

// =============================================================================
// Compute shader
// =============================================================================
static const char* computeSrc = R"GLSL(
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

#define STRIDE 10
#define FX 0
#define FY 1
#define FZ 2
#define VX 3
#define VY 4
#define VZ 5
#define AX 6
#define AY 7
#define AZ 8

layout(std430, binding = 0) readonly  buffer BufIn  { float inData[];  };
layout(std430, binding = 1) writeonly buffer BufOut { float outData[]; };

uniform int   uRes;
uniform float uInvH2;
uniform float uDiffusion;
uniform float uDt;
uniform int   uMode;  // 0=diffusion  1=wave  2=schrod_re  3=schrod_im

float get(int cell, int comp) { return inData[cell * STRIDE + comp]; }

float laplacian(int r, int l, int u, int d, int c, int comp)
{
    return (get(r,comp)+get(l,comp)+get(u,comp)+get(d,comp) - 4.0*get(c,comp)) * uInvH2;
}

void main()
{
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    if (id.x >= uRes || id.y >= uRes) return;
    int c = id.y*uRes + id.x;
    int r = id.y*uRes + (id.x+1) % uRes;
    int l = id.y*uRes + (id.x-1+uRes) % uRes;
    int u = ((id.y+1) % uRes)*uRes + id.x;
    int d = ((id.y-1+uRes) % uRes)*uRes + id.x;
    int base = c * STRIDE;

    if (uMode == 0) {
        float vx = uDiffusion * laplacian(r,l,u,d,c,FX);
        float vy = uDiffusion * laplacian(r,l,u,d,c,FY);
        float vz = uDiffusion * laplacian(r,l,u,d,c,FZ);
        outData[base+FX] = get(c,FX) + vx*uDt;
        outData[base+FY] = get(c,FY) + vy*uDt;
        outData[base+FZ] = get(c,FZ) + vz*uDt;
        outData[base+VX] = vx; outData[base+VY] = vy; outData[base+VZ] = vz;
        outData[base+AX] = 0;  outData[base+AY] = 0;  outData[base+AZ] = 0;
        outData[base+9]  = 0.0;
    }
    else if (uMode == 1) {
        float ax = uDiffusion * laplacian(r,l,u,d,c,FX);
        float ay = uDiffusion * laplacian(r,l,u,d,c,FX);
        float az = uDiffusion * laplacian(r,l,u,d,c,FZ);
        float vx = get(c,VX)*0.999 + ax * uDt;
        float vy = get(c,VY) + ay * uDt;
        float vz = get(c,VZ) + az * uDt;
        outData[base+FX] = get(c,FX) + vx*uDt;
        outData[base+FY] = get(c,FY) + vy*uDt;
        outData[base+FZ] = get(c,FZ) + vz*uDt;
        outData[base+VX] = vx; outData[base+VY] = vy; outData[base+VZ] = vz;
        outData[base+AX] = ax; outData[base+AY] = ay;  outData[base+AZ] = az;
        outData[base+9]  = 0.0;
    }
    else if (uMode == 2) {
        float lapIm = uDiffusion * laplacian(r,l,u,d,c,FY);
        outData[base+FX] = get(c,FX) - 0.5 * lapIm * uDt;
        outData[base+FY] = get(c,FY);
        outData[base+FZ] = get(c,FZ);
        outData[base+VX] = 0; outData[base+VY] = 0; outData[base+VZ] = 0;
        outData[base+AX] = 0; outData[base+AY] = 0; outData[base+AZ] = 0;
        outData[base+9]  = 0.0;
    }
    else {
        float lapRe = uDiffusion * laplacian(r,l,u,d,c,FX);
        outData[base+FX] = get(c,FX);
        outData[base+FY] = get(c,FY) + 0.5 * lapRe * uDt;
        outData[base+FZ] = get(c,FZ);
        outData[base+VX] = 0; outData[base+VY] = 0; outData[base+VZ] = 0;
        outData[base+AX] = 0; outData[base+AY] = 0; outData[base+AZ] = 0;
        outData[base+9]  = 0.0;
    }
}
)GLSL";

// =============================================================================
// Field render shaders — with normal-based Lambert lighting
// =============================================================================
static const char* fieldVertSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in uint aIdx;
#define STRIDE 10
layout(std430, binding = 0) readonly buffer Field { float field[]; };
uniform mat4 uRotation;
uniform int  uMode;
uniform int  uRes;
out vec3 vColor;

vec3 hueToRgb(float h)
{
    float hp = h * 6.0;
    float xc = 1.0 - abs(mod(hp, 2.0) - 1.0);
    int   s  = int(hp) % 6;
    if (s < 0) s += 6;
    if      (s==0) return vec3(1,  xc, 0 );
    else if (s==1) return vec3(xc, 1,  0 );
    else if (s==2) return vec3(0,  1,  xc);
    else if (s==3) return vec3(0,  xc, 1 );
    else if (s==4) return vec3(xc, 0,  1 );
    else           return vec3(1,  0,  xc);
}

// Returns the display height of any cell index (clamped to grid).
float cellHeight(int idx)
{
    idx = clamp(idx, 0, uRes*uRes - 1);
    if (uMode == 2) {
        float re = field[idx*STRIDE + 0];
        float im = field[idx*STRIDE + 1];
        return atan(sqrt(re*re + im*im) * 0.02) / 3.14159265;
    } else {
        return atan(field[idx*STRIDE] * 0.02) / 3.14159265;
    }
}

void main()
{
    int id = int(aIdx);
    int ix = id % uRes;
    int iy = id / uRes;

    // ── Neighbour heights for surface normal estimation ──────────────────────
    float hr = cellHeight(iy*uRes + min(ix+1, uRes-1));
    float hl = cellHeight(iy*uRes + max(ix-1, 0      ));
    float hu = cellHeight(min(iy+1, uRes-1)*uRes + ix );
    float hd = cellHeight(max(iy-1, 0      )*uRes + ix );

    // zScale amplifies the normal tilt so shading is visible even on gentle slopes.
    float zScale = 3.0;
    vec3 tx = normalize(vec3(2.0, 0.0, (hr - hl) * zScale));
    vec3 ty = normalize(vec3(0.0, 2.0, (hu - hd) * zScale));
    vec3 N  = normalize(cross(tx, ty));

    // ── Position & hue ───────────────────────────────────────────────────────
    float height;
    float hue;

    if (uMode == 2) {
        float re  = field[id*STRIDE + 0];
        float im  = field[id*STRIDE + 1];
        float mag = sqrt(re*re + im*im);
        height = atan(mag * 0.02) / 3.14159265 - 0.25;
        float angle = atan(im, re);
        hue = angle / (2.0 * 3.14159265) + 0.5;
    } else {
        float fx = field[id*STRIDE];
        height  = atan( fx*0.02) / 3.14159265 - 0.5;
        hue     = atan(-fx*0.02) / 3.14159265 + 0.5;
    }

    gl_Position = uRotation * vec4(aPos, height, 1.0);

    // ── Lambert lighting — fixed light in world space (upper-front-left) ─────
    vec3 lightDir = normalize(vec3(-0.4, 0.6, 1.0));
    float ambient = 0.25;
    float diffuse = abs(dot(N, lightDir));   // abs = lit from both sides
    float light   = ambient + (1.0 - ambient) * diffuse;

    vColor = hueToRgb(hue) * light;
}
)GLSL";

static const char* fieldFragSrc = R"GLSL(
#version 430 core
in  vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }
)GLSL";

// =============================================================================
// Text overlay shaders
// =============================================================================
static const char* textVertSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;

uniform vec2  uOrigin;
uniform vec2  uCharSize;
uniform float uAdvance;
uniform uint  uFont[21];
uniform uint  uChars[16];

out vec2 vUV;
flat out uint vGlyph;

void main()
{
    vec2 cellOrigin = uOrigin + vec2(float(gl_InstanceID) * uAdvance, 0.0);
    vec2 ndcPos = cellOrigin + vec2(aPos.x * uCharSize.x, -aPos.y * uCharSize.y);
    gl_Position = vec4(ndcPos, 0.0, 1.0);
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
    int col = clamp(int(vUV.x * 4.0), 0, 3);
    int row = clamp(int(vUV.y * 6.0), 0, 5);
    if (((vGlyph >> uint(row*4 + col)) & 1u) == 0u) discard;
    FragColor = uTextColor;
}
)GLSL";

// =============================================================================
// Solid 2-D rect shader
// =============================================================================
static const char* rectVertSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
uniform vec2 uRectOrigin;
uniform vec2 uRectSize;
void main()
{
    vec2 p = uRectOrigin + vec2(aPos.x * uRectSize.x, -aPos.y * uRectSize.y);
    gl_Position = vec4(p, 0.0, 1.0);
}
)GLSL";

static const char* rectFragSrc = R"GLSL(
#version 430 core
uniform vec4 uRectColor;
out vec4 FragColor;
void main() { FragColor = uRectColor; }
)GLSL";

// =============================================================================
// Global input state
// =============================================================================
struct GlobeState {
    bool  dragging = false;
    double lastX = 0, lastY = 0;
    float spin = 0.0f;   // rotation around field normal (Z) — left/right drag
    float pitch = 0.5f;   // tilt toward viewer — 0=top-down, π=bottom-up
    static constexpr float SENSITIVITY = 0.008f;  // slightly faster drag
};
static GlobeState rot;

struct HeatState { bool active = false; double x = 0, y = 0; };
static HeatState heat;

static float heatValue = 100.0f;
static int   heatRadius = 5;
static float zoom = 2.5f;          // camera distance — scroll to change
static float cameraY = 0.0f;          // vertical pan — shift+scroll
static bool  paused = false;
static bool  resetRequested = false;

// =============================================================================
// Callbacks
// =============================================================================
static void keyCallback(GLFWwindow*, int key, int, int action, int)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        paused = !paused;
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
        resetRequested = true;

    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_1) {
            simulationMode = 0; resetRequested = true;
            std::cout << "Mode: Diffusion  (1)\n";
        }
        if (key == GLFW_KEY_2) {
            simulationMode = 1; resetRequested = true;
            std::cout << "Mode: Wave  (2)\n";
        }
        if (key == GLFW_KEY_3) {
            simulationMode = 2; resetRequested = true;
            std::cout << "Mode: Schrodinger  (3)\n";
        }
    }
}

static void scrollCallback(GLFWwindow* w, double, double yoff)
{
    bool shift = (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
    bool ctrl = (glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
        glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
    bool alt = (glfwGetKey(w, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
        glfwGetKey(w, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS);

    if (shift) {
        cameraY = std::clamp(cameraY + (float)yoff * 0.05f, -5.0f, 5.0f);
    }
    else if (alt) {
        heatRadius = std::clamp(heatRadius + (yoff > 0 ? 1 : -1), 1, 64);
        std::cout << "Brush radius: " << heatRadius << "\n";
    }
    else if (ctrl) {
        heatValue = std::clamp(heatValue + (float)yoff * 10.0f, 1.0f, 5000.0f);
        std::cout << "Paint value: " << heatValue << "\n";
    }
    else {
        zoom = std::clamp(zoom - (float)yoff * 0.15f, 0.5f, 10.0f);
    }
}

static void mouseButtonCallback(GLFWwindow* w, int btn, int action, int)
{
    if (btn == GLFW_MOUSE_BUTTON_RIGHT) {
        rot.dragging = (action == GLFW_PRESS);
        if (rot.dragging)
            glfwGetCursorPos(w, &rot.lastX, &rot.lastY);
    }
    if (btn == GLFW_MOUSE_BUTTON_LEFT)
        heat.active = (action == GLFW_PRESS);
}

static void cursorPosCallback(GLFWwindow*, double x, double y)
{
    if (rot.dragging) {
        float dx = (float)(x - rot.lastX);
        float dy = (float)(y - rot.lastY);
        rot.spin -= dx * GlobeState::SENSITIVITY;
        rot.pitch += dy * GlobeState::SENSITIVITY;
        // 0 = looking straight down (north pole), π-ε = looking straight up (south pole)
        rot.pitch = std::clamp(rot.pitch, 0.0f, 3.14159265f - 0.0001f);
        rot.lastX = x; rot.lastY = y;
    }
    heat.x = x; heat.y = -y + 800;
}

// Generic column-major 4x4 inverse (Cramer's rule).
static void mat4Inverse(const float* m, float* out)
{
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (fabsf(det) < 1e-8f) { memcpy(out, m, 64); return; }
    det = 1.0f / det;
    for (int i = 0; i < 16; i++) out[i] = inv[i] * det;
}

// Unproject screen position (OpenGL convention: y=0 at bottom) through MVP
// to the z=0 world plane (the field surface). Returns false if ray misses.
static bool unprojectToField(float sx, float sy, const float* MVP,
    float& worldX, float& worldY)
{
    float invMVP[16];
    mat4Inverse(MVP, invMVP);

    float nx = 2.0f * sx / 800.0f - 1.0f;
    float ny = 2.0f * sy / 800.0f - 1.0f;

    // Unproject clip-space point at near (-1) and far (+1) depths to world.
    auto unproj = [&](float nz, float out[3]) {
        float clip[4] = { nx, ny, nz, 1.0f };
        float w[4] = {};
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                w[r] += invMVP[c * 4 + r] * clip[c];
        float invW = (fabsf(w[3]) > 1e-8f) ? 1.0f / w[3] : 1.0f;
        out[0] = w[0] * invW; out[1] = w[1] * invW; out[2] = w[2] * invW;
        };

    float nearPt[3], farPt[3];
    unproj(-1.0f, nearPt);
    unproj(1.0f, farPt);

    // Intersect ray with z=0 plane.
    float dz = farPt[2] - nearPt[2];
    if (fabsf(dz) < 1e-6f) return false;
    float t = -nearPt[2] / dz;

    worldX = nearPt[0] + t * (farPt[0] - nearPt[0]);
    worldY = nearPt[1] + t * (farPt[1] - nearPt[1]);
    return true;
}

// Turntable rotation: Rx(pitch) * Rz(spin)
//   spin  — left/right drag, rotates around the field normal (Z)
//   pitch — up/down drag, tilts the field toward/away from viewer
//   North (field +Y) always stays pointing up on screen at any spin angle.
static void buildGlobeMatrix(float spin, float pitch, float* m)
{
    float cs = cosf(spin), ss = sinf(spin);
    float cp = cosf(pitch), sp = sinf(pitch);
    // Column-major Rx(pitch) * Rz(spin)
    m[0] = cs;      m[1] = -ss * cp;   m[2] = ss * sp;   m[3] = 0;
    m[4] = ss;      m[5] = cs * cp;   m[6] = -cs * sp;   m[7] = 0;
    m[8] = 0;       m[9] = sp;      m[10] = cp;     m[11] = 0;
    m[12] = 0;       m[13] = 0;       m[14] = 0;       m[15] = 1;
}

// Column-major 4x4 multiply:  C = A * B
static void matMul(const float* A, const float* B, float* C)
{
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += A[k * 4 + row] * B[col * 4 + k];
            C[col * 4 + row] = s;
        }
}

// Standard OpenGL perspective matrix (column-major)
static void buildPerspective(float fovY, float aspect, float near, float far, float* m)
{
    float f = 1.0f / tanf(fovY * 0.5f);
    memset(m, 0, 64);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) / (near - far);
    m[11] = -1.0f;
    m[14] = 2.0f * far * near / (near - far);
}

// Translation matrix (column-major)
static void buildTranslation(float tx, float ty, float tz, float* m)
{
    memset(m, 0, 64);
    m[0] = m[5] = m[10] = m[15] = 1.0f;
    m[12] = tx; m[13] = ty; m[14] = tz;
}

// =============================================================================
// Shader helpers
// =============================================================================
static unsigned int compileShader(GLenum type, const char* src)
{
    unsigned int s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    int ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(s, 1024, nullptr, log); std::cerr << log << "\n"; }
    return s;
}

static unsigned int makeProgram(std::initializer_list<unsigned int> shaders)
{
    unsigned int p = glCreateProgram();
    for (auto s : shaders) glAttachShader(p, s);
    glLinkProgram(p);
    int ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, 1024, nullptr, log); std::cerr << log << "\n"; }
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

    GLFWwindow* window = glfwCreateWindow(800, 800,
        "Field Simulation (GPU) — 1:Diffusion  2:Wave  3:Schrodinger", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n"; return -1;
    }
    glViewport(0, 0, 800, 800);
    glEnable(GL_DEPTH_TEST);

    // ── Programs ──────────────────────────────────────────────────────────────
    unsigned int computeProg = makeProgram({ compileShader(GL_COMPUTE_SHADER,  computeSrc) });
    unsigned int fieldProg = makeProgram({ compileShader(GL_VERTEX_SHADER,   fieldVertSrc),
                                             compileShader(GL_FRAGMENT_SHADER, fieldFragSrc) });
    unsigned int textProg = makeProgram({ compileShader(GL_VERTEX_SHADER,   textVertSrc),
                                             compileShader(GL_FRAGMENT_SHADER, textFragSrc) });
    unsigned int rectProg = makeProgram({ compileShader(GL_VERTEX_SHADER,   rectVertSrc),
                                             compileShader(GL_FRAGMENT_SHADER, rectFragSrc) });

    // ── Uniform locations ─────────────────────────────────────────────────────
    int uResU = glGetUniformLocation(computeProg, "uRes");
    int uInvH2U = glGetUniformLocation(computeProg, "uInvH2");
    int uDiffusionU = glGetUniformLocation(computeProg, "uDiffusion");
    int uDtU = glGetUniformLocation(computeProg, "uDt");
    int uComputeModeU = glGetUniformLocation(computeProg, "uMode");

    int uRotationU = glGetUniformLocation(fieldProg, "uRotation");
    int uFieldModeU = glGetUniformLocation(fieldProg, "uMode");
    int uFieldResU = glGetUniformLocation(fieldProg, "uRes");   // ← new

    int uTxtOrigin = glGetUniformLocation(textProg, "uOrigin");
    int uTxtSize = glGetUniformLocation(textProg, "uCharSize");
    int uTxtAdvance = glGetUniformLocation(textProg, "uAdvance");
    int uTxtFont = glGetUniformLocation(textProg, "uFont");
    int uTxtChars = glGetUniformLocation(textProg, "uChars");
    int uTxtColor = glGetUniformLocation(textProg, "uTextColor");

    int uRectOrigin = glGetUniformLocation(rectProg, "uRectOrigin");
    int uRectSize = glGetUniformLocation(rectProg, "uRectSize");
    int uRectColor = glGetUniformLocation(rectProg, "uRectColor");

    // ── Grid geometry ─────────────────────────────────────────────────────────
    const int   N = RES;
    const float h = (N > 1) ? (2.0f / (N - 1)) : 1.0f;
    const float invH2 = 1.0f / (h * h);

    const float SIM_SPEED = 1.0f;
    const float TARGET_STEP_DT = 1.0f / 120.0f;

    const float subDtDiff = std::min((h * h) / (4.0f * DIFFUSION) * 0.9f, TARGET_STEP_DT);
    const float subDtWave = std::min((h / (std::sqrt(DIFFUSION) * std::sqrt(2.0f))) * 0.9f, TARGET_STEP_DT);
    const float subDtSchrod = 0.45f * h * h;

    std::cout << "Resolution: " << N << "x" << N
        << "  dt=" << subDtSchrod
        << "\nControls:\n"
        << "  1/2/3             — Diffusion / Wave / Schrodinger mode\n"
        << "  Left-click+drag   — paint excitation\n"
        << "  Scroll            — zoom in / out\n"
        << "  Shift+Scroll      — camera up / down\n"
        << "  Alt+Scroll        — change brush radius (cur: " << heatRadius << ")\n"
        << "  Ctrl+Scroll       — change paint value (cur: " << heatValue << ")\n"
        << "  Right-click+drag  — rotate view\n"
        << "  Space             — pause / resume\n"
        << "  R                 — reset field\n";

    // ── Field SSBOs ───────────────────────────────────────────────────────────
    const int STRIDE = 10;
    std::vector<float> zeroBuf(N * N * STRIDE, 0.0f);
    unsigned int ssbo[2];
    glGenBuffers(2, ssbo);
    for (int b = 0; b < 2; ++b) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[b]);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
            zeroBuf.size() * sizeof(float),
            b == 0 ? zeroBuf.data() : nullptr,
            GL_DYNAMIC_COPY);
    }

    // ── Field mesh ────────────────────────────────────────────────────────────
    struct Vert { float x, y; uint32_t idx; };
    std::vector<Vert> mesh;
    mesh.reserve((N - 1) * (N - 1) * 6);
    std::vector<float> pos(N);
    for (int k = 0; k < N; ++k) pos[k] = -1.0f + k * h;
    for (int i = 0; i < N - 1; ++i)
        for (int j = 0; j < N - 1; ++j) {
            uint32_t bl = i * N + j, br = i * N + (j + 1), tl = (i + 1) * N + j, tr = (i + 1) * N + (j + 1);
            mesh.push_back({ pos[j],   pos[i],   bl });
            mesh.push_back({ pos[j + 1], pos[i],   br });
            mesh.push_back({ pos[j],   pos[i + 1], tl });
            mesh.push_back({ pos[j + 1], pos[i],   br });
            mesh.push_back({ pos[j + 1], pos[i + 1], tr });
            mesh.push_back({ pos[j],   pos[i + 1], tl });
        }
    unsigned int fieldVAO, fieldVBO;
    glGenVertexArrays(1, &fieldVAO); glGenBuffers(1, &fieldVBO);
    glBindVertexArray(fieldVAO);
    glBindBuffer(GL_ARRAY_BUFFER, fieldVBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.size() * sizeof(Vert), mesh.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vert), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, sizeof(Vert), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // ── Shared unit-quad VAO ──────────────────────────────────────────────────
    float quadVerts[] = {
        0,0, 1,0, 0,1,
        1,0, 1,1, 0,1
    };
    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // ── UI layout constants ───────────────────────────────────────────────────
    const float PX = 2.0f / 800.0f;
    const float CHAR_W = 20.0f * PX;
    const float CHAR_H = 30.0f * PX;
    const float CHAR_ADV = 22.0f * PX;
    const float MARGIN = 12.0f * PX;
    const float PAD = 6.0f * PX;
    const float TEXT_X = -1.0f + MARGIN;
    const float TEXT_Y = 1.0f - MARGIN;

    const int groups = (N + 15) / 16;
    int       current = 0;
    float     simTime = 0.0f;
    double    lastTime = glfwGetTime();
    float     accumDiff = 0.0f;
    float     accumWave = 0.0f;
    float     accumSchrod = 0.0f;

    // ── Main loop ─────────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(window))
    {
        // ── Wall-clock delta time ─────────────────────────────────────────────
        double nowTime = glfwGetTime();
        float  realDt = (float)(nowTime - lastTime);
        lastTime = nowTime;
        float simBudget = std::min(realDt, 4.0f / 60.0f) * SIM_SPEED;

        // ── Reset ─────────────────────────────────────────────────────────────
        if (resetRequested) {
            resetRequested = false;
            simTime = 0.0f;
            current = 0;
            for (int b = 0; b < 2; ++b) {
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[b]);
                glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F,
                    GL_RED, GL_FLOAT, nullptr);
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        // ── Build MVP early — needed for brush unprojection and rendering ────
        float R[16], T[16], P[16], TR[16], MVP[16];
        buildGlobeMatrix(rot.spin, rot.pitch, R);
        buildTranslation(0.0f, cameraY, -zoom, T);
        buildPerspective(3.14159265f / 3.0f, 1.0f, 0.01f, 100.0f, P);
        matMul(T, R, TR);
        matMul(P, TR, MVP);

        // ── Paint excitation ──────────────────────────────────────────────────
        if (heat.active) {
            // heat.y is already in OpenGL convention (0=bottom, 800=top)
            float worldX, worldY;
            int cx = N / 2, cy = N / 2;   // fallback: field centre
            if (unprojectToField((float)heat.x, (float)heat.y, MVP, worldX, worldY)) {
                // field spans [-1,+1] in world XY → map to grid indices
                cx = (int)((worldX + 1.0f) * 0.5f * N);
                cy = (int)((worldY + 1.0f) * 0.5f * N);
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[current]);
            for (int dy = -heatRadius; dy <= heatRadius; ++dy)
                for (int dx = -heatRadius; dx <= heatRadius; ++dx) {
                    if (dx * dx + dy * dy > heatRadius * heatRadius) continue;
                    int gi = cy + dy, gj = cx + dx;
                    if (gi < 0 || gi >= N || gj < 0 || gj >= N) continue;
                    GLintptr off = (GLintptr)((gi * N + gj) * STRIDE) * sizeof(float);
                    glBufferSubData(GL_SHADER_STORAGE_BUFFER, off, sizeof(float), &heatValue);
                }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        // ── Compute pass ──────────────────────────────────────────────────────
        if (!paused) {
            glUseProgram(computeProg);
            glUniform1i(uResU, N);
            glUniform1f(uInvH2U, invH2);
            glUniform1f(uDiffusionU, DIFFUSION);

            if (simulationMode == 0) {
                accumDiff += simBudget;
                glUniform1i(uComputeModeU, 0);
                glUniform1f(uDtU, subDtDiff);
                while (accumDiff >= subDtDiff) {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
                    glDispatchCompute(groups, groups, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    current = 1 - current;
                    accumDiff -= subDtDiff;
                    simTime += subDtDiff;
                }
            }
            else if (simulationMode == 1) {
                accumWave += simBudget;
                glUniform1i(uComputeModeU, 1);
                glUniform1f(uDtU, subDtWave);
                while (accumWave >= subDtWave) {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
                    glDispatchCompute(groups, groups, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    current = 1 - current;
                    accumWave -= subDtWave;
                    simTime += subDtWave;
                }
            }
            else {
                accumSchrod += simBudget;
                glUniform1f(uDtU, subDtSchrod);
                while (accumSchrod >= subDtSchrod) {
                    // Pass A: update Re from Im
                    glUniform1i(uComputeModeU, 2);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
                    glDispatchCompute(groups, groups, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    current = 1 - current;
                    // Pass B: update Im from (updated) Re
                    glUniform1i(uComputeModeU, 3);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
                    glDispatchCompute(groups, groups, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    current = 1 - current;
                    accumSchrod -= subDtSchrod;
                    simTime += subDtSchrod;
                }
            }
        }

        // ── Field render ──────────────────────────────────────────────────────
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(fieldProg);
        glUniformMatrix4fv(uRotationU, 1, GL_FALSE, MVP);
        glUniform1i(uFieldModeU, simulationMode);
        glUniform1i(uFieldResU, N);                  // ← pass resolution
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
        glBindVertexArray(fieldVAO);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)mesh.size());

        // ── 2-D overlay ───────────────────────────────────────────────────────
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // — Timer —
        uint32_t charBuf[16];
        int numChars = formatSimTime(simTime, charBuf, 16);
        float textW = CHAR_W + (numChars - 1) * CHAR_ADV;

        glUseProgram(rectProg);
        glUniform2f(uRectOrigin, TEXT_X - PAD, TEXT_Y + PAD);
        glUniform2f(uRectSize, textW + 2 * PAD, CHAR_H + 2 * PAD);
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

        // — Mode label —
        uint32_t modeBuf[3];
        int numMode = formatModeLabel(simulationMode, modeBuf);
        float modeW = CHAR_W + (numMode - 1) * CHAR_ADV;
        float modeY = TEXT_Y - CHAR_H - PAD - MARGIN;

        glUseProgram(rectProg);
        glUniform2f(uRectOrigin, TEXT_X - PAD, modeY + PAD);
        glUniform2f(uRectSize, modeW + 2 * PAD, CHAR_H + 2 * PAD);
        glUniform4f(uRectColor, 0, 0, 0, 0.65f);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glUseProgram(textProg);
        glUniform2f(uTxtOrigin, TEXT_X, modeY);
        glUniform2f(uTxtSize, CHAR_W, CHAR_H);
        glUniform1f(uTxtAdvance, CHAR_ADV);
        glUniform1uiv(uTxtFont, FONT_COUNT, FONT);
        glUniform1uiv(uTxtChars, numMode, modeBuf);
        if (simulationMode == 0) glUniform4f(uTxtColor, 1.0f, 0.65f, 0.0f, 1.0f);
        else if (simulationMode == 1) glUniform4f(uTxtColor, 0.0f, 1.0f, 1.0f, 1.0f);
        else                          glUniform4f(uTxtColor, 1.0f, 0.3f, 1.0f, 1.0f);
        glBindVertexArray(quadVAO);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, numMode);

        glDisable(GL_BLEND);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteBuffers(2, ssbo);
    glDeleteBuffers(1, &fieldVBO);  glDeleteVertexArrays(1, &fieldVAO);
    glDeleteBuffers(1, &quadVBO);   glDeleteVertexArrays(1, &quadVAO);
    glDeleteProgram(computeProg);   glDeleteProgram(fieldProg);
    glDeleteProgram(textProg);      glDeleteProgram(rectProg);
    glfwTerminate();
    return 0;
}