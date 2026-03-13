// =============================================================================
// Field Simulation — GPU Compute Edition
// Requires OpenGL 4.3 (compute shaders + SSBOs)
// =============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

static constexpr int   RES = 256;
static constexpr float DIFFUSION = 0.001f;

// =============================================================================
// Bitmap font  (4 wide × 6 tall = 24 bits per glyph, packed into uint32_t)
//
// Bit layout: bit = row*4 + col   (row 0 = top, col 0 = left)
// Nibble N covers row N (bits N*4 .. N*4+3).
//
// To read: given (col, row) → bit = row*4+col → (glyph >> bit) & 1
//
// Characters and their indices:
//   0-9  digits
//   10   '.'
//   11   's'
//   12   ' '
//
// Encoding helper — for each row write the 4-bit pattern as a hex nibble,
// then pack: glyph = row0 | (row1<<4) | (row2<<8) | (row3<<12) | (row4<<16) | (row5<<20)
//
// Pixel key: col0=LSB, col3=MSB   e.g. .##. = 0110 = 6
//   .... = 0x0   #... = 0x1   .#.. = 0x2   ##.. = 0x3
//   ..#. = 0x4   #.#. = 0x5   .##. = 0x6   ###. = 0x7
//   ...# = 0x8   #..# = 0x9   .#.# = 0xA   ##.# = 0xB
//   ..## = 0xC   #.## = 0xD   .### = 0xE   #### = 0xF
// =============================================================================
// clang-format off
//                      row0  row1  row2  row3  row4  row5
static const uint32_t FONT[] = {
    // '0'  .##. #..# #..# #..# #..# .##.
    0x6 | (0x9 << 4) | (0x9 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20),  // 0
    // '1'  .#.. ##.. .#.. .#.. .#.. .###
    0x2 | (0x3 << 4) | (0x2 << 8) | (0x2 << 12) | (0x2 << 16) | (0xE << 20),  // 1
    // '2'  ###. ...# .##. #... #... ####
    0x7 | (0x8 << 4) | (0x6 << 8) | (0x1 << 12) | (0x1 << 16) | (0xF << 20),  // 2
    // '3'  ###. ...# .##. ...# ...# ###.
    0x7 | (0x8 << 4) | (0x6 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20),  // 3
    // '4'  #..# #..# #### ...# ...# ...#
    0x9 | (0x9 << 4) | (0xF << 8) | (0x8 << 12) | (0x8 << 16) | (0x8 << 20),  // 4
    // '5'  #### #... ###. ...# ...# ###.
    0xF | (0x1 << 4) | (0x7 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20),  // 5
    // '6'  .##. #... ###. #..# #..# .##.
    0x6 | (0x1 << 4) | (0x7 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20),  // 6
    // '7'  #### ...# ..#. .#.. .#.. .#..
    0xF | (0x8 << 4) | (0x4 << 8) | (0x2 << 12) | (0x2 << 16) | (0x2 << 20),  // 7
    // '8'  .##. #..# .##. #..# #..# .##.
    0x6 | (0x9 << 4) | (0x6 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20),  // 8
    // '9'  .##. #..# .### ...# ...# .##.
    0x6 | (0x9 << 4) | (0xE << 8) | (0x8 << 12) | (0x8 << 16) | (0x6 << 20),  // 9
    // '.'  .... .... .... .... .#.. ....
    0x0 | (0x0 << 4) | (0x0 << 8) | (0x0 << 12) | (0x2 << 16) | (0x0 << 20),  // 10
    // 's'  .### #... .##. ...# ...# ###.
    0xE | (0x1 << 4) | (0x6 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20),  // 11
    // ' '  (space)
    0x0,                                                               // 12
};
// clang-format on
static constexpr int GLYPH_DOT = 10;
static constexpr int GLYPH_S = 11;
static constexpr int GLYPH_SPACE = 12;
static constexpr int FONT_COUNT = 13;

// Converts sim time (seconds) into a glyph-index sequence, e.g. "42.7s"
// Returns the number of characters written into out[].
static int formatSimTime(float t, uint32_t* out, int maxOut)
{
    if (t < 0.0f) t = 0.0f;
    int intPart = static_cast<int>(t);
    int frac = static_cast<int>(t * 10.0f) % 10;

    // collect integer digits, most-significant-first
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

// =============================================================================
// Compute shader — full 3-component diffusion, CFL-stable dt from CPU
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

    float ax = 0;
    float ay = 0;
    float az = 0;

    float vx = uDiffusion * laplacian(r,l,u,d,c,FX);
    float vy = uDiffusion * laplacian(r,l,u,d,c,FY);
    float vz = uDiffusion * laplacian(r,l,u,d,c,FZ);

    // Velocity accumulates from acceleration (symplectic Euler)
    // float vx = get(c,VX) + ax * uDt;
    // float vy = get(c,VY) + ay * uDt;
    // float vz = get(c,VZ) + az * uDt;

    int base = c * STRIDE;
    outData[base+FX] = get(c,FX) + vx*uDt;
    outData[base+FY] = get(c,FY) + vy*uDt;
    outData[base+FZ] = get(c,FZ) + vz*uDt;
    outData[base+VX] = vx;  outData[base+VY] = vy;  outData[base+VZ] = vz;
    outData[base+AX] = ax;  outData[base+AY] = ay;  outData[base+AZ] = az;
    outData[base+9]  = 0.0;
}
)GLSL";

// =============================================================================
// Field render shaders
// =============================================================================
static const char* fieldVertSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in uint aIdx;
#define STRIDE 10
layout(std430, binding = 0) readonly buffer Field { float field[]; };
uniform mat4 uRotation;
out vec3 vColor;

vec3 fieldToColor(float f)
{
    float h  = atan(f*0.02)/3.14159265 + 0.5;
    float hp = h * 6.0;
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
    float fx      = field[int(aIdx)*STRIDE];
    float hueNorm = atan(fx*0.02)/3.14159265 + 0.5;
    vColor        = fieldToColor(fx);
    gl_Position   = uRotation * vec4(aPos, hueNorm - 0.5, 1.0);
}
)GLSL";

static const char* fieldFragSrc = R"GLSL(
#version 430 core
in  vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }
)GLSL";

// =============================================================================
// Text overlay shaders  (instanced bitmap font)
//
// One instance = one character.
// gl_InstanceID selects the character position in the string.
// Each character is a unit quad [0,1]×[0,1] stretched to NDC.
// The fragment shader looks up the glyph bitmap and discards unlit pixels.
// =============================================================================
static const char* textVertSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;     // [0,1]x[0,1] unit quad

uniform vec2  uOrigin;      // NDC position of the top-left of the first char
uniform vec2  uCharSize;    // NDC width, height of one character cell
uniform float uAdvance;     // NDC x-step between character cells
uniform uint  uFont[13];    // glyph bitmaps, indexed by glyph id
uniform uint  uChars[16];   // glyph id for each character in the string

out vec2 vUV;
flat out uint vGlyph;

void main()
{
    // Move the quad to the correct character cell
    vec2 cellOrigin = uOrigin + vec2(float(gl_InstanceID) * uAdvance, 0.0);
    // NDC: x increases right, y increases up — so subtract v to go downward
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
    // 4-wide x 6-tall font. bit = row*4 + col, col 0 = leftmost.
    int col = clamp(int(vUV.x * 4.0), 0, 3);
    int row = clamp(int(vUV.y * 6.0), 0, 5);
    if (((vGlyph >> uint(row*4 + col)) & 1u) == 0u) discard;
    FragColor = uTextColor;
}
)GLSL";

// =============================================================================
// Solid 2D rect shader  (used for the dark background behind the timer text)
// =============================================================================
static const char* rectVertSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;     // [0,1]x[0,1]
uniform vec2 uRectOrigin;   // NDC top-left of rect
uniform vec2 uRectSize;     // NDC width, height (positive)
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
// Rotation state + callbacks
// =============================================================================
struct RotationState {
    bool dragging = false;
    double lastX = 0, lastY = 0;
    float yaw = 0, pitch = 0, sensitivity = 0.005f;
};
static RotationState rot;

static void mouseButtonCallback(GLFWwindow* w, int button, int action, int)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        rot.dragging = (action == GLFW_PRESS);
        if (rot.dragging) glfwGetCursorPos(w, &rot.lastX, &rot.lastY);
    }
}
static void cursorPosCallback(GLFWwindow*, double x, double y)
{
    if (!rot.dragging) return;
    rot.yaw += float(x - rot.lastX) * rot.sensitivity;
    rot.pitch += float(y - rot.lastY) * rot.sensitivity;
    rot.pitch = std::clamp(rot.pitch, -1.5707963f, 1.5707963f);
    rot.lastX = x; rot.lastY = y;
}
static void buildRotationMatrix(float yaw, float pitch, float* m)
{
    float cy = cosf(yaw), sy = sinf(yaw), cx = cosf(pitch), sx = sinf(pitch);
    m[0] = cy;    m[1] = 0;   m[2] = -sy;    m[3] = 0;
    m[4] = sy * sx; m[5] = cx;  m[6] = cy * sx;  m[7] = 0;
    m[8] = sy * cx; m[9] = -sx; m[10] = cy * cx; m[11] = 0;
    m[12] = 0;    m[13] = 0;  m[14] = 0;     m[15] = 1;
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

    GLFWwindow* window = glfwCreateWindow(800, 800, "Field Simulation (GPU)", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n"; return -1;
    }
    glViewport(0, 0, 800, 800);
    glEnable(GL_DEPTH_TEST);

    // ── Compile all programs ──────────────────────────────────────────────────
    unsigned int computeProg = makeProgram({ compileShader(GL_COMPUTE_SHADER,  computeSrc) });
    unsigned int fieldProg = makeProgram({ compileShader(GL_VERTEX_SHADER,   fieldVertSrc),
                                             compileShader(GL_FRAGMENT_SHADER, fieldFragSrc) });
    unsigned int textProg = makeProgram({ compileShader(GL_VERTEX_SHADER,   textVertSrc),
                                             compileShader(GL_FRAGMENT_SHADER, textFragSrc) });
    unsigned int rectProg = makeProgram({ compileShader(GL_VERTEX_SHADER,   rectVertSrc),
                                             compileShader(GL_FRAGMENT_SHADER, rectFragSrc) });

    // Cache uniform locations
    int uRes = glGetUniformLocation(computeProg, "uRes");
    int uInvH2u = glGetUniformLocation(computeProg, "uInvH2");
    int uDiffusion = glGetUniformLocation(computeProg, "uDiffusion");
    int uDt = glGetUniformLocation(computeProg, "uDt");
    int uRotation = glGetUniformLocation(fieldProg, "uRotation");
    int uTxtOrigin = glGetUniformLocation(textProg, "uOrigin");
    int uTxtSize = glGetUniformLocation(textProg, "uCharSize");
    int uTxtAdvance = glGetUniformLocation(textProg, "uAdvance");
    int uTxtFont = glGetUniformLocation(textProg, "uFont");
    int uTxtChars = glGetUniformLocation(textProg, "uChars");
    int uTxtColor = glGetUniformLocation(textProg, "uTextColor");
    int uRectOrigin = glGetUniformLocation(rectProg, "uRectOrigin");
    int uRectSize = glGetUniformLocation(rectProg, "uRectSize");
    int uRectColor = glGetUniformLocation(rectProg, "uRectColor");

    // ── CFL stability ─────────────────────────────────────────────────────────
    const int   N = RES;
    const float h = (N > 1) ? (2.0f / (N - 1)) : 1.0f;
    const float invH2 = 1.0f / (h * h);
    const float dtStable = (h * h) / (4.0f * DIFFUSION);
    const float dtMax = 0.016f;
    const int   substeps = std::max(1, (int)std::ceil(dtMax / dtStable));
    const float subDt = dtStable * 0.9f;

    std::cout << "Resolution: " << N << "x" << N
        << "  substeps/frame: " << substeps
        << "  subDt: " << subDt << "\n";

    // ── Field SSBOs ───────────────────────────────────────────────────────────
    const int STRIDE = 10;
    std::vector<float> initBuf(N * N * STRIDE, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float x = -1.0f + j * h, y = -1.0f + i * h;
            initBuf[(i * N + j) * STRIDE] = (x > 0.0f && y > 0.0f) ? 100.0f : -100.0f;
        }

    unsigned int ssbo[2];
    glGenBuffers(2, ssbo);
    for (int b = 0; b < 2; ++b) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[b]);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
            initBuf.size() * sizeof(float),
            b == 0 ? initBuf.data() : nullptr,
            GL_DYNAMIC_COPY);
    }

    // ── Static field mesh ─────────────────────────────────────────────────────
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
    glGenVertexArrays(1, &fieldVAO);
    glGenBuffers(1, &fieldVBO);
    glBindVertexArray(fieldVAO);
    glBindBuffer(GL_ARRAY_BUFFER, fieldVBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.size() * sizeof(Vert), mesh.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vert), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribIPointer(1, 1, GL_UNSIGNED_INT, sizeof(Vert), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // ── Shared unit quad VAO  [0,1]×[0,1]  ───────────────────────────────────
    // Used by both the text overlay and the background rect.
    float quadVerts[] = {
        0.0f,0.0f, 1.0f,0.0f, 0.0f,1.0f,
        1.0f,0.0f, 1.0f,1.0f, 0.0f,1.0f
    };
    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // ── Timer layout constants ─────────────────────────────────────────────────
    // Each font pixel = 5 screen pixels.
    // Character cell = 4×5px font pixels = 20×30 screen pixels.
    // NDC unit = 2/800 = 0.0025 per screen pixel.
    const float PX = 2.0f / 800.0f;   // 1 screen pixel in NDC
    const float CHAR_W = 20.0f * PX;       // NDC width of one character cell
    const float CHAR_H = 30.0f * PX;       // NDC height
    const float CHAR_ADV = 22.0f * PX;       // x advance (cell + 2px gap)
    const float MARGIN = 12.0f * PX;       // padding from window edge
    const float PAD = 6.0f * PX;       // padding inside background box
    // Top-left corner of the first character in NDC (origin = top-left = (-1, 1))
    const float TEXT_X = -1.0f + MARGIN;
    const float TEXT_Y = 1.0f - MARGIN;

    const int   groups = (N + 15) / 16;
    int         current = 0;
    float       simTime = 0.0f;
    double      lastTime = glfwGetTime();
    float       rotMat[16];

    while (!glfwWindowShouldClose(window))
    {
        lastTime = glfwGetTime();   // we use fixed substepDt, not wall dt

        // ── Compute pass: CFL-limited substeps ────────────────────────────────
        glUseProgram(computeProg);
        glUniform1i(uRes, N);
        glUniform1f(uInvH2u, invH2);
        glUniform1f(uDiffusion, DIFFUSION);
        glUniform1f(uDt, subDt);

        for (int step = 0; step < substeps; ++step) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
            glDispatchCompute(groups, groups, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            current = 1 - current;
        }

        // Advance simulation clock by exactly what we stepped
        simTime += subDt * static_cast<float>(substeps);

        // ── Field render pass ─────────────────────────────────────────────────
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(fieldProg);
        buildRotationMatrix(rot.yaw, rot.pitch, rotMat);
        glUniformMatrix4fv(uRotation, 1, GL_FALSE, rotMat);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
        glBindVertexArray(fieldVAO);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(mesh.size()));

        // ── Text overlay pass ─────────────────────────────────────────────────
        // Disable depth test so text always draws on top.
        // Enable blending for the semi-transparent background box.
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Format simulation time into glyph indices
        uint32_t charBuf[16];
        int numChars = formatSimTime(simTime, charBuf, 16);

        // Width of the full text string in NDC
        float textW = CHAR_W + (numChars - 1) * CHAR_ADV;

        // 1. Draw dark semi-transparent background rectangle behind the text
        glUseProgram(rectProg);
        glUniform2f(uRectOrigin, TEXT_X - PAD, TEXT_Y + PAD);
        glUniform2f(uRectSize, textW + 2.0f * PAD, CHAR_H + 2.0f * PAD);
        glUniform4f(uRectColor, 0.0f, 0.0f, 0.0f, 0.65f);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // 2. Draw white text on top
        glUseProgram(textProg);
        glUniform2f(uTxtOrigin, TEXT_X, TEXT_Y);
        glUniform2f(uTxtSize, CHAR_W, CHAR_H);
        glUniform1f(uTxtAdvance, CHAR_ADV);
        glUniform1uiv(uTxtFont, FONT_COUNT, FONT);
        glUniform1uiv(uTxtChars, numChars, charBuf);
        glUniform4f(uTxtColor, 1.0f, 1.0f, 1.0f, 1.0f);
        glBindVertexArray(quadVAO);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, numChars);

        glDisable(GL_BLEND);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteBuffers(2, ssbo);
    glDeleteBuffers(1, &fieldVBO);   glDeleteVertexArrays(1, &fieldVAO);
    glDeleteBuffers(1, &quadVBO);    glDeleteVertexArrays(1, &quadVAO);
    glDeleteProgram(computeProg);    glDeleteProgram(fieldProg);
    glDeleteProgram(textProg);       glDeleteProgram(rectProg);
    glfwTerminate();
    return 0;
}
