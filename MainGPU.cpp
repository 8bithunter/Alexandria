// =============================================================================
// Field Simulation — GPU Compute Edition
// Requires OpenGL 4.3 (compute shaders + SSBOs)
//
// Simulation modes (keys 1 / 2 / 3 / 4):
//   1 – Diffusion   : ∂u/∂t = D ∇²u
//   2 – Wave        : ∂²u/∂t² = c² ∇²u
//   3 – Fluid Flow  : semi-Lagrangian advection + viscosity
//       Drag cursor to push fluid; speed proportional to drag speed
//       hue = flow direction, height = speed magnitude
//   4 – Schrödinger : i ∂ψ/∂t = –½ ∇²ψ   (ħ = m = 1)
//       FX = Re(ψ),  FY = Im(ψ)
//       height = |ψ|,  hue = arg(ψ)
//
// Other controls:
//   Left-click+drag  – paint excitation (fluid: pushes in drag direction)
//   Ctrl+Scroll      – change paint value / fluid push strength
//   Alt+Scroll       – change brush radius
//   Shift+Scroll     – camera up / down
//   Scroll           – zoom in / out
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
#include "ShaderLoader.h"

static int   RES = 200;
static float DIFFUSION = 0.01f;

// simulationMode:  0=diffusion  1=wave  2=fluid  3=schrödinger
static int simulationMode = 0;
static bool g_returnToConsole = false;


// Setter functions for runtime parameter modification
void setResolution(int resolution) {
	if (resolution > 0 && resolution <= 1000) {
		RES = resolution;
		std::cout << "Resolution set to: " << RES << "\n";
	}
	else {
		std::cout << "Invalid resolution. Must be between 1 and 1000.\n";
	}
}

void setCoefficient(float coefficient) {
	if (coefficient > 0.0f && coefficient <= 10.0f) {
		DIFFUSION = coefficient;
		std::cout << "Coefficient set to: " << DIFFUSION << "\n";
	}
	else {
		std::cout << "Invalid coefficient. Must be positive and <= 10.0.\n";
	}
}

// =============================================================================
// Bitmap font  (4 wide × 6 tall, packed as 6 nibbles into uint32_t)
// =============================================================================
// clang-format off
static const uint32_t FONT[] = {
	0x6 | (0x9 << 4) | (0x9 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20),  //  0  'O'
	0x2 | (0x3 << 4) | (0x2 << 8) | (0x2 << 12) | (0x2 << 16) | (0xE << 20),  //  1
	0x7 | (0x8 << 4) | (0x6 << 8) | (0x1 << 12) | (0x1 << 16) | (0xF << 20),  //  2
	0x7 | (0x8 << 4) | (0x6 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20),  //  3
	0x9 | (0x9 << 4) | (0xF << 8) | (0x8 << 12) | (0x8 << 16) | (0x8 << 20),  //  4
	0xF | (0x1 << 4) | (0x7 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20),  //  5
	0x6 | (0x1 << 4) | (0x7 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20),  //  6
	0xF | (0x8 << 4) | (0x4 << 8) | (0x2 << 12) | (0x2 << 16) | (0x2 << 20),  //  7
	0x6 | (0x9 << 4) | (0x6 << 8) | (0x9 << 12) | (0x9 << 16) | (0x6 << 20),  //  8
	0x6 | (0x9 << 4) | (0xE << 8) | (0x8 << 12) | (0x8 << 16) | (0x6 << 20),  //  9
	0x0 | (0x0 << 4) | (0x0 << 8) | (0x0 << 12) | (0x2 << 16) | (0x0 << 20),  // 10  '.'
	0xE | (0x1 << 4) | (0x6 << 8) | (0x8 << 12) | (0x8 << 16) | (0x7 << 20),  // 11  'S'
	0x0,                                                    // 12  ' '
	0x7 | (0x9 << 4) | (0x9 << 8) | (0x9 << 12) | (0x9 << 16) | (0x7 << 20),  // 13  'D'
	0xF | (0x6 << 4) | (0x6 << 8) | (0x6 << 12) | (0x6 << 16) | (0xF << 20),  // 14  'I'
	0xF | (0x1 << 4) | (0x7 << 8) | (0x1 << 12) | (0x1 << 16) | (0x1 << 20),  // 15  'F'
	0x9 | (0x9 << 4) | (0xF << 8) | (0x6 << 12) | (0x9 << 16) | (0x9 << 20),  // 16  'W'
	0x6 | (0x9 << 4) | (0xF << 8) | (0x9 << 12) | (0x9 << 16) | (0x9 << 20),  // 17  'A'
	0x9 | (0x9 << 4) | (0x9 << 8) | (0x9 << 12) | (0x6 << 16) | (0x6 << 20),  // 18  'V'
	0xE | (0x1 << 4) | (0x1 << 8) | (0x1 << 12) | (0x1 << 16) | (0xE << 20),  // 19  'C'
	0x9 | (0x9 << 4) | (0xF << 8) | (0x9 << 12) | (0x9 << 16) | (0x9 << 20),  // 20  'H'
	0x8 | (0x8 << 4) | (0x8 << 8) | (0x8 << 12) | (0x8 << 16) | (0xF << 20),  // 21  'L'
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
static constexpr int GLYPH_L = 21;
static constexpr int GLYPH_O = 0;
static constexpr int FONT_COUNT = 22;

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
	else if (mode == 2) { out[0] = GLYPH_F; out[1] = GLYPH_L; out[2] = GLYPH_O; }
	else { out[0] = GLYPH_S; out[1] = GLYPH_C; out[2] = GLYPH_H; }
	return 3;
}

// =============================================================================
// Compute shader
// uMode: 0=diffusion  1=wave  2=fluid  3=schrod_re  4=schrod_im
// =============================================================================

// =============================================================================
// Shaders loaded from files
// =============================================================================
static std::string computeSrc = loadShaderFile("shaders/compute.glsl");
static std::string fieldVertSrc = loadShaderFile("shaders/field.vert");
static std::string fieldFragSrc = loadShaderFile("shaders/field.frag");
static std::string textVertSrc = loadShaderFile("shaders/text.vert");
static std::string textFragSrc = loadShaderFile("shaders/text.frag");
static std::string rectVertSrc = loadShaderFile("shaders/rect.vert");
static std::string rectFragSrc = loadShaderFile("shaders/rect.frag");
static std::string axesVertSrc = loadShaderFile("shaders/axes.vert");
static std::string axesFragSrc = loadShaderFile("shaders/axes.frag");

// =============================================================================
// Global input state
// =============================================================================
struct GlobeState {
	bool dragging = false;
	double lastX = 0, lastY = 0;
	float spin = 0.0f;
	float pitch = 0.5f;
	static constexpr float SENSITIVITY = 0.008f;
};
static GlobeState rot;

// Cursor position (OpenGL convention: y=0 at bottom) and per-frame delta.
// heatDX/heatDY are accumulated in cursorPosCallback and consumed once per
// frame in the paint section, then cleared.
static bool  heatActive = false;
static float heatCurX = 0.0f;
// Cursor position in field space (normalized -1 to 1)
static float cursorX = 0.0f;
static float cursorY = 0.0f;
static float heatCurY = 0.0f;   // OpenGL y (0 = bottom)
static float heatDX = 0.0f;
static float heatDY = 0.0f;

static float heatValue = 100.0f;
static int   heatRadius = 5;
static float zoom = 2.5f;
static float cameraY = 0.0f;
static bool  paused = false;
static bool  resetRequested = false;

// =============================================================================
// Callbacks
// =============================================================================
static void keyCallback(GLFWwindow* window, int key, int, int action, int)
{
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) paused = !paused;
	if (key == GLFW_KEY_R && action == GLFW_PRESS) resetRequested = true;
}

static void scrollCallback(GLFWwindow* w, double, double yoff)
{
	bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
	bool ctrl = glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
	bool alt = glfwGetKey(w, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
	if (shift) cameraY = std::clamp(cameraY + (float)yoff * 0.05f, -5.0f, 5.0f);
	else if (alt) { heatRadius = std::clamp(heatRadius + (yoff > 0 ? 1 : -1), 1, 64); std::cout << "Brush radius: " << heatRadius << "\n"; }
	else if (ctrl) { heatValue = std::clamp(heatValue + (float)yoff * 10.0f, 1.0f, 5000.0f); std::cout << "Paint value: " << heatValue << "\n"; }
	else            zoom = std::clamp(zoom - (float)yoff * 0.15f, 0.5f, 10.0f);
}

static void mouseButtonCallback(GLFWwindow* w, int btn, int action, int)
{
	if (btn == GLFW_MOUSE_BUTTON_RIGHT) {
		rot.dragging = (action == GLFW_PRESS);
		if (rot.dragging) glfwGetCursorPos(w, &rot.lastX, &rot.lastY);
	}
	if (btn == GLFW_MOUSE_BUTTON_LEFT) {
		heatActive = (action == GLFW_PRESS);
		if (action == GLFW_PRESS) {
			// Zero the delta so the first paint tick doesn't get a stale value.
			heatDX = 0.0f;
			heatDY = 0.0f;
		}
	}
}

static void cursorPosCallback(GLFWwindow*, double x, double y)
{
	// y is flipped to OpenGL convention (0 = bottom).
	float newX = (float)x;
	float newY = -(float)y + 800.0f;

	if (rot.dragging) {
		float dx = newX - (float)rot.lastX;
		float dy = (float)y - (float)rot.lastY;   // screen-space dy (unflipped)
		rot.spin -= dx * GlobeState::SENSITIVITY;
		rot.pitch += dy * GlobeState::SENSITIVITY;
		rot.pitch = std::clamp(rot.pitch, 0.0f, 3.14159265f - 0.0001f);
		rot.lastX = x; rot.lastY = y;
	}

	// Accumulate delta since last frame.  Multiple callbacks can fire between
	// frames (e.g. on high-refresh monitors), so we add rather than assign.
	heatDX += newX - heatCurX;
	heatDY += newY - heatCurY;
	heatCurX = newX;
	heatCurY = newY;
	cursorX = heatCurX;
	cursorY = heatCurY;
}

// =============================================================================
// Math helpers
// =============================================================================
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

static bool unprojectToField(float sx, float sy, const float* MVP,
	float& worldX, float& worldY)
{
	float invMVP[16];
	mat4Inverse(MVP, invMVP);
	float nx = 2.0f * sx / 800.0f - 1.0f;
	float ny = 2.0f * sy / 800.0f - 1.0f;
	auto unproj = [&](float nz, float out[3]) {
		float clip[4] = { nx,ny,nz,1.0f }, w[4] = {};
		for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) w[r] += invMVP[c * 4 + r] * clip[c];
		float invW = (fabsf(w[3]) > 1e-8f) ? 1.0f / w[3] : 1.0f;
		out[0] = w[0] * invW; out[1] = w[1] * invW; out[2] = w[2] * invW;
		};
	float nearPt[3], farPt[3];
	unproj(-1.0f, nearPt);
	unproj(1.0f, farPt);
	float dz = farPt[2] - nearPt[2];
	if (fabsf(dz) < 1e-6f) return false;
	float t = -nearPt[2] / dz;
	worldX = nearPt[0] + t * (farPt[0] - nearPt[0]);
	worldY = nearPt[1] + t * (farPt[1] - nearPt[1]);
	return true;
}

static void buildGlobeMatrix(float spin, float pitch, float* m)
{
	float cs = cosf(spin), ss = sinf(spin), cp = cosf(pitch), sp = sinf(pitch);
	m[0] = cs;  m[1] = -ss * cp; m[2] = ss * sp; m[3] = 0;
	m[4] = ss;  m[5] = cs * cp; m[6] = -cs * sp; m[7] = 0;
	m[8] = 0;   m[9] = sp;     m[10] = cp;    m[11] = 0;
	m[12] = 0;  m[13] = 0;     m[14] = 0;     m[15] = 1;
}

static void matMul(const float* A, const float* B, float* C)
{
	for (int col = 0; col < 4; ++col) for (int row = 0; row < 4; ++row) {
		float s = 0; for (int k = 0; k < 4; ++k) s += A[k * 4 + row] * B[col * 4 + k]; C[col * 4 + row] = s;
	}
}

static void buildPerspective(float fovY, float aspect, float near, float far, float* m)
{
	float f = 1.0f / tanf(fovY * 0.5f);
	memset(m, 0, 64);
	m[0] = f / aspect; m[5] = f;
	m[10] = (far + near) / (near - far); m[11] = -1.0f;
	m[14] = 2.0f * far * near / (near - far);
}

static void buildTranslation(float tx, float ty, float tz, float* m)
{
	memset(m, 0, 64); m[0] = m[5] = m[10] = m[15] = 1.0f;
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
int runSimulation(int simulationMode)
{
	if (!glfwInit()) return -1;
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(800, 800,
		"Field Simulation (GPU) — 1:Diffusion  2:Wave  3:Fluid  4:Schrodinger",
		nullptr, nullptr);
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

  // ── Programs ──────────────────────────────────────────────────────────────
	unsigned int computeProg = makeProgram({ compileShader(GL_COMPUTE_SHADER, computeSrc.c_str()) });
	unsigned int fieldProg = makeProgram({ compileShader(GL_VERTEX_SHADER, fieldVertSrc.c_str()),
										   compileShader(GL_FRAGMENT_SHADER, fieldFragSrc.c_str()) });
	unsigned int textProg = makeProgram({ compileShader(GL_VERTEX_SHADER, textVertSrc.c_str()),
										  compileShader(GL_FRAGMENT_SHADER, textFragSrc.c_str()) });
	unsigned int rectProg = makeProgram({ compileShader(GL_VERTEX_SHADER, rectVertSrc.c_str()),
										  compileShader(GL_FRAGMENT_SHADER, rectFragSrc.c_str()) });



	unsigned int axesProg = makeProgram({ compileShader(GL_VERTEX_SHADER, axesVertSrc.c_str()), compileShader(GL_FRAGMENT_SHADER, axesFragSrc.c_str()) });
	static std::string topomapVertSrc = loadShaderFile("shaders/topomap.vert");
	static std::string topomapFragSrc = loadShaderFile("shaders/topomap.frag");
	unsigned int topomapProg = makeProgram({ compileShader(GL_VERTEX_SHADER, topomapVertSrc.c_str()), compileShader(GL_FRAGMENT_SHADER, topomapFragSrc.c_str()) });
	static std::string linegraphVertSrc = loadShaderFile("shaders/linegraph.vert");
	static std::string linegraphFragSrc = loadShaderFile("shaders/linegraph.frag");
	unsigned int linegraphProg = makeProgram({ compileShader(GL_VERTEX_SHADER, linegraphVertSrc.c_str()), compileShader(GL_FRAGMENT_SHADER, linegraphFragSrc.c_str()) });
	// ── Uniform locations ─────────────────────────────────────────────────────
	int uResU = glGetUniformLocation(computeProg, "uRes");
	int uInvH2U = glGetUniformLocation(computeProg, "uInvH2");
	int uDiffusionU = glGetUniformLocation(computeProg, "uDiffusion");
	int uDensityU = glGetUniformLocation(computeProg, "uDensity");
	int uDtU = glGetUniformLocation(computeProg, "uDt");
	int uComputeModeU = glGetUniformLocation(computeProg, "uMode");

	int uRotationU = glGetUniformLocation(fieldProg, "uRotation");
	int uFieldModeU = glGetUniformLocation(fieldProg, "uMode");
	int uFieldResU = glGetUniformLocation(fieldProg, "uRes");
	int uFieldYOffsetU = glGetUniformLocation(fieldProg, "uYOffset");

	int uTxtOrigin = glGetUniformLocation(textProg, "uOrigin");
	int uTxtSize = glGetUniformLocation(textProg, "uCharSize");
	int uTxtAdvance = glGetUniformLocation(textProg, "uAdvance");
	int uTxtFont = glGetUniformLocation(textProg, "uFont");
	int uTxtChars = glGetUniformLocation(textProg, "uChars");
	int uTxtColor = glGetUniformLocation(textProg, "uTextColor");

	int uRectOrigin = glGetUniformLocation(rectProg, "uRectOrigin");
	int uRectSize = glGetUniformLocation(rectProg, "uRectSize");
	int uRectColor = glGetUniformLocation(rectProg, "uRectColor");

	int uAxesRotationU = glGetUniformLocation(axesProg, "uRotation");
	int uAxesLengthU = glGetUniformLocation(axesProg, "uAxisLength");
	int uAxesOriginU = glGetUniformLocation(axesProg, "uOrigin");

	int uTopomapRotationU = glGetUniformLocation(topomapProg, "uRotation");
	int uTopomapModeU = glGetUniformLocation(topomapProg, "uMode");
	int uTopomapResU = glGetUniformLocation(topomapProg, "uRes");
	int uTopomapZOffsetU = glGetUniformLocation(topomapProg, "uZOffset");
	int uTopomapOpacityU = glGetUniformLocation(topomapProg, "uOpacity");
	int uTopomapFieldMinU = glGetUniformLocation(topomapProg, "uFieldMin");
	int uTopomapFieldMaxU = glGetUniformLocation(topomapProg, "uFieldMax");
	int uTopomapDrawContoursU = glGetUniformLocation(topomapProg, "uDrawContours");

	int uLinegraphModeU = glGetUniformLocation(linegraphProg, "uMode");
	int uLinegraphResU = glGetUniformLocation(linegraphProg, "uRes");
	int uLinegraphCursorXU = glGetUniformLocation(linegraphProg, "uCursorX");
	int uLinegraphCursorYU = glGetUniformLocation(linegraphProg, "uCursorY");
	int uLinegraphOpacityU = glGetUniformLocation(linegraphProg, "uOpacity");
	int uLinegraphAxisOffsetU = glGetUniformLocation(linegraphProg, "uAxisOffset");
	int uLinegraphAxisU = glGetUniformLocation(linegraphProg, "uAxis");

	// ── Grid geometry ─────────────────────────────────────────────────────────
	const int   N = RES;
	const float h = (N > 1) ? (2.0f / (N - 1)) : 1.0f;
	const float invH2 = 1.0f / (h * h);

	const float SIM_SPEED = 1.0f;
	const float TARGET_STEP_DT = 1.0f / 120.0f;

	const float subDtDiff = std::min((h * h) / (4.0f * DIFFUSION) * 0.9f, TARGET_STEP_DT);
	const float subDtWave = std::min((h / (sqrtf(DIFFUSION) * sqrtf(2.0f))) * 0.9f, TARGET_STEP_DT);
	const float subDtSchrod = 0.45f * h * h;
	const float subDtFluid = TARGET_STEP_DT;

	// FLUID_VEL_SCALE converts screen-pixel-delta to world velocity.
	// With heatValue=100 and a 10 px/frame drag → 100 * 10 * 0.0005 = 0.5 world/s.
	// The drag speed is naturally proportional because heatDX/heatDY encode
	// pixels moved since the last frame — faster drag = bigger delta = more push.
	const float FLUID_VEL_SCALE = 0.0005f;

	std::cout << "Resolution: " << N << "x" << N
		<< "  dt(schrod)=" << subDtSchrod
		<< "\nControls:\n"
		<< "  1/2/3/4           — Diffusion / Wave / Fluid / Schrodinger\n"
		<< "  Left-click+drag   — paint / push fluid (speed ∝ drag speed)\n"
		<< "  Scroll            — zoom\n"
		<< "  Shift+Scroll      — camera Y\n"
		<< "  Alt+Scroll        — brush radius (" << heatRadius << ")\n"
		<< "  Ctrl+Scroll       — paint / push strength (" << heatValue << ")\n"
		<< "  Right-click+drag  — rotate view\n"
		<< "  Space             — pause / resume\n"
		<< "  R                 — reset\n";


	// ── 3D Axes geometry ──────────────────────────────────────────────────────
	struct AxesVert { float x, y, z; float r, g, b; };
	std::vector<AxesVert> axesVerts;
	// All axes span -1.0 to 1.0 for consistency (simpler implementation)
	float axisLength = 2.0f;  // All axes span -1.0 to 1.0
	// X axis (red): -1.0 to 1.0
	axesVerts.push_back({ -1.0f, 0, 0, 1, 0, 0 });
	axesVerts.push_back({ 1.0f, 0, 0, 1, 0, 0 });
	// Y axis (green): -1.0 to 1.0 (not 0 to 1)
	axesVerts.push_back({ 0, -1.0f, 0, 0, 1, 0 });
	axesVerts.push_back({ 0, 1.0f, 0, 0, 1, 0 });
	// Z axis (blue): -1.0 to 1.0
	axesVerts.push_back({ 0, 0, -1.0f, 0, 0, 1 });
	axesVerts.push_back({ 0, 0, 1.0f, 0, 0, 1 });
	// Add tick marks every 0.1 units (21 ticks: -1.0 to 1.0)
	for (int i = 0; i <= 20; ++i) {
		float t = -1.0f + i * 0.1f; // from -1.0 to 1.0
		// X axis ticks
		axesVerts.push_back({ t, -0.05f, 0, 0.5f, 0, 0 });
		axesVerts.push_back({ t, 0.05f, 0, 0.5f, 0, 0 });
		// Y axis ticks
		axesVerts.push_back({ -0.05f, t, 0, 0, 0.5f, 0 });
		axesVerts.push_back({ 0.05f, t, 0, 0, 0.5f, 0 });
		// Z axis ticks
		axesVerts.push_back({ -0.05f, 0, t, 0, 0, 0.5f });
		axesVerts.push_back({ 0.05f, 0, t, 0, 0, 0.5f });
	}
	unsigned int axesVAO, axesVBO;
	glGenVertexArrays(1, &axesVAO);
	glGenBuffers(1, &axesVBO);
	glBindVertexArray(axesVAO);
	glBindBuffer(GL_ARRAY_BUFFER, axesVBO);
	glBufferData(GL_ARRAY_BUFFER, axesVerts.size() * sizeof(AxesVert), axesVerts.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(AxesVert), (void*)0); // position
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(AxesVert), (void*)(3 * sizeof(float))); // color
	glEnableVertexAttribArray(1);
	glBindVertexArray(0);
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
	// ── Topological map VAO ──────────────────────────────────────────────────
	// Create 21x21 grid for 20x20 cells (slices) for field visualization
	struct TopomapVert { float x, y, u, v; };
	std::vector<TopomapVert> topomapVerts;
	for (int i = 0; i <= 20; ++i) {
		for (int j = 0; j <= 20; ++j) {

			float x = -1.0f + i * (2.0f / 20.0f);
			float y = -1.0f + j * (2.0f / 20.0f);
			float u = float(j) / 20.0f;
			float v = float(i) / 20.0f;
			topomapVerts.push_back({ x, y, u, v });
		}
	}
	std::vector<unsigned int> topomapIndices;
	for (int i = 0; i < 20; ++i) {
		for (int j = 0; j < 20; ++j) {
			int bl = i * 21 + j; int br = i * 21 + (j + 1);
			int tl = (i + 1) * 21 + j; int tr = (i + 1) * 21 + (j + 1);
			topomapIndices.push_back(bl); topomapIndices.push_back(br); topomapIndices.push_back(tl);
			topomapIndices.push_back(br); topomapIndices.push_back(tr); topomapIndices.push_back(tl);
		}
	}
	unsigned int topomapVAO, topomapVBO, topomapEBO;
	glGenVertexArrays(1, &topomapVAO); glGenBuffers(1, &topomapVBO); glGenBuffers(1, &topomapEBO);
	glBindVertexArray(topomapVAO); glBindBuffer(GL_ARRAY_BUFFER, topomapVBO);
	glBufferData(GL_ARRAY_BUFFER, topomapVerts.size() * sizeof(TopomapVert), topomapVerts.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, topomapEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, topomapIndices.size() * sizeof(unsigned int), topomapIndices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(TopomapVert), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(TopomapVert), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);
	glBindVertexArray(0);

	// ── Shared unit-quad VAO ──────────────────────────────────────────────────
	float quadVerts[] = { 0,0, 1,0, 0,1,  1,0, 1,1, 0,1 };
	unsigned int quadVAO, quadVBO;
	glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO);
	glBindVertexArray(quadVAO);
	// ── Line graph VAOs ─────────────────────────────────────────────────────────
	struct LinegraphVert { float pos; float coord; };
	std::vector<LinegraphVert> horizLineVerts, vertLineVerts;
	const int LINE_SEGS = 20;
	for (int i = 0; i <= LINE_SEGS; ++i) {
		float pos = -1.0f + i * (2.0f / LINE_SEGS);
		float coord = float(i) / LINE_SEGS;
		horizLineVerts.push_back({ pos, coord });
		vertLineVerts.push_back({ pos, coord });
	}
	unsigned int linegraphVAO[2], linegraphVBO[2];
	glGenVertexArrays(2, linegraphVAO); glGenBuffers(2, linegraphVBO);
	for (int i = 0; i < 2; ++i) {
		glBindVertexArray(linegraphVAO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, linegraphVBO[i]);
		if (i == 0) {
			glBufferData(GL_ARRAY_BUFFER, horizLineVerts.size() * sizeof(LinegraphVert), horizLineVerts.data(), GL_STATIC_DRAW);
		}
		else {
			glBufferData(GL_ARRAY_BUFFER, vertLineVerts.size() * sizeof(LinegraphVert), vertLineVerts.data(), GL_STATIC_DRAW);
		}
		glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(LinegraphVert), (void*)0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(LinegraphVert), (void*)(sizeof(float)));
		glEnableVertexAttribArray(1);
		glBindVertexArray(0);
	}
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

	// ── UI layout ─────────────────────────────────────────────────────────────
	const float PX = 2.0f / 800.0f;
	const float CHAR_W = 20.0f * PX;
	const float CHAR_H = 30.0f * PX;
	const float CHAR_ADV = 22.0f * PX;
	const float MARGIN = 12.0f * PX;
	const float PAD = 6.0f * PX;
	const float TEXT_X = -1.0f + MARGIN;
	const float TEXT_Y = 1.0f - MARGIN;

	const int groups = (N + 15) / 16;
	int   current = 0;
	float simTime = 0.0f;
	double lastTime = glfwGetTime();
	float accumDiff = 0.0f;
	float accumWave = 0.0f;
	float accumSchrod = 0.0f;
	float accumFluid = 0.0f;

	// ── Main loop ─────────────────────────────────────────────────────────────
	while (!glfwWindowShouldClose(window) && !g_returnToConsole)
	{
		// ── Poll events FIRST so heatDX/heatDY are fresh before painting ──────
	 // Optional: Check for ESC key directly (in case callback missed it)
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
			g_returnToConsole = true;
		}
		glfwPollEvents();

		// ── Wall-clock delta time ─────────────────────────────────────────────
		double nowTime = glfwGetTime();
		float  realDt = (float)(nowTime - lastTime);
		lastTime = nowTime;
		float simBudget = std::min(realDt, 4.0f / 60.0f) * SIM_SPEED;

		// ── Reset ─────────────────────────────────────────────────────────────
		if (resetRequested) {
			resetRequested = false;
			simTime = 0.0f; current = 0;
			accumDiff = accumWave = accumSchrod = accumFluid = 0.0f;
			for (int b = 0; b < 2; ++b) {
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[b]);
				glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
			}
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		}

		// ── Build MVP ─────────────────────────────────────────────────────────
		float R[16], T[16], P[16], TR[16], MVP[16];
		buildGlobeMatrix(rot.spin, rot.pitch, R);
		buildTranslation(0.0f, cameraY, -zoom, T);
		buildPerspective(3.14159265f / 3.0f, 1.0f, 0.01f, 100.0f, P);
		matMul(T, R, TR);
		matMul(P, TR, MVP);

		// ── Paint excitation ──────────────────────────────────────────────────
		if (heatActive) {
			float worldX, worldY;
			int cx = N / 2, cy = N / 2;
			if (unprojectToField(heatCurX, heatCurY, MVP, worldX, worldY)) {
				cx = (int)((worldX + 1.0f) * 0.5f * N);
				cy = (int)((worldY + 1.0f) * 0.5f * N);
			}

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[current]);

			if (simulationMode == 2) {
				if (fabsf(heatDX) > 0.1f || fabsf(heatDY) > 0.1f) {
					// Unproject current and previous screen positions → world-space drag vector.
					// This naturally accounts for camera spin and pitch.
					float wx0, wy0, wx1, wy1;
					bool ok0 = unprojectToField(heatCurX, heatCurY, MVP, wx0, wy0);
					bool ok1 = unprojectToField(heatCurX - heatDX, heatCurY - heatDY, MVP, wx1, wy1);
					if (ok0 && ok1) {
						float velX = (wx0 - wx1) * heatValue * 0.1f;
						float velY = (wy0 - wy1) * heatValue * 0.1f;
						float vel[2] = { velX, velY };
						for (int dy = -heatRadius; dy <= heatRadius; ++dy)
							for (int dx = -heatRadius; dx <= heatRadius; ++dx) {
								if (dx * dx + dy * dy > heatRadius * heatRadius) continue;
								int gi = cy + dy, gj = cx + dx;
								if (gi < 0 || gi >= N || gj < 0 || gj >= N) continue;
								GLintptr off = (GLintptr)((gi * N + gj) * STRIDE) * sizeof(float);
								glBufferSubData(GL_SHADER_STORAGE_BUFFER, off, 2 * sizeof(float), vel);
							}
					}
				}
			}
			else
			{
				// Scalar modes: paint heatValue into FX.
				for (int dy = -heatRadius; dy <= heatRadius; ++dy)
					for (int dx = -heatRadius; dx <= heatRadius; ++dx) {
						if (dx * dx + dy * dy > heatRadius * heatRadius) continue;
						int gi = cy + dy, gj = cx + dx;
						if (gi < 0 || gi >= N || gj < 0 || gj >= N) continue;
						GLintptr off = (GLintptr)((gi * N + gj) * STRIDE) * sizeof(float);
						glBufferSubData(GL_SHADER_STORAGE_BUFFER, off, sizeof(float), &heatValue);
					}
			}
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		}

		// Delta consumed — clear for next frame.
		heatDX = 0.0f;
		heatDY = 0.0f;

		// ── Compute pass ──────────────────────────────────────────────────────
		if (!paused) {
			glUseProgram(computeProg);
			glUniform1i(uResU, N);
			glUniform1f(uInvH2U, invH2);

			if (simulationMode == 0) {
				accumDiff += simBudget;
				glUniform1f(uDiffusionU, DIFFUSION);
				glUniform1i(uComputeModeU, 0);
				glUniform1f(uDtU, subDtDiff);
				while (accumDiff >= subDtDiff) {
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
					glDispatchCompute(groups, groups, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
					current = 1 - current; accumDiff -= subDtDiff; simTime += subDtDiff;
				}
			}
			else if (simulationMode == 1) {
				accumWave += simBudget;
				glUniform1f(uDiffusionU, DIFFUSION);
				glUniform1i(uComputeModeU, 1);
				glUniform1f(uDtU, subDtWave);
				while (accumWave >= subDtWave) {
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
					glDispatchCompute(groups, groups, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
					current = 1 - current; accumWave -= subDtWave; simTime += subDtWave;
				}
			}
			else if (simulationMode == 2) {
				accumFluid += simBudget;
				glUniform1f(uDiffusionU, 0.0001f); // kinematic viscosity (reduced for less viscous fluid)
				glUniform1f(uDensityU, 1.0f); // fluid density
				glUniform1f(uDtU, subDtFluid);
				while (accumFluid >= subDtFluid) {
					// Pass 1: Advection + diffusion
					glUniform1i(uComputeModeU, 2);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
					glDispatchCompute(groups, groups, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
					current = 1 - current;

					// Pass 2: Compute divergence
					glUniform1i(uComputeModeU, 4);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
					glDispatchCompute(groups, groups, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
					current = 1 - current;

					// Pass 3: Pressure solve (Jacobi iterations)
					for (int i = 0; i < 20; i++) { // Fixed 20 iterations for now
						glUniform1i(uComputeModeU, 5);
						glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
						glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
						glDispatchCompute(groups, groups, 1);
						glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
						current = 1 - current;
					}

					// Pass 4: Project velocity (pressure projection)
					glUniform1i(uComputeModeU, 6);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
					glDispatchCompute(groups, groups, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
					current = 1 - current;

					accumFluid -= subDtFluid; simTime += subDtFluid;
				}
			}
			else {
				accumSchrod += simBudget;
				glUniform1f(uDiffusionU, DIFFUSION);
				glUniform1f(uDtU, subDtSchrod);
				while (accumSchrod >= subDtSchrod) {
					glUniform1i(uComputeModeU, 3);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
					glDispatchCompute(groups, groups, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
					current = 1 - current;
					glUniform1i(uComputeModeU, 4);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
					glDispatchCompute(groups, groups, 1);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
					current = 1 - current; accumSchrod -= subDtSchrod; simTime += subDtSchrod;
				}
			}
		}

		// ── Field render ──────────────────────────────────────────────────────
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		//glDepthMask(simulationMode == 2 ? GL_FALSE : GL_TRUE); for transparency stuff
		glDepthMask(GL_TRUE);

		// ── Field and axes ─────────────────────────────────────────────
		glUseProgram(fieldProg);
		glUniformMatrix4fv(uRotationU, 1, GL_FALSE, MVP);
		glUniform1i(uFieldModeU, simulationMode);
		glUniform1i(uFieldResU, N);
		glUniform1f(uFieldYOffsetU, 0.5f); // Raise field up by 0.5 units
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
		glBindVertexArray(fieldVAO);
		glDrawArrays(GL_TRIANGLES, 0, (GLsizei)mesh.size());
		// Compute field min/max for proper contour scaling
		float fieldMin = FLT_MAX;
		float fieldMax = -FLT_MAX;
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[current]);
		void* fieldPtr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
		if (fieldPtr) {
			float* fieldData = (float*)fieldPtr;
			for (int i = 0; i < N * N; i++) {
				float val;
				if (simulationMode >= 2) {
					float re = fieldData[i * 10 + 0];
					float im = fieldData[i * 10 + 1];
					val = sqrtf(re * re + im * im);
				}
				else {
					val = fieldData[i * 10 + 0];
				}
				if (val < fieldMin) fieldMin = val;
				if (val > fieldMax) fieldMax = val;
			}
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
		// ── Topological map of field data ───────────────────────────────────────────
		glUseProgram(topomapProg);
		glUniformMatrix4fv(uTopomapRotationU, 1, GL_FALSE, MVP);
		glUniform1f(uTopomapFieldMinU, fieldMin);
		glUniform1f(uTopomapFieldMaxU, fieldMax);
		glUniform1i(uTopomapDrawContoursU, 1);
		glUniform1i(uTopomapModeU, simulationMode);
		glUniform1i(uTopomapResU, N);
		glUniform1f(uTopomapZOffsetU, -1.0f); // Bottom of blue axis
		glUniform1f(uTopomapOpacityU, 0.7f); // 70% opaque
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
		glBindVertexArray(topomapVAO);
		glDrawElements(GL_TRIANGLES, 20 * 20 * 6, GL_UNSIGNED_INT, 0);

		// ── Line graphs ─────────────────────────────────────────────────────────────
		glUseProgram(linegraphProg);
		glUniform1i(uLinegraphModeU, simulationMode);
		glUniform1i(uLinegraphResU, N);
		glUniform1f(uLinegraphCursorXU, cursorX);
		glUniform1f(uLinegraphCursorYU, cursorY);
		glUniform1f(uLinegraphOpacityU, 1.0f);

		// Horizontal: along X at cursorY, positioned at Y = -1.0
		glUniform1f(uLinegraphAxisOffsetU, -1.0f);
		glUniform1i(uLinegraphAxisU, 0);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
		glBindVertexArray(linegraphVAO[0]);
		glDrawArrays(GL_LINE_STRIP, 0, 21);

		// Vertical: along Y at cursorX, positioned at X = -1.0
		glUniform1f(uLinegraphAxisOffsetU, -1.0f);
		glUniform1i(uLinegraphAxisU, 1);
		glBindVertexArray(linegraphVAO[1]);
		glDrawArrays(GL_LINE_STRIP, 0, 21);
		glUseProgram(axesProg);
		glUniform3f(uAxesOriginU, 0.0f, 0.0f, 0.0f); // Position at field center
		glUniformMatrix4fv(uAxesRotationU, 1, GL_FALSE, MVP);
		glUniform1f(uAxesLengthU, 1.0f); // Scale to fit
		glLineWidth(3.0f); // Make axes thicker and visible
		glBindVertexArray(axesVAO);
		glDrawArrays(GL_LINES, 0, 132);
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
		int   numMode = formatModeLabel(simulationMode, modeBuf);
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
		else if (simulationMode == 2) glUniform4f(uTxtColor, 0.2f, 1.0f, 0.4f, 1.0f);
		else                        glUniform4f(uTxtColor, 1.0f, 0.3f, 1.0f, 1.0f);
		glBindVertexArray(quadVAO);
		glDrawArraysInstanced(GL_TRIANGLES, 0, 6, numMode);

		glDisable(GL_BLEND);
		glfwSwapBuffers(window);
		// NOTE: glfwPollEvents() is at the TOP of the loop so heatDX/heatDY
		// are always fresh (non-zero) by the time painting runs.
	}

	glDeleteBuffers(2, ssbo);
	glDeleteBuffers(1, &fieldVBO);  glDeleteVertexArrays(1, &fieldVAO);
	glDeleteBuffers(1, &quadVBO);   glDeleteVertexArrays(1, &quadVAO);
	glDeleteProgram(computeProg);  glDeleteProgram(fieldProg);
	glDeleteProgram(textProg);     glDeleteProgram(rectProg);

	// ── Axes resources cleanup ─────────────────────────────────────────────────
	glDeleteBuffers(1, &axesVBO);
	glDeleteVertexArrays(1, &axesVAO);
	glDeleteProgram(axesProg);
	glfwTerminate();
	return 0;
}