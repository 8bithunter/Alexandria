# Integration Guide for 3D Axes and Planes

## Overview
This guide explains how to integrate the new 3D axes and reference planes visualization into MainGPU.cpp

## Part 1: Shader Loading (around line 145)

Add these lines after the existing shader loading:

```cpp
static std::string planeVertSrc = loadShaderFile("shaders/plane.vert");
static std::string planeFragSrc = loadShaderFile("shaders/plane.frag");
static std::string axesVertSrc = loadShaderFile("shaders/axes.vert");
static std::string axesFragSrc = loadShaderFile("shaders/axes.frag");
```

## Part 2: Global Variables (after line 179)

Add plane opacity variable after the other globals:

```cpp
static float pressureRadius = 20.0f;
static bool paused = false;
static bool resetRequested = false;
static float planeOpacity = 0.5f;  // ADD THIS
```

## Part 3: Key Callback (after line 187)

Add opacity toggle in keyCallback:

```cpp
static void keyCallback(GLFWwindow* window, int key, int, int action, int)
{
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) paused = !paused;
  if (key == GLFW_KEY_R && action == GLFW_PRESS) resetRequested = true;
  if (key == GLFW_KEY_O && action == GLFW_PRESS) {  // ADD THIS
    planeOpacity = (planeOpacity > 0.0f) ? 0.0f : 0.5f;
    std::cout << "Plane opacity: " << planeOpacity << "\n";
  }
}
```

## Part 4: Program Creation (after line 382)

Add new programs after rectProg creation:

```cpp
unsigned int planeProg = makeProgram({
  compileShader(GL_VERTEX_SHADER, planeVertSrc.c_str()),
  compileShader(GL_FRAGMENT_SHADER, planeFragSrc.c_str())
});

unsigned int axesProg = makeProgram({
  compileShader(GL_VERTEX_SHADER, axesVertSrc.c_str()),
  compileShader(GL_FRAGMENT_SHADER, axesFragSrc.c_str())
});
```

## Part 5: Uniform Locations (after line 406)

Add uniform locations after the rectProg uniforms:

```cpp
int uPlaneRotation = glGetUniformLocation(planeProg, "uRotation");
int uPlaneOpacity = glGetUniformLocation(planeProg, "uOpacity");
int uPlaneRes = glGetUniformLocation(planeProg, "uRes");
int uPlaneMode = glGetUniformLocation(planeProg, "uMode");
int uPlaneType = glGetUniformLocation(planeProg, "uPlaneType");
int uPlaneSliceValue = glGetUniformLocation(planeProg, "uSliceValue");

int uAxesRotation = glGetUniformLocation(axesProg, "uRotation");
int uAxesLength = glGetUniformLocation(axesProg, "uAxisLength");
```

## Part 6: Geometry Generation (after line 490)

Insert after the quadVAO setup, before main loop:

```cpp
// ── Reference planes geometry ─────────────────────────────────────────────────
// XY bottom plane
float planeVertices[] = {
  // X, Y, Z, plane type, slice coord
  0, 0, 0,   0, 0,
  RES, 0, 0, 0, 0,
  RES, RES, 0, 0, RES,
  0, RES, 0, 0, RES
};
unsigned int planeIndices[] = { 0,1,2, 0,2,3 };

unsigned int planeVAO, planeVBO, planeEBO;
glGenVertexArrays(1, &planeVAO);
glGenBuffers(1, &planeVBO);
glGenBuffers(1, &planeEBO);

glBindVertexArray(planeVAO);
glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeEBO);
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(planeIndices), planeIndices, GL_STATIC_DRAW);

// Position attribute
GLint posAttrib = glGetAttribLocation(planeProg, "aPos");
glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
glEnableVertexAttribArray(posAttrib);

// Plane type attribute
GLint typeAttrib = glGetAttribLocation(planeProg, "aPlaneType");
glVertexAttribPointer(typeAttrib, 1, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
glEnableVertexAttribArray(typeAttrib);

// Slice coord attribute
GLint sliceAttrib = glGetAttribLocation(planeProg, "aSliceCoord");
glVertexAttribPointer(sliceAttrib, 1, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(4 * sizeof(float)));
glEnableVertexAttribArray(sliceAttrib);

// ── Axes geometry ──────────────────────────────────────────────────────────────
// Three lines: X (red), Y (green), Z (blue)
float axesVertices[] = {
  // Start point, end point, color
  0,0,0, 1,0,0, 1,0,0,  // X axis (red)
  0,0,0, 0,1,0, 0,1,0,  // Y axis (green)
  0,0,0, 0,0,1, 0,0,1   // Z axis (blue)
};

unsigned int axesVAO, axesVBO;
glGenVertexArrays(1, &axesVAO);
glGenBuffers(1, &axesVBO);

glBindVertexArray(axesVAO);
glBindBuffer(GL_ARRAY_BUFFER, axesVBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(axesVertices), axesVertices, GL_STATIC_DRAW);

// Position attribute
GLint axesPosAttrib = glGetAttribLocation(axesProg, "aPos");
glVertexAttribPointer(axesPosAttrib, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
glEnableVertexAttribArray(axesPosAttrib);

// Color attribute
GLint axesColorAttrib = glGetAttribLocation(axesProg, "aColor");
glVertexAttribPointer(axesColorAttrib, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
glEnableVertexAttribArray(axesColorAttrib);
```

## Part 7: Rendering (insert after line 740, before cleanup)

Add before ENABLE_BLENDING:

```cpp
// ── Render reference planes ─────────────────────────────────────────────────
if (planeOpacity > 0.0f) {
  // Enable blending for transparency
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glUseProgram(planeProg);
  glUniformMatrix4fv(uPlaneRotation, 1, GL_FALSE, MVP);
  glUniform1f(uPlaneOpacity, planeOpacity);
  glUniform1i(uPlaneRes, N);
  glUniform1i(uPlaneMode, mode);

  // Bind SSBO
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);

  // Bind plane VAO
  glBindVertexArray(planeVAO);

  // Render each plane
  for (int planeType = 0; planeType < 3; ++planeType) {
    glUniform1i(uPlaneType, planeType);

    // Set slice value based on brush position
    float sliceValue = 0.0f;
    if (planeType == 1) { // XZ plane - slice at Y
      sliceValue = (heatCurY / 800.0f) * N;
    } else if (planeType == 2) { // YZ plane - slice at X
      sliceValue = (heatCurX / 800.0f) * N;
    }
    glUniform1f(uPlaneSliceValue, sliceValue);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  }

  // Render axes
  glUseProgram(axesProg);
  glUniformMatrix4fv(uAxesRotation, 1, GL_FALSE, MVP);
  glUniform1f(uAxesLength, (float)N);

  glBindVertexArray(axesVAO);
  glDrawArrays(GL_LINES, 0, 6);  // 3 lines * 2 vertices each

  glDisable(GL_BLEND);
}
```

## Part 8: Cleanup (add before line 790)

Add VAO/VBO deletion:

```cpp
glDeleteBuffers(1, &fieldVBO); glDeleteVertexArrays(1, &fieldVAO);
glDeleteBuffers(1, &quadVBO); glDeleteVertexArrays(1, &quadVAO);
// ADD THESE:
glDeleteBuffers(1, &planeVBO);
glDeleteBuffers(1, &planeEBO);
glDeleteVertexArrays(1, &planeVAO);
glDeleteBuffers(1, &axesVBO);
glDeleteVertexArrays(1, &axesVAO);
```
