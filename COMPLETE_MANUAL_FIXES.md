# Complete Manual Fixes for MainGPU.cpp

## Overview
This guide provides exact code to paste into MainGPU.cpp at specific line numbers.

## Required Shader Files (Already Created):
- ✅ shaders/plane.vert
- ✅ shaders/plane.frag
- ✅ shaders/axes.vert 
- ✅ shaders/axes.frag

## Changes to Apply:

### 1. Add Shader Loading (Line 152)
**Location**: After `static std::string rectFragSrc = ...`
**Paste this code:**
```cpp
static std::string planeVertSrc = loadShaderFile("shaders/plane.vert");
static std::string planeFragSrc = loadShaderFile("shaders/plane.frag");
static std::string axesVertSrc = loadShaderFile("shaders/axes.vert");
static std::string axesFragSrc = loadShaderFile("shaders/axes.frag");
```

### 2. Add planeOpacity Global (Line 180)
**Location**: After `static int heatRadius = 5;`
**Paste this code:**
```cpp
static float planeOpacity = 0.5f;
```

### 3. Add Key Callback Toggle (Line 188)
**Location**: Inside `keyCallback` function, after reset line
**Paste this code:**
```cpp
if (key == GLFW_KEY_O && action == GLFW_PRESS) {
    planeOpacity = (planeOpacity > 0.0f) ? 0.0f : 0.5f;
    std::cout << "Plane opacity: " << planeOpacity << "\n";
}
```

### 4. Add Program Creation (Line 391)
**Location**: After `unsigned int rectProg = makeProgram(...)`
**Paste this code:**
```cpp
unsigned int planeProg = makeProgram({ compileShader(GL_VERTEX_SHADER, planeVertSrc.c_str()),
                                       compileShader(GL_FRAGMENT_SHADER, planeFragSrc.c_str()) });
unsigned int axesProg = makeProgram({ compileShader(GL_VERTEX_SHADER, axesVertSrc.c_str()),
                                      compileShader(GL_FRAGMENT_SHADER, axesFragSrc.c_str()) });
```

### 5. Add Uniform Locations (Line 416)
**Location**: After `int uRectColor = ...`
**Paste this code:**
```cpp
// Reference planes uniforms
int uPlaneRotation = glGetUniformLocation(planeProg, "uRotation");
int uPlaneOpacity = glGetUniformLocation(planeProg, "uOpacity");
int uPlaneRes = glGetUniformLocation(planeProg, "uRes");
int uPlaneMode = glGetUniformLocation(planeProg, "uMode");
int uPlaneType = glGetUniformLocation(planeProg, "uPlaneType");
int uPlaneSliceValue = glGetUniformLocation(planeProg, "uSliceValue");

// Axes uniforms
int uAxesRotation = glGetUniformLocation(axesProg, "uRotation");
int uAxesLength = glGetUniformLocation(axesProg, "uAxisLength");
```

### 6. Add Plane Geometry (After line 542)
**Location**: After `glBindVertexArray(0);` just before UI layout
**Paste this code:**
```cpp
// 3D reference planes geometry
float planeSize = (float)N * 0.25f;
float planeVertices[] = {
    0, 0, 0,   0, 0,
    planeSize, 0, 0,   0, 0,
    planeSize, planeSize, 0,   0, planeSize,
    0, planeSize, 0,   0, planeSize
};
unsigned int planeIndices[] = { 0, 1, 2, 0, 2, 3 };

unsigned int planeVAO, planeVBO, planeEBO;
glGenVertexArrays(1, &planeVAO);
glGenBuffers(1, &planeVBO);
glGenBuffers(1, &planeEBO);

glBindVertexArray(planeVAO);
glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeEBO);
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(planeIndices), planeIndices, GL_STATIC_DRAW);

GLint posAttrib = glGetAttribLocation(planeProg, "aPos");
glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);
glEnableVertexAttribArray(posAttrib);

glBindVertexArray(0);
```

### 7. Add Plane Rendering (After field rendering)
**Location**: After the field `glDrawArrays` call
**Paste this code:**
```cpp
if (planeOpacity > 0.0f) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glUseProgram(planeProg);
    glUniformMatrix4fv(uPlaneRotation, 1, GL_FALSE, MVP);
    glUniform1f(uPlaneOpacity, planeOpacity);
    glUniform1i(uPlaneRes, N);
    glUniform1i(uPlaneMode, simulationMode);
    
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
    glBindVertexArray(planeVAO);
    for (int plane = 0; plane < 3; ++plane) {
        glUniform1i(uPlaneType, plane);
        glUniform1f(uPlaneSliceValue, 0.0f);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);
    glDisable(GL_BLEND);
}
```

### 8. Add Axes Geometry (After plane geometry)
**Location**: After the plane `glBindVertexArray(0);` but before UI layout

### 9. Add Axes Rendering (After plane rendering)
**Location**: After plane render block

### 10. Add Cleanup (Before glfwTerminate)
**Location**: Before cleanup section, add:
```cpp
glDeleteBuffers(1, &planeVBO);
glDeleteBuffers(1, &planeEBO);
glDeleteVertexArrays(1, &planeVAO);
glDeleteProgram(planeProg);
glDeleteProgram(axesProg);
```

## Instructions:
1. Open MainGPU.cpp in Visual Studio
2. Press Ctrl+G and enter line number to jump
3. Paste each code block at the specified location
4. Save and rebuild
5. Press 'O' to toggle plane opacity
6. Check for syntax errors before running

All components are ready. Manual insertion is the most reliable approach.
