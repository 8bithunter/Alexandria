# MainGPU.cpp Fix Instructions

Due to persistent tool issues with line endings, here's how to manually complete the integration:

## Changes to Apply:

### 1. Add Includes
After line 32, ensure: `#include <algorithm>`

### 2. Add Shader Loading
After line 151 (rectFragSrc loading), add:
```cpp
static std::string planeVertSrc = loadShaderFile("shaders/plane.vert");
static std::string planeFragSrc = loadShaderFile("shaders/plane.frag");
static std::string axesVertSrc = loadShaderFile("shaders/axes.vert");
static std::string axesFragSrc = loadShaderFile("shaders/axes.frag");
```

### 3. Add Global Variable
After line 179 (heatRadius), add:
```cpp
static float planeOpacity = 0.5f;
```

### 4. Add Key Callback
Inside keyCallback function, after reset line, add:
```cpp
if (key == GLFW_KEY_O && action == GLFW_PRESS) {
    planeOpacity = (planeOpacity > 0.0f) ? 0.0f : 0.5f;
    std::cout << "Plane opacity: " << planeOpacity << "\n";
}
```

### 5. Add Program Creation
After rectProg creation, add:
```cpp
unsigned int planeProg = makeProgram({ compileShader(GL_VERTEX_SHADER, planeVertSrc.c_str()),
                                       compileShader(GL_FRAGMENT_SHADER, planeFragSrc.c_str()) });
unsigned int axesProg = makeProgram({ compileShader(GL_VERTEX_SHADER, axesVertSrc.c_str()),
                                      compileShader(GL_FRAGMENT_SHADER, axesFragSrc.c_str()) });
```

### 6. Add Uniform Locations
After uRectColor uniform, add:
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

### 7. Add Plane Geometry
Before UI layout section, add:
```cpp
// 3D reference planes geometry
float planeSize = (float)N;
float planeVertices[] = {
    0, 0, 0,   0, 0,
    planeSize, 0, 0,   0, 0,
    planeSize, planeSize, 0,   0, planeSize,
    0, planeSize, 0,   0, planeSize
};
unsigned int planeIndices[] = {0, 1, 2, 0, 2, 3};

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

### 8. Add Plane Rendering
After field rendering, add:
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
    glUniform1i(uPlaneType, 0);
    glUniform1f(uPlaneSliceValue, 0.0f);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    
    glBindVertexArray(0);
    glDisable(GL_BLEND);
}
```

### 9. Add Cleanup
Before `glfwTerminate()`, add:
```cpp
glDeleteBuffers(1, &planeVBO);
glDeleteBuffers(1, &planeEBO);
glDeleteVertexArrays(1, &planeVAO);
glDeleteProgram(planeProg);
glDeleteProgram(axesProg);
```

## After Applying All Changes:
Build in Visual Studio and verify no compilation errors.
