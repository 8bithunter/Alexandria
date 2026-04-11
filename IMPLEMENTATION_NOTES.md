# Integration Steps for Myself

Since I need to do this carefully and the Edit tool is having matching issues, here's my plan:

1. The shader loading has duplicate lines 156-159 that need to be removed
2. Need to add planeOpacity global variable  
3. Need to add key callback for opacity toggle (GLFW_KEY_O)
4. Need to add program creation for planeProg and axesProg
5. Need to add uniform location caching
6. Need to generate geometry for planes and axes
7. Need to add rendering code before UI overlay
8. Need to add cleanup code

Given the Edit tool continues to fail with string matching, let me use a different approach - I'll create a comprehensive patch file with line numbers showing exactly what to change.

Key changes needed:

**Global Variable Addition (after line 179):**
```cpp
static float planeOpacity = 0.5f;
```

**Program Creation Addition (after line 382):**
```cpp
unsigned int planeProg = makeProgram({ compileShader(GL_VERTEX_SHADER, planeVertSrc.c_str()),
                                       compileShader(GL_FRAGMENT_SHADER, planeFragSrc.c_str()) });
unsigned int axesProg = makeProgram({ compileShader(GL_VERTEX_SHADER, axesVertSrc.c_str()),
                                      compileShader(GL_FRAGMENT_SHADER, axesFragSrc.c_str()) });
```

**Uniform Location Addition (after line 406):**
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