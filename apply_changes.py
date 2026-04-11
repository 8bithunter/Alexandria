#!/usr/bin/env python3
"""
Apply integration changes to MainGPU.cpp for 3D axes and planes
"""

import re

def apply_changes():
    """Apply all changes to MainGPU.cpp"""

    # Read the file
    with open('MainGPU.cpp', 'r') as f:
        content = f.read()

    # 1. Add #include <algorithm>
    if '<algorithm>' not in content:
        content = content.replace(
            '#include <cstdint>',
            '#include <cstdint>\n#include <algorithm>'
        )

    # 2. Add shader loading after rectFragSrc
    if 'planeVertSrc' not in content:
        content = content.replace(
            'static std::string rectFragSrc = loadShaderFile("shaders/rect.frag");',
            'static std::string rectFragSrc = loadShaderFile("shaders/rect.frag");\n'
            'static std::string planeVertSrc = loadShaderFile("shaders/plane.vert");\n'
            'static std::string planeFragSrc = loadShaderFile("shaders/plane.frag");\n'
            'static std::string axesVertSrc = loadShaderFile("shaders/axes.vert");\n'
            'static std::string axesFragSrc = loadShaderFile("shaders/axes.frag");'
        )

    # 3. Add planeOpacity global
    if 'planeOpacity' not in content:
        content = content.replace(
            'static int heatRadius = 5;\nstatic float zoom = 2.5f;',
            'static int heatRadius = 5;\nstatic float planeOpacity = 0.5f;\nstatic float zoom = 2.5f;'
        )

    # 4. Add keyCallback opacity toggle
    if 'GLFW_KEY_O' not in content:
        content = content.replace(
            'if (key == GLFW_KEY_R && action == GLFW_PRESS) resetRequested = true;\n}',
            'if (key == GLFW_KEY_R && action == GLFW_PRESS) resetRequested = true;\n'
            '  if (key == GLFW_KEY_O && action == GLFW_PRESS) {\n'
            '    planeOpacity = (planeOpacity > 0.0f) ? 0.0f : 0.5f;\n'
            '    std::cout << "Plane opacity: " << planeOpacity << "\\n";\n'
            '  }\n'
            '}'
        )

    # 5. Add program creation
    if 'planeProg' not in content:
        program_section = re.search(
            r'(unsigned int rectProg = makeProgram\(\{ compileShader\(GL_VERTEX_SHADER, rectVertSrc\.c_str\(\)\),\s*compileShader\(GL_FRAGMENT_SHADER, rectFragSrc\.c_str\(\)\) \}\);)',
            content, re.MULTILINE
        )
        if program_section:
            insert_pos = program_section.end()
            program_code = '''\n  unsigned int planeProg = makeProgram({ compileShader(GL_VERTEX_SHADER, planeVertSrc.c_str()),
                                         compileShader(GL_FRAGMENT_SHADER, planeFragSrc.c_str()) });
  unsigned int axesProg = makeProgram({ compileShader(GL_VERTEX_SHADER, axesVertSrc.c_str()),
                                        compileShader(GL_FRAGMENT_SHADER, axesFragSrc.c_str()) });'''
            content = content[:insert_pos] + program_code + content[insert_pos:]

    # 6. Add uniform locations
    if 'uPlaneOpacity' not in content:
        uniform_section = re.search(
            r'(int uRectColor = glGetUniformLocation\(rectProg, "uRectColor"\);)',
            content
        )
        if uniform_section:
            insert_pos = uniform_section.end()
            uniform_code = '''\n\n  // Reference planes uniforms
  int uPlaneRotation = glGetUniformLocation(planeProg, "uRotation");
  int uPlaneOpacity = glGetUniformLocation(planeProg, "uOpacity");
  int uPlaneRes = glGetUniformLocation(planeProg, "uRes");
  int uPlaneMode = glGetUniformLocation(planeProg, "uMode");
  int uPlaneType = glGetUniformLocation(planeProg, "uPlaneType");
  int uPlaneSliceValue = glGetUniformLocation(planeProg, "uSliceValue");
\n  // Axes uniforms
  int uAxesRotation = glGetUniformLocation(axesProg, "uRotation");
  int uAxesLength = glGetUniformLocation(axesProg, "uAxisLength");'''
            content = content[:insert_pos] + uniform_code + content[insert_pos:]

    # 7. Add plane geometry
    if 'planeVertices' not in content:
        geom_section = re.search(
            r'(glBindVertexArray\(0\);\s+// ── UI layout ─────────────────────────────────────────────────────────────)',
            content
        )
        if geom_section:
            insert_pos = geom_section.start()
            geom_code = '''\n\n  // 3D reference planes geometry\n  // XY bottom plane (z=0) - 0,0,0 to RES,RES,0\n  float planeSize = (float)N;\n  float planeVertices[] = {\n    0, 0, 0,   0, 0,\n    planeSize, 0, 0,   0, 0,\n    planeSize, planeSize, 0,   0, planeSize,\n    0, planeSize, 0,   0, planeSize\n  };\n  unsigned int planeIndices[] = {0, 1, 2, 0, 2, 3};\n\n  unsigned int planeVAO, planeVBO, planeEBO;\n  glGenVertexArrays(1, &planeVAO);\n  glGenBuffers(1, &planeVBO);\n  glGenBuffers(1, &planeEBO);\n\n  glBindVertexArray(planeVAO);\n  glBindBuffer(GL_ARRAY_BUFFER, planeVBO);\n  glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);\n  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeEBO);\n  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(planeIndices), planeIndices, GL_STATIC_DRAW);\n\n  GLint posAttrib = glGetAttribLocation(planeProg, "aPos");\n  glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);\n  glEnableVertexAttribArray(posAttrib);\n\n  glBindVertexArray(0);'''
            content = content[:insert_pos] + geom_code + content[insert_pos:]

    # Write the file
    with open('MainGPU.cpp', 'w') as f:
        f.write(content)
    print("Successfully applied changes to MainGPU.cpp")

if __name__ == "__main__":
    apply_changes()
