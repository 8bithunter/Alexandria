#!/usr/bin/env python3
"""
Script to integrate 3D axes and reference planes into MainGPU.cpp
Handles the modifications that failed in previous attempts
"""

import re
import sys

def modify_maingpu():
    file_path = "MainGPU.cpp"

    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return False

    # Fix 1: Remove duplicate shader declarations if they exist
    # Look for multiple declarations of the new shader variables
    shader_pattern = r'(static\s+std::string\s+(planeVertSrc|planeFragSrc|axesVertSrc|axesFragSrc)\s+=\s+loadShaderFile\([^)]+\);)\s*\1'
    content = re.sub(shader_pattern, r'\1', content)

    # Remove if already added (prevent duplicates)
    if 'planeVertSrc' in content and content.count('planeVertSrc') > 1:
        print("Warning: Duplicate shader declarations detected")
        # Keep only the first occurrence
        lines = content.split('\n')
        new_lines = []
        seen_shaders = set()
        for line in lines:
            if 'planeVertSrc' in line or 'planeFragSrc' in line or 'axesVertSrc' in line or 'axesFragSrc' in line:
                if line.strip() not in seen_shaders:
                    seen_shaders.add(line.strip())
                    new_lines.append(line)
                else:
                    continue  # Skip duplicate
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)

    # Fix 2: Ensure shader loading is present (if not, add it)
    if 'planeVertSrc' not in content:
        print("Adding new shader loading...")
        shader_section = r'(static\s+std::string\s+rectFragSrc\s+=\s+loadShaderFile\("shaders/rect\.frag"\);)'
        new_shaders = r'''\1
static std::string planeVertSrc = loadShaderFile("shaders/plane.vert");
static std::string planeFragSrc = loadShaderFile("shaders/plane.frag");
static std::string axesVertSrc = loadShaderFile("shaders/axes.vert");
static std::string axesFragSrc = loadShaderFile("shaders/axes.frag");'''
        content = re.sub(shader_section, new_shaders, content)

    # Fix 3: Add planeOpacity global
    if 'planeOpacity' not in content:
        print("Adding planeOpacity global...")
        opacity_pattern = r'(static\s+bool\s+resetRequested\s+=\s+false;)'
        opacity_add = r'\1\nstatic float planeOpacity = 0.5f;'
        content = re.sub(opacity_pattern, opacity_add, content)

    # Fix 4: Add key callback
    if 'GLFW_KEY_O' not in content:
        print("Adding opacity keybind...")
        key_pattern = r'(if\s*\(\s*key\s*==\s*GLFW_KEY_R\s*&&\s*action\s*==\s*GLFW_PRESS\s*\)\s*resetRequested\s*=\s*true;)'
        key_add = r'''\1
  if (key == GLFW_KEY_O && action == GLFW_PRESS) {
    planeOpacity = (planeOpacity > 0.0f) ? 0.0f : 0.5f;
    std::cout << "Plane opacity: " << planeOpacity << "\n";
  }'''
        content = re.sub(key_pattern, key_add, content)

    # Fix 5: Add program creation
    if 'planeProg' not in content:
        print("Adding program creation...")
        prog_pattern = r'(unsigned int rectProg\s*=\s*makeProgram\(\{\s*compileShader\(GL_VERTEX_SHADER, rectVertSrc\.c_str\(\)\),\s*compileShader\(GL_FRAGMENT_SHADER, rectFragSrc\.c_str\(\)\)\s*\}\);)'
        prog_add = r'''\1
  unsigned int planeProg = makeProgram({ compileShader(GL_VERTEX_SHADER, planeVertSrc.c_str()),
                                         compileShader(GL_FRAGMENT_SHADER, planeFragSrc.c_str()) });
  unsigned int axesProg = makeProgram({ compileShader(GL_VERTEX_SHADER, axesVertSrc.c_str()),
                                         compileShader(GL_FRAGMENT_SHADER, axesFragSrc.c_str()) });'''
        content = re.sub(prog_pattern, prog_add, content, flags=re.DOTALL)

    # Fix 6: Add uniform locations
    if 'uPlaneOpacity' not in content:
        print("Adding uniform locations...")
        # Find after uRectColor section
        uniform_pattern = r'(int\s+uRectColor\s*=\s*glGetUniformLocation\(rectProg.*\);)'
        uniform_add = r'''\1

  // Reference planes uniforms
  int uPlaneRotation = glGetUniformLocation(planeProg, "uRotation");
  int uPlaneOpacity = glGetUniformLocation(planeProg, "uOpacity");
  int uPlaneRes = glGetUniformLocation(planeProg, "uRes");
  int uPlaneMode = glGetUniformLocation(planeProg, "uMode");
  int uPlaneType = glGetUniformLocation(planeProg, "uPlaneType");
  int uPlaneSliceValue = glGetUniformLocation(planeProg, "uSliceValue");

  // Axes uniforms
  int uAxesRotation = glGetUniformLocation(axesProg, "uRotation");
  int uAxesLength = glGetUniformLocation(axesProg, "uAxisLength");'''
        content = re.sub(uniform_pattern, uniform_add, content)

    # Write out the modified content
    print("Writing modified MainGPU.cpp...")
    with open(file_path, 'w') as f:
        f.write(content)

    return True

if __name__ == "__main__":
    success = modify_maingpu()
    if success:
        print("Successfully modified MainGPU.cpp")
        sys.exit(0)
    else:
        print("Failed to modify MainGPU.cpp")
        sys.exit(1)
