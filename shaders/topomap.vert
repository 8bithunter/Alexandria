#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

uniform mat4 uRotation;
uniform float uZOffset;  // Position along Z-axis (blue axis)

out vec2 vTexCoord;

void main()
{
    // Map to XY plane (varying X and Y), fixed Z = uZOffset
    vec3 position = vec3(aPos.x, aPos.y, uZOffset);
    gl_Position = uRotation * vec4(position, 1.0);
    vTexCoord = aTexCoord;
}
