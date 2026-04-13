#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

uniform mat4 uRotation;
uniform float uYOffset;

out vec2 vTexCoord;

void main()
{
    vec3 position = vec3(aPos.x, uYOffset, aPos.y);
    gl_Position = uRotation * vec4(position, 1.0);
    vTexCoord = aTexCoord;
}
