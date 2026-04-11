#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 uRotation;
uniform float uAxisLength;

out vec3 vColor;

void main()
{
    vec3 pos = aPos * uAxisLength;
    gl_Position = uRotation * vec4(pos, 1.0);
    vColor = aColor;
}
