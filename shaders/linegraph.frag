#version 430 core

in vec3 vLineColor;
in float vHeight;

uniform float uOpacity;

out vec4 FragColor;

void main() {
    // Lines are 100% opaque as requested
    FragColor = vec4(vLineColor, 1.0);
}
