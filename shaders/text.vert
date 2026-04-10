#version 430 core
layout(location = 0) in vec2 aPos;
uniform vec2 uOrigin;
uniform vec2 uCharSize;
uniform float uAdvance;
uniform uint uFont[22];
uniform uint uChars[16];
out vec2 vUV;
flat out uint vGlyph;
void main()
{
    vec2 cellOrigin = uOrigin + vec2(float(gl_InstanceID) * uAdvance, 0.0);
    gl_Position = vec4(cellOrigin + vec2(aPos.x*uCharSize.x, -aPos.y*uCharSize.y), 0.0, 1.0);
    vUV = aPos;
    vGlyph = uFont[uChars[gl_InstanceID]];
}
