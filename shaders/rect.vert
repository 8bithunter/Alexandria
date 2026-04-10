#version 430 core
layout(location = 0) in vec2 aPos;
uniform vec2 uRectOrigin;
uniform vec2 uRectSize;
void main()
{
    vec2 p = uRectOrigin + vec2(aPos.x*uRectSize.x, -aPos.y*uRectSize.y);
    gl_Position = vec4(p, 0.0, 1.0);
}
