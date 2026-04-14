#version 430 core
layout(location = 0) in float aPos; // position along axis (-1 to 1)
layout(location = 1) in float aCoord; // coordinate along axis (0 to 1)

#define STRIDE 10
layout(std430, binding = 0) readonly buffer Field {
    float field[];
};

uniform int uRes;
uniform int uMode;
uniform float uCursorX; // cursor X position (in UV coords)
uniform float uCursorY; // cursor Y position (in UV coords)
uniform float uAxisOffset; // position offset for the line (e.g., -1.0 for negative end)
uniform int uAxis; // 0 for horizontal (green axis), 1 for vertical (red axis)

out vec3 vLineColor;
out float vHeight;

const float PI = 3.14159265359;
const float INV_PI = 0.31830988618;

vec3 hueToRgb(float h) {
    float hp = h * 6.0;
    float xc = 1.0 - abs(mod(hp, 2.0) - 1.0);
    int s = int(hp) % 6;
    if (s==0) return vec3(1, xc, 0);
    else if (s==1) return vec3(xc, 1, 0);
    else if (s==2) return vec3(0, 1, xc);
    else if (s==3) return vec3(0, xc, 1);
    else if (s==4) return vec3(xc, 0, 1);
    else return vec3(1, 0, xc);
}

void main() {
    float axisCoord = aCoord; // 0 to 1 along axis
    vec3 color;
    float height;

    if (uMode >= 2) {
        // Complex field (fluid/schrodinger)
        int coordIdx;
        if (uAxis == 0) {
            // Horizontal line (along X axis) at cursor Y
            // Sample at (axisCoord, uCursorY)
            int ix = int(axisCoord * float(uRes));
            int iy = int(uCursorY * float(uRes));
            coordIdx = clamp(iy, 0, uRes - 1) * uRes + clamp(ix, 0, uRes - 1);
        } else {
            // Vertical line (along Y axis) at cursor X
            // Sample at (uCursorX, axisCoord)
            int ix = int(uCursorX * float(uRes));
            int iy = int(axisCoord * float(uRes));
            coordIdx = clamp(iy, 0, uRes - 1) * uRes + clamp(ix, 0, uRes - 1);
        }

        float re = field[coordIdx * STRIDE + 0];
        float im = field[coordIdx * STRIDE + 1];
        float mag = length(vec2(re, im));

        // Compute color using same logic as field.vert
        float hue;
        if (uMode == 2) {
            // Fluid: atan(-im, -re) / (2π) + 0.5
            hue = atan(-im, -re) / (2.0 * PI) + 0.5;
        } else {
            // Schrödinger: atan(im, re) / (2π) + 0.5
            hue = atan(im, re) / (2.0 * PI) + 0.5;
        }
        float value = (uMode == 2) ? (2.0 * atan(2.0 * mag) * INV_PI) : 1.0;
        color = hueToRgb(hue) * value;
        height = mag * 0.3;
    } else {
        // Real field (diffusion/wave)
        int coordIdx;
        if (uAxis == 0) {
            // Horizontal line at cursor Y
            int ix = int(axisCoord * float(uRes));
            int iy = int(uCursorY * float(uRes));
            coordIdx = clamp(iy, 0, uRes - 1) * uRes + clamp(ix, 0, uRes - 1);
        } else {
            // Vertical line at cursor X
            int ix = int(uCursorX * float(uRes));
            int iy = int(axisCoord * float(uRes));
            coordIdx = clamp(iy, 0, uRes - 1) * uRes + clamp(ix, 0, uRes - 1);
        }

        float fx = field[coordIdx * STRIDE];
        float hue = atan(-fx * 0.02) * INV_PI + 0.5;
        color = hueToRgb(hue);
        height = fx * 0.02;
    }

    vLineColor = color;
    vHeight = height;

    // Position the line in 3D space
    if (uAxis == 0) {
        // Horizontal line (along X axis) at Y = uAxisOffset (e.g., -1.0)
        gl_Position = vec4(aPos, uAxisOffset, 0, 1.0);
    } else {
        // Vertical line (along Y axis) at X = uAxisOffset (e.g., -1.0)
        gl_Position = vec4(uAxisOffset, aPos, 0, 1.0);
    }
}
