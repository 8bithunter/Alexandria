#version 430 core
layout(std430, binding = 0) readonly buffer Field { float field[]; };
#define STRIDE 10

in vec2 vTexCoord;
in float vPlaneType;
in float vSliceCoord;

uniform int uRes;
uniform int uMode;
uniform float uOpacity;
uniform float uSliceValue; // For slice planes

out vec4 FragColor;

vec3 hueToRgb(float h)
{
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

// Contour slice function for bottom plane
float getContourValue(int mode, float re, float im)
{
    if (mode >= 2) {
        float mag = sqrt(re*re + im*im);
        return atan(mag * 0.02) / 3.14159265;
    } else {
        return atan(re * 0.02) / 3.14159265;
    }
}

void main()
{
    vec2 uv = vTexCoord;
    uv = clamp(uv, 0.0, 1.0);

    int ix = int(uv.x * float(uRes));
    int iy = int(uv.y * float(uRes));
    ix = clamp(ix, 0, uRes-1);
    iy = clamp(iy, 0, uRes-1);

    int idx = iy * uRes + ix;
    float re = field[idx*STRIDE];

    vec3 color;
    float alpha = uOpacity;

    if (int(vPlaneType) == 0) { // XY bottom plane - with slices
        float contourValue = getContourValue(uMode, re, field[idx*STRIDE + 1]);

        // Create slice bands every 0.1 units
        int sliceBand = int(contourValue * 10.0);
        float sliceFraction = fract(contourValue * 10.0);

        // Use field value to color the slice
        float hue = (re + 1.0) * 0.5; // Map field to hue
        if (uMode >= 2) {
            float im = field[idx*STRIDE + 1];
            float mag = sqrt(re*re + im*im);
            hue = mag; // Use magnitude for hue
        }

        // Add edge highlight for slice boundaries
        float edge = smoothstep(0.8, 1.0, sliceFraction);
        color = hueToRgb(hue) * (0.8 + edge * 0.2);
    }
    else if (int(vPlaneType) == 1) { // XZ back plane - slice at current Y
        // Use same visualization as field
        float hue = (re + 1.0) * 0.5;
        color = hueToRgb(hue) * 0.7;
    }
    else { // YZ side plane - slice at current X
        float hue = (re + 1.0) * 0.5;
        color = hueToRgb(hue) * 0.7;
    }

    FragColor = vec4(color, alpha);
}
