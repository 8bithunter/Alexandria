#version 430 core

#define STRIDE 10

layout(std430, binding = 0) readonly buffer Field {
    float field[];
};

in vec2 vTexCoord;

uniform int uRes;
uniform int uMode;
uniform float uOpacity;
uniform float uFieldMin;    // Minimum field value for scaling
uniform float uFieldMax;    // Maximum field value for scaling
uniform bool uDrawContours; // Enable contour drawing

out vec4 FragColor;

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

float sampleField(vec2 uv) {
    int ix = int(uv.x * float(uRes));
    int iy = int(uv.y * float(uRes));
    ix = clamp(ix, 0, uRes - 1);
    iy = clamp(iy, 0, uRes - 1);
    int idx = iy * uRes + ix;

    float value;
    if (uMode >= 2) {
        float re = field[idx * STRIDE + 0];
        float im = field[idx * STRIDE + 1];
        value = length(vec2(re, im));
    } else {
        value = field[idx * STRIDE];
    }

    // Normalize to [0, 1] range using min/max
    return clamp((value - uFieldMin) / (uFieldMax - uFieldMin), 0.0, 1.0);
}

void main() {
    vec2 uv = vTexCoord;

    // Fix reflection about green axis - apply 90° clockwise rotation
    // This rotates the map to proper orientation without mirroring
    uv = vec2(uv.y, uv.x);

    // Sample actual field values (before normalization)
    int ix = int(uv.x * float(uRes));
    int iy = int(uv.y * float(uRes));
    ix = clamp(ix, 0, uRes - 1);
    iy = clamp(iy, 0, uRes - 1);
    int idx = iy * uRes + ix;

    vec3 lineColor;
    float value;

    if (uMode >= 2) {
        // Complex field (fluid/schrodinger)
        float re = field[idx * STRIDE + 0];
        float im = field[idx * STRIDE + 1];
        value = length(vec2(re, im));

        // Compute color using same logic as field.vert
        float hue;
        float val;
        if (uMode == 2) {
            // Fluid: same as field.vert lines 62-66
            hue = atan(-im, -re) / (2.0 * PI) + 0.5;
            val = 2.0 * atan(2.0 * value) * INV_PI;
        } else {
            // Schrödinger: same as field.vert lines 69-72
            hue = atan(im, re) / (2.0 * PI) + 0.5;
            val = 1.0;
        }
        lineColor = hueToRgb(hue) * val;
    } else {
        // Real field (diffusion/wave)
        float fx = field[idx * STRIDE];
        value = fx;

        // Same as field.vert lines 77-82
        float hue = atan(-fx * 0.02) * INV_PI + 0.5;
        lineColor = hueToRgb(hue);
    }

    // Now get normalized value for contour detection
    float normalizedValue = clamp((value - uFieldMin) / (uFieldMax - uFieldMin), 0.0, 1.0);

    // For contours: create 20 levels
    const int NUM_CONTOURS = 20;
    float contourInterval = 1.0 / float(NUM_CONTOURS);

    // Find which contour interval we're in
    float contourIndex = normalizedValue / contourInterval;
    float lowerLevel = floor(contourIndex) * contourInterval;
    float upperLevel = lowerLevel + contourInterval;

    // Position within the interval [0, 1]
    float t = (normalizedValue - lowerLevel) / contourInterval;

    // Check if we're close to a contour line (either the lower or upper boundary)
    // Use smoothstep for anti-aliasing
    const float CONTOUR_WIDTH = 0.05; // Width of lines relative to interval
    float lowerLine = 1.0 - smoothstep(0.0, CONTOUR_WIDTH, t);
    float upperLine = smoothstep(1.0 - CONTOUR_WIDTH, 1.0, t);
    float lineIntensity = max(lowerLine, upperLine);

    // If no contour line, make fully transparent
    // If contour line, use field-value-based color
    FragColor = vec4(lineColor * 0.8, lineIntensity * uOpacity); // Slightly darker to make lines visible
}