#version 430 core

#define STRIDE 10

layout(std430, binding = 0) readonly buffer Field {
    float field[];
};

in vec2 vTexCoord;

uniform int uRes;
uniform int uMode;
uniform float uOpacity;

out vec4 FragColor;

const float PI = 3.14159265359;
const float INV_PI = 0.31830988618;

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

void main()
{
    int ix = int(vTexCoord.x * float(uRes));
    int iy = int(vTexCoord.y * float(uRes));
    ix = clamp(ix, 0, uRes - 1);
    iy = clamp(iy, 0, uRes - 1);
    int idx = iy * uRes + ix;

    vec3 color;
    if (uMode >= 2) {
        float re = field[idx * STRIDE + 0];
        float im = field[idx * STRIDE + 1];
        float mag = sqrt(re * re + im * im);
        float height = atan(mag * 0.02) * INV_PI;
        color = hueToRgb(height);
    } else {
        float fx = field[idx * STRIDE];
        float height = atan(fx * 0.02) * INV_PI + 0.5;
        color = hueToRgb(clamp(height, 0.0, 1.0));
    }

    FragColor = vec4(color * 0.7, uOpacity);
}
