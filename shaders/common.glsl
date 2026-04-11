#ifndef COMMON_GLSL
#define COMMON_GLSL

// Field buffer layout constants
#define STRIDE 10

// Mathematical constants
const float PI = 3.14159265359;
const float INV_PI = 0.31830988618;

// Convert hue to RGB color
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

#endif
