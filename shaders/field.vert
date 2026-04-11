#version 430 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in uint aIdx;
#define STRIDE 10
layout(std430, binding = 0) readonly buffer Field { float field[]; };
uniform mat4 uRotation;
uniform int uMode;
uniform int uRes;
out vec4 vColor;

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

float cellHeight(int idx)
{
    idx = clamp(idx, 0, uRes*uRes - 1);
    if (uMode >= 2) {
        float re = field[idx*STRIDE + 0];
        float im = field[idx*STRIDE + 1];
        return atan(sqrt(re*re + im*im) * 0.02) * INV_PI;
    } else {
        return atan(field[idx*STRIDE] * 0.02) * INV_PI;
    }
}

void main()
{
    int id = int(aIdx);
    int ix = id % uRes;
    int iy = id / uRes;

    float hr = cellHeight(iy*uRes + min(ix+1, uRes-1));
    float hl = cellHeight(iy*uRes + max(ix-1, 0 ));
    float hu = cellHeight(min(iy+1, uRes-1)*uRes + ix );
    float hd = cellHeight(max(iy-1, 0 )*uRes + ix );
    float zScale = 3.0;
    vec3 tx = normalize(vec3(2.0, 0.0, (hr-hl)*zScale));
    vec3 ty = normalize(vec3(0.0, 2.0, (hu-hd)*zScale));
    vec3 N = normalize(cross(tx, ty));

    float height, hue, value, alpha;
    if (uMode >= 2) {
        float re = field[id*STRIDE + 0];
        float im = field[id*STRIDE + 1];
        float mag = sqrt(re*re + im*im);
        if (uMode == 2)
        {
            height = atan(2 * mag)  * INV_PI - 0.5;
            value = 2 * atan(2 * mag)  * INV_PI;
            hue = atan(-im, -re) / (2 * 3.14159265) + 0.5;
        }
        else
        {
            height = atan(mag * 0.02) * INV_PI - 0.25;
            value = 1.0;
            hue = atan(im, re) / (2.0*PI) + 0.5;
        }
        alpha = 1.0;
    }
    else
    {
        float fx = field[id*STRIDE];
        height = atan( fx*0.02)  * INV_PI - 0.5;
        hue = atan(-fx*0.02)  * INV_PI + 0.5;
        alpha = 1.0;
        value = 1.0;
    }

    gl_Position = uRotation * vec4(aPos, height, 1.0);

    vec3 lightDir = normalize(vec3(-0.4, 0.6, 1.0));
    float light = 0.25 + 0.75 * abs(dot(N, lightDir));
    vColor = vec4(hueToRgb(hue) * light * value, alpha);
}
