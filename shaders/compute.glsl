#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

#define STRIDE 10
#define FX 0
#define FY 1
#define FZ 2
#define VX 3
#define VY 4
#define VZ 5
#define AX 6
#define AY 7
#define AZ 8

layout(std430, binding = 0) readonly buffer BufIn { float inData[]; };
layout(std430, binding = 1) writeonly buffer BufOut { float outData[]; };

uniform int uRes;
uniform float uInvH2;
uniform float uDiffusion;
uniform float uDt;
uniform int uMode;

float get(int cell, int comp) { return inData[cell * STRIDE + comp]; }

float laplacian(int r, int l, int u, int d, int c, int comp)
{
    return (get(r,comp)+get(l,comp)+get(u,comp)+get(d,comp) - 4.0*get(c,comp)) * uInvH2;
}

float bilinear(float fx, float fy, int comp)
{
    fx = clamp(fx, 0.0, float(uRes - 1));
    fy = clamp(fy, 0.0, float(uRes - 1));
    int x0 = int(fx), y0 = int(fy);
    int x1 = min(x0 + 1, uRes - 1);
    int y1 = min(y0 + 1, uRes - 1);
    float tx = fx - float(x0);
    float ty = fy - float(y0);
    float v00 = get(y0*uRes+x0, comp);
    float v10 = get(y0*uRes+x1, comp);
    float v01 = get(y1*uRes+x0, comp);
    float v11 = get(y1*uRes+x1, comp);
    return mix(mix(v00,v10,tx), mix(v01,v11,tx), ty);
}

void main()
{
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    if (id.x >= uRes || id.y >= uRes) return;
    int c = id.y*uRes + id.x;
    int r = id.y*uRes + (id.x+1) % uRes;
    int l = id.y*uRes + (id.x-1+uRes) % uRes;
    int u = ((id.y+1) % uRes)*uRes + id.x;
    int d = ((id.y-1+uRes) % uRes)*uRes + id.x;
    int base = c * STRIDE;

    if (uMode == 0) {
        float vx = uDiffusion * laplacian(r,l,u,d,c,FX);
        float vy = uDiffusion * laplacian(r,l,u,d,c,FY);
        float vz = uDiffusion * laplacian(r,l,u,d,c,FZ);
        outData[base+FX] = get(c,FX) + vx*uDt;
        outData[base+FY] = get(c,FY) + vy*uDt;
        outData[base+FZ] = get(c,FZ) + vz*uDt;
        outData[base+VX] = vx; outData[base+VY] = vy; outData[base+VZ] = vz;
        outData[base+AX] = 0; outData[base+AY] = 0; outData[base+AZ] = 0;
        outData[base+9] = 0.0;
    }
    else if (uMode == 1) {
        float ax = uDiffusion * laplacian(r,l,u,d,c,FX);
        float ay = uDiffusion * laplacian(r,l,u,d,c,FY);
        float az = uDiffusion * laplacian(r,l,u,d,c,FZ);
        float vx = get(c,VX)*0.999 + ax*uDt;
        float vy = get(c,VY) + ay*uDt;
        float vz = get(c,VZ) + az*uDt;
        outData[base+FX] = get(c,FX) + vx*uDt;
        outData[base+FY] = get(c,FY) + vy*uDt;
        outData[base+FZ] = get(c,FZ) + vz*uDt;
        outData[base+VX] = vx; outData[base+VY] = vy; outData[base+VZ] = vz;
        outData[base+AX] = ax; outData[base+AY] = ay; outData[base+AZ] = az;
        outData[base+9] = 0.0;
    }
    else if (uMode == 2) {
        float h = 2.0 / float(uRes - 1);
        float vx = get(c, FX);
        float vy = get(c, FY);
        float px = float(id.x) - vx * uDt / h;
        float py = float(id.y) - vy * uDt / h;
        float newVx = bilinear(px, py, FX);
        float newVy = bilinear(px, py, FY);
        newVx += uDiffusion * laplacian(r,l,u,d,c,FX) * uDt;
        newVy += uDiffusion * laplacian(r,l,u,d,c,FY) * uDt;
        outData[base+FX] = newVx;
        outData[base+FY] = newVy;
        outData[base+FZ] = 0.0;
        outData[base+VX] = 0; outData[base+VY] = 0; outData[base+VZ] = 0;
        outData[base+AX] = 0; outData[base+AY] = 0; outData[base+AZ] = 0;
        outData[base+9] = 0.0;
    }
    else if (uMode == 3) {
        float lapIm = laplacian(r,l,u,d,c,FY);
        outData[base+FX] = get(c,FX) - 0.5*lapIm*uDt;
        outData[base+FY] = get(c,FY);
        outData[base+FZ] = get(c,FZ);
        outData[base+VX] = 0; outData[base+VY] = 0; outData[base+VZ] = 0;
        outData[base+AX] = 0; outData[base+AY] = 0; outData[base+AZ] = 0;
        outData[base+9] = 0.0;
    }
    else {
        float lapRe = laplacian(r,l,u,d,c,FX);
        outData[base+FX] = get(c,FX);
        outData[base+FY] = get(c,FY) + 0.5*lapRe*uDt;
        outData[base+FZ] = get(c,FZ);
        outData[base+VX] = 0; outData[base+VY] = 0; outData[base+VZ] = 0;
        outData[base+AX] = 0; outData[base+AY] = 0; outData[base+AZ] = 0;
        outData[base+9] = 0.0;
    }
}
