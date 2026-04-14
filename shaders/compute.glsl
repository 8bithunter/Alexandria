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
uniform float uDensity;
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
        // Store advected velocity as intermediate result
        outData[base+FX] = newVx;
        outData[base+FY] = newVy;
        outData[base+FZ] = 0.0;
        outData[base+VX] = 0; outData[base+VY] = 0; outData[base+VZ] = 0;
        outData[base+AX] = 0; outData[base+AY] = 0; outData[base+AZ] = 0;
        outData[base+9] = 0.0;
    }
    else if (uMode == 4) {
        // Compute divergence: ∇·u = ∂u/∂x + ∂v/∂y
        // Store in FZ component (component 2)
        float h = 2.0 / float(uRes - 1);
        float inv2h = 0.5 / h;

        float right_vx = get(r, FX);
        float left_vx = get(l, FX);
        float up_vy = get(u, FY);
        float down_vy = get(d, FY);

        float divergence = (right_vx - left_vx) * inv2h + (up_vy - down_vy) * inv2h;

        // Copy velocity to output and store divergence
        outData[base+FX] = get(c, FX);
        outData[base+FY] = get(c, FY);
        outData[base+FZ] = divergence;
        outData[base+VX] = 0; outData[base+VY] = 0; outData[base+VZ] = 0;
        outData[base+AX] = 0; outData[base+AY] = 0; outData[base+AZ] = 0;
        outData[base+9] = 0.0;
    }
    else if (uMode == 5) {
        // Jacobi iteration for pressure solve: ∇²p = divergence
        // Stores pressure in VZ component (component 5) during iterations
        float div = get(c, FZ);
        float h = 2.0 / float(uRes - 1);

        // Get neighbor pressures (stored in VZ)
        float p_right = get(r, VZ);
        float p_left = get(l, VZ);
        float p_up = get(u, VZ);
        float p_down = get(d, VZ);

        // Jacobi update: p_new = (p_left + p_right + p_up + p_down - div * h²) / 4
        // The MINUS sign is critical - we solve ∇²p = divergence, not ∇²p = -divergence
        float p_new = (p_left + p_right + p_up + p_down - div * h * h) * 0.25;

        // Copy velocity and divergence, update only pressure component
        outData[base+FX] = get(c, FX);
        outData[base+FY] = get(c, FY);
        outData[base+FZ] = get(c, FZ);
        outData[base+VX] = get(c, VX);
        outData[base+VY] = get(c, VY);
        outData[base+VZ] = p_new;
        outData[base+AX] = get(c, AX);
        outData[base+AY] = get(c, AY);
        outData[base+AZ] = get(c, AZ);
        outData[base+9] = 0.0;
    }
    else if (uMode == 6) {
        // Pressure projection: u_final = u_intermediate - Δt * (1/ρ) ∇p
        // The factor dt/ρ scales the pressure gradient to ensure stability
        float h = 2.0 / float(uRes - 1);
        float inv2h = -uDt / (2.0 * h * uDensity); // Negative because we subtract

        // Get pressure gradient
        float p_right = get(r, VZ);
        float p_left = get(l, VZ);
        float p_up = get(u, VZ);
        float p_down = get(d, VZ);

        float dpx = (p_right - p_left) * inv2h;
        float dpy = (p_up - p_down) * inv2h;

        // Apply pressure correction with proper dt scaling
        float vx = get(c, FX) + dpx;
        float vy = get(c, FY) + dpy;

        // Clamp to prevent blow-up
        const float MAX_VEL = 10.0;
        vx = clamp(vx, -MAX_VEL, MAX_VEL);
        vy = clamp(vy, -MAX_VEL, MAX_VEL);

        outData[base+FX] = vx;
        outData[base+FY] = vy;
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
