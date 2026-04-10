#include <cmath>
#include <numbers>
#include <algorithm>
#include <cstdint>

class FieldVertex
{
private:
    // Grid-space position (fixed after construction)
    double x, y, z;

    // Packed colour
    uint8_t red = 255, green = 255, blue = 255, alpha = 255;

    // Cached spacing reciprocals — computed once from distanceBetweenVertexes
    // Declaration order matters: distanceBetweenVertexes is listed first so the
    // default-member-initialisers below see its final value.
    double distanceBetweenVertexes;
    double inv2H;   // 0.5 / h  — used by centred-difference gradient
    double invH2;   // 1 / h²   — used by Laplacian

public:
    // ── Field state ──────────────────────────────────────────────────────────
    double fieldX = 0, fieldY = 0, fieldZ = 0;
    double dxdt = 0, dydt = 0, dzdt = 0;
    double d2xdt2 = 0, d2ydt2 = 0, d2zdt2 = 0;

    // ── Neighbour pointers ───────────────────────────────────────────────────
    FieldVertex* neighbourUp, * neighbourDown,
        * neighbourLeft, * neighbourRight,
        * neighbourOut, * neighbourIn;

    // ── Constructor ──────────────────────────────────────────────────────────
    FieldVertex(double x_, double y_, double z_, double d,
        double fx = 0, double fy = 0, double fz = 0)
        : x(x_), y(y_), z(z_)
        , distanceBetweenVertexes(d)
        , inv2H(0.5 / d)
        , invH2(1.0 / (d * d))
        , fieldX(fx), fieldY(fy), fieldZ(fz)
        , neighbourUp(nullptr), neighbourDown(nullptr)
        , neighbourLeft(nullptr), neighbourRight(nullptr)
        , neighbourOut(nullptr), neighbourIn(nullptr)
    {
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    void   setZ(double newZ) { z = newZ; }
    double getZ()       const { return z; }

    void setColour(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
    {
        red = r; green = g; blue = b; alpha = a;
    }

    void getColorFloat(float& outR, float& outG, float& outB) const
    {
        constexpr float inv255 = 1.0f / 255.0f;
        outR = red * inv255;
        outG = green * inv255;
        outB = blue * inv255;
    }

    // ── Differential operators ───────────────────────────────────────────────
    // All computed directly — no caching.
    // Each result is a trivial 2-op centred difference; the cache comparisons
    // (2–7 double equality checks) cost more than just recomputing.

    // Gradient of the vector field in the X-grid direction
    // Returns ∂Fx/∂x, ∂Fy/∂x, ∂Fz/∂x
    double gradXx() const { return (neighbourRight->fieldX - neighbourLeft->fieldX) * inv2H; }
    double gradXy() const { return (neighbourRight->fieldY - neighbourLeft->fieldY) * inv2H; }
    double gradXz() const { return (neighbourRight->fieldZ - neighbourLeft->fieldZ) * inv2H; }

    // Gradient in the Y-grid direction
    double gradYx() const { return (neighbourUp->fieldX - neighbourDown->fieldX) * inv2H; }
    double gradYy() const { return (neighbourUp->fieldY - neighbourDown->fieldY) * inv2H; }
    double gradYz() const { return (neighbourUp->fieldZ - neighbourDown->fieldZ) * inv2H; }

    // Gradient in the Z-grid direction
    double gradZx() const { return (neighbourOut->fieldX - neighbourIn->fieldX) * inv2H; }
    double gradZy() const { return (neighbourOut->fieldY - neighbourIn->fieldY) * inv2H; }
    double gradZz() const { return (neighbourOut->fieldZ - neighbourIn->fieldZ) * inv2H; }

    // Divergence, curl
    double getDiv()   const { return gradXx() + gradYy() + gradZz(); }
    double getCurlX() const { return gradYz() - gradZy(); }
    double getCurlY() const { return gradZx() - gradXz(); }
    double getCurlZ() const { return gradXy() - gradYx(); }

    // Scalar Laplacians (6-point stencil)
    double getLaplacianX() const
    {
        return (neighbourRight->fieldX + neighbourLeft->fieldX +
            neighbourUp->fieldX + neighbourDown->fieldX +
            neighbourOut->fieldX + neighbourIn->fieldX -
            6.0 * fieldX) * invH2;
    }

    double getLaplacianY() const
    {
        return (neighbourRight->fieldY + neighbourLeft->fieldY +
            neighbourUp->fieldY + neighbourDown->fieldY +
            neighbourOut->fieldY + neighbourIn->fieldY -
            6.0 * fieldY) * invH2;
    }

    double getLaplacianZ() const
    {
        return (neighbourRight->fieldZ + neighbourLeft->fieldZ +
            neighbourUp->fieldZ + neighbourDown->fieldZ +
            neighbourOut->fieldZ + neighbourIn->fieldZ -
            6.0 * fieldZ) * invH2;
    }

    // ── Simulation step ───────────────────────────────────────────────────────
    void calculateddt()
    {
        //dxdt = 0.001 * getLaplacianX();

		d2xdt2 = 0.001 * getLaplacianX();
		dxdt *= 0.99;

        // dydt = 0.01 * getLaplacianY();
        // dzdt = 0.01 * getLaplacianZ();
    }

    void updateField(double dt)
    {
        // Verlet-style integration: x += v*dt + ½a*dt²
        double dt2 = 0.5 * dt * dt;
        fieldX += dxdt * dt + d2xdt2 * dt2;
        fieldY += dydt * dt + d2ydt2 * dt2;
        fieldZ += dzdt * dt + d2zdt2 * dt2;  

        dxdt += d2xdt2 * dt;
        dydt += d2ydt2 * dt;
        dzdt += d2zdt2 * dt;
    }

    // ── Visuals ───────────────────────────────────────────────────────────────
    // Maps fieldX → hue via atan (smooth saturation for large values),
    // then drives vertex Z height with the same curve so rotation shows relief.
    void updateVisuals()
    {
        // atan maps ℝ → (-π/2, π/2); dividing by π gives (-0.5, 0.5); +0.5 → [0,1)
        double hueNorm = (std::atan(fieldX * 0.02) * (1.0 / std::numbers::pi)) + 0.5;
        double hue = hueNorm * 360.0;   // [0°, 360°)

        // Full-brightness, full-saturation HSV → RGB
        // Avoids std::floor + cast + fmod by using integer truncation directly
        double hprime = hue * (1.0 / 60.0);
        int    sector = static_cast<int>(hprime) % 6;
        if (sector < 0) sector += 6;
        double frac = hprime - static_cast<int>(hprime);  // fractional part
        double xc = 1.0 - std::fabs(std::fmod(hprime, 2.0) - 1.0);

        double r1, g1, b1;
        switch (sector)
        {
        case 0: r1 = 1;  g1 = xc; b1 = 0;  break;
        case 1: r1 = xc; g1 = 1;  b1 = 0;  break;
        case 2: r1 = 0;  g1 = 1;  b1 = xc; break;
        case 3: r1 = 0;  g1 = xc; b1 = 1;  break;
        case 4: r1 = xc; g1 = 0;  b1 = 1;  break;
        case 5: r1 = 1;  g1 = 0;  b1 = xc; break;
        default:r1 = 0;  g1 = 0;  b1 = 0;  break;
        }

        constexpr double scale = 255.0;
        setColour(
            static_cast<uint8_t>(std::clamp(r1 * scale, 0.0, 255.0)),
            static_cast<uint8_t>(std::clamp(g1 * scale, 0.0, 255.0)),
            static_cast<uint8_t>(std::clamp(b1 * scale, 0.0, 255.0))
        );

        // Z height mirrors the hue curve so colour and relief are in sync
        setZ(hueNorm - 0.5);  // maps to [-0.5, +0.5]
    }
};