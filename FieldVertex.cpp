using namespace std;
#include <cmath>
#include <numbers>
#include <algorithm>

class FieldVertex
{
private:
	double x, y, z; // we should never have to look at or change the cords
	short int red, green, blue, alpha; // we should never have to look at the colours

	double distanceBetweenVertexes = 1;

public:   
	void setZ(double z) { this->z = z; } // we might want to change z, but not x and y for graphing a 2d thing
	void setColour(short int r, short int g, short int b, short int a) { red = r; green = g; blue = b; alpha = a; } // we might want to change the color of the vertex, but not its position

    // Accessor: get color as normalized floats in [0,1]
    void getColorFloat(float &outR, float &outG, float &outB) const
    {
        outR = static_cast<float>(red) / 255.0f;
        outG = static_cast<float>(green) / 255.0f;
        outB = static_cast<float>(blue) / 255.0f;
    }

	double fieldX = 0, fieldY = 0, fieldZ = 0;
	double dxdt = 0, dydt = 0, dzdt = 0;
	double d2xdt2 = 0, d2ydt2 = 0, d2zdt2 = 0;
	double gradX = 0, gradY = 0, gradZ = 0;
	double div = 0;
	double curlX = 0, curlY = 0, curlZ = 0;
	double laplacianX = 0, laplacianY = 0, laplacianZ = 0;

	FieldVertex(double x, double y, double z, double d, double fx = 0, double fy = 0, double fz = 0)
		: x(x), y(y), z(z), distanceBetweenVertexes(d),
		  red(255), green(255), blue(255), alpha(255),
		  fieldX(fx), fieldY(fy), fieldZ(fz),
		  neighbourUp(nullptr), neighbourDown(nullptr), neighbourLeft(nullptr), neighbourRight(nullptr), neighbourOut(nullptr), neighbourIn(nullptr)
	{}



	FieldVertex *neighbourUp, *neighbourDown, *neighbourLeft, *neighbourRight, *neighbourOut, *neighbourIn; // pointers otherwise poopy

    double invH = 1.0 / distanceBetweenVertexes;
    double inv2H = 0.5 * invH;
    double invH2 = invH * invH;

    void calculateGrad()
    {
        if (neighbourRight && neighbourLeft)
            gradX = (neighbourRight->fieldX - neighbourLeft->fieldX) * inv2H;
        else if (neighbourRight)
            gradX = (neighbourRight->fieldX - fieldX) * invH;
        else if (neighbourLeft)
            gradX = (fieldX - neighbourLeft->fieldX) * invH;
        else
            gradX = 0.0;

        if (neighbourUp && neighbourDown)
            gradY = (neighbourUp->fieldY - neighbourDown->fieldY) * inv2H;
        else if (neighbourUp)
            gradY = (neighbourUp->fieldY - fieldY) * invH;
        else if (neighbourDown)
            gradY = (fieldY - neighbourDown->fieldY) * invH;
        else
            gradY = 0.0;

        if (neighbourOut && neighbourIn)
            gradZ = (neighbourOut->fieldZ - neighbourIn->fieldZ) * inv2H;
        else if (neighbourOut)
            gradZ = (neighbourOut->fieldZ - fieldZ) * invH;
        else if (neighbourIn)
            gradZ = (fieldZ - neighbourIn->fieldZ) * invH;
        else
            gradZ = 0.0;
    }


    void calculateDiv()
    {
        double dFx_dx = 0.0, dFy_dy = 0.0, dFz_dz = 0.0;

        if (neighbourRight && neighbourLeft)
            dFx_dx = (neighbourRight->fieldX - neighbourLeft->fieldX) * inv2H;
        else if (neighbourRight)
            dFx_dx = (neighbourRight->fieldX - fieldX) * invH;
        else if (neighbourLeft)
            dFx_dx = (fieldX - neighbourLeft->fieldX) * invH;

        if (neighbourUp && neighbourDown)
            dFy_dy = (neighbourUp->fieldY - neighbourDown->fieldY) * inv2H;
        else if (neighbourUp)
            dFy_dy = (neighbourUp->fieldY - fieldY) * invH;
        else if (neighbourDown)
            dFy_dy = (fieldY - neighbourDown->fieldY) * invH;

        if (neighbourOut && neighbourIn)
            dFz_dz = (neighbourOut->fieldZ - neighbourIn->fieldZ) * inv2H;
        else if (neighbourOut)
            dFz_dz = (neighbourOut->fieldZ - fieldZ) * invH;
        else if (neighbourIn)
            dFz_dz = (fieldZ - neighbourIn->fieldZ) * invH;

        div = dFx_dx + dFy_dy + dFz_dz;
    }


    void calculateCurl()
    {
        double dFz_dy = 0.0, dFy_dz = 0.0, dFx_dz = 0.0, dFz_dx = 0.0, dFy_dx = 0.0, dFx_dy = 0.0;

        if (neighbourUp && neighbourDown)
            dFz_dy = (neighbourUp->fieldZ - neighbourDown->fieldZ) * inv2H;

        if (neighbourOut && neighbourIn)
            dFy_dz = (neighbourOut->fieldY - neighbourIn->fieldY) * inv2H;

        if (neighbourOut && neighbourIn)
            dFx_dz = (neighbourOut->fieldX - neighbourIn->fieldX) * inv2H;

        if (neighbourRight && neighbourLeft)
            dFz_dx = (neighbourRight->fieldZ - neighbourLeft->fieldZ) * inv2H;

        if (neighbourRight && neighbourLeft)
            dFy_dx = (neighbourRight->fieldY - neighbourLeft->fieldY) * inv2H;

        if (neighbourUp && neighbourDown)
            dFx_dy = (neighbourUp->fieldX - neighbourDown->fieldX) * inv2H;

        curlX = dFz_dy - dFy_dz;
        curlY = dFx_dz - dFz_dx;
        curlZ = dFy_dx - dFx_dy;
    }

    void calculateLaplacian()
    {
        double sumX =
            (neighbourRight ? neighbourRight->fieldX : fieldX) +
            (neighbourLeft ? neighbourLeft->fieldX : fieldX) +
            (neighbourUp ? neighbourUp->fieldX : fieldX) +
            (neighbourDown ? neighbourDown->fieldX : fieldX) +
            (neighbourOut ? neighbourOut->fieldX : fieldX) +
            (neighbourIn ? neighbourIn->fieldX : fieldX);

        double sumY =
            (neighbourRight ? neighbourRight->fieldY : fieldY) +
            (neighbourLeft ? neighbourLeft->fieldY : fieldY) +
            (neighbourUp ? neighbourUp->fieldY : fieldY) +
            (neighbourDown ? neighbourDown->fieldY : fieldY) +
            (neighbourOut ? neighbourOut->fieldY : fieldY) +
            (neighbourIn ? neighbourIn->fieldY : fieldY);

        double sumZ =
            (neighbourRight ? neighbourRight->fieldZ : fieldZ) +
            (neighbourLeft ? neighbourLeft->fieldZ : fieldZ) +
            (neighbourUp ? neighbourUp->fieldZ : fieldZ) +
            (neighbourDown ? neighbourDown->fieldZ : fieldZ) +
            (neighbourOut ? neighbourOut->fieldZ : fieldZ) +
            (neighbourIn ? neighbourIn->fieldZ : fieldZ);

        /*
        double sumX =
            (neighbourRight ? neighbourRight->fieldX : 0) +
            (neighbourLeft ? neighbourLeft->fieldX : 0) +
            (neighbourUp ? neighbourUp->fieldX : 0) +
            (neighbourDown ? neighbourDown->fieldX : 0) +
            (neighbourOut ? neighbourOut->fieldX : 0) +
            (neighbourIn ? neighbourIn->fieldX : 0);

        double sumY =
            (neighbourRight ? neighbourRight->fieldY : 0) +
            (neighbourLeft ? neighbourLeft->fieldY : 0) +
            (neighbourUp ? neighbourUp->fieldY : 0) +
            (neighbourDown ? neighbourDown->fieldY : 0) +
            (neighbourOut ? neighbourOut->fieldY : 0) +
            (neighbourIn ? neighbourIn->fieldY : 0);

        double sumZ =
            (neighbourRight ? neighbourRight->fieldZ : 0) +
            (neighbourLeft ? neighbourLeft->fieldZ : 0) +
            (neighbourUp ? neighbourUp->fieldZ : 0) +
            (neighbourDown ? neighbourDown->fieldZ : 0) +
            (neighbourOut ? neighbourOut->fieldZ : 0) +
            (neighbourIn ? neighbourIn->fieldZ : 0);
            */

        laplacianX = (sumX - 6.0 * fieldX) * invH2;
        laplacianY = (sumY - 6.0 * fieldY) * invH2;
        laplacianZ = (sumZ - 6.0 * fieldZ) * invH2;
    }

    void calculateddt()
    {
        d2xdt2 = 0.01 * laplacianX;
        dydt = 0.01 * laplacianY;
        dzdt = 0.01 * laplacianZ;
	}

    void updateField(double deltaTime)
    {
        fieldX = fieldX + dxdt * deltaTime + 0.5 * d2xdt2 * deltaTime * deltaTime;
        fieldY = fieldY + dydt * deltaTime + 0.5 * d2ydt2 * deltaTime * deltaTime;
        fieldZ = fieldZ + dxdt * deltaTime + 0.5 * d2zdt2 * deltaTime * deltaTime;

		dxdt = dxdt + d2xdt2 * deltaTime;
        dydt = dydt + d2ydt2 * deltaTime;
		dzdt = dzdt + d2zdt2 * deltaTime;
	}

    void updateColour()
    {
        // this is ai, so be suspicious
        
        // Map temperature (fieldX) to hue, and diffusion speed (|dxdt|) to value (brightness)
        // Normalize hue in [0,360), use an atan stretch so large values saturate smoothly
        double hueNorm = (std::atan(fieldX * 0.02) / std::numbers::pi) + 0.5; // maps to [0,1]
        if (hueNorm < 0.0) hueNorm = 0.0;
        if (hueNorm > 1.0) hueNorm = 1.0;
        double hue = hueNorm * 360.0;

        // Map speed to value [0,1] (use abs of dxdt)
        double speed = std::abs(dxdt);
        double value = 1; // std::atan(speed * 100.0) / (std::numbers::pi / 2.0); // maps to [0,1)
        // value = std::clamp(value, 0.0, 1.0);

        double saturation = 1.0; // full saturation

        // HSV -> RGB
        double c = value * saturation;
        double hprime = hue / 60.0;
        double x = c * (1.0 - std::fabs(std::fmod(hprime, 2.0) - 1.0));
        double r1 = 0.0, g1 = 0.0, b1 = 0.0;
        int sector = static_cast<int>(std::floor(hprime)) % 6;
        if (sector < 0) sector += 6;
        switch (sector)
        {
            case 0: r1 = c; g1 = x; b1 = 0; break;
            case 1: r1 = x; g1 = c; b1 = 0; break;
            case 2: r1 = 0; g1 = c; b1 = x; break;
            case 3: r1 = 0; g1 = x; b1 = c; break;
            case 4: r1 = x; g1 = 0; b1 = c; break;
            case 5: r1 = c; g1 = 0; b1 = x; break;
        }
        double m = value - c;
        double rf = r1 + m;
        double gf = g1 + m;
        double bf = b1 + m;

        short int R = static_cast<short int>(std::clamp(rf * 255.0, 0.0, 255.0));
        short int G = static_cast<short int>(std::clamp(gf * 255.0, 0.0, 255.0));
        short int B = static_cast<short int>(std::clamp(bf * 255.0, 0.0, 255.0));

        setColour(R, G, B, 255);
        setZ(fieldX);
    }
};