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

    double gradXx = 0, gradXy = 0, gradXz = 0,
           gradYx = 0, gradYy = 0, gradYz = 0,
           gradZx = 0, gradZy = 0, gradZz = 0;
	double lapX = 0, lapY = 0, lapZ = 0;

    double previousFieldX = 0, previousFieldY = 0, previousFieldZ = 0,
           previousFieldXNeighbourUp = 0, previousFieldYNeighbourUp = 0, previousFieldZNeighbourUp = 0,
           previousFieldXNeighbourDown = 0, previousFieldYNeighbourDown = 0, previousFieldZNeighbourDown = 0,
           previousFieldXNeighbourLeft = 0, previousFieldYNeighbourLeft = 0, previousFieldZNeighbourLeft = 0,
           previousFieldXNeighbourRight = 0, previousFieldYNeighbourRight = 0, previousFieldZNeighbourRight = 0,
           previousFieldXNeighbourOut = 0, previousFieldYNeighbourOut = 0, previousFieldZNeighbourOut = 0,
           previousFieldXNeighbourIn= 0, previousFieldYNeighbourIn = 0, previousFieldZNeighbourIn = 0;
           

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

    double getGradXx()
    {

        if (neighbourLeft->fieldX == previousFieldXNeighbourLeft && neighbourRight->fieldX == previousFieldXNeighbourRight)
        {
            return gradXx;
        }
        else
        {
            previousFieldXNeighbourLeft = neighbourLeft->fieldX;
            previousFieldXNeighbourRight = neighbourRight->fieldX;
            previousFieldYNeighbourLeft = neighbourLeft->fieldY;
            previousFieldYNeighbourRight = neighbourRight->fieldY;
            previousFieldZNeighbourLeft = neighbourLeft->fieldZ;
            previousFieldZNeighbourRight = neighbourRight->fieldZ;
            calculateGradX();
            return gradXx;
        }
    }

    double getGradXy()
    {
        if (neighbourLeft->fieldX == previousFieldXNeighbourLeft && neighbourRight->fieldX == previousFieldXNeighbourRight)
        {
            return gradXy;
        }
        else
        {
            previousFieldXNeighbourLeft = neighbourLeft->fieldX;
            previousFieldXNeighbourRight = neighbourRight->fieldX;
            previousFieldYNeighbourLeft = neighbourLeft->fieldY;
            previousFieldYNeighbourRight = neighbourRight->fieldY;
            previousFieldZNeighbourLeft = neighbourLeft->fieldZ;
            previousFieldZNeighbourRight = neighbourRight->fieldZ;

            calculateGradX();
            return gradXy;
        }
    }

    double getGradXz()
    {
        if (neighbourLeft->fieldX == previousFieldXNeighbourLeft && neighbourRight->fieldX == previousFieldXNeighbourRight)
        {
            return gradXz;
        }
        else
        {
            previousFieldXNeighbourLeft = neighbourLeft->fieldX;
            previousFieldXNeighbourRight = neighbourRight->fieldX;
            previousFieldYNeighbourLeft = neighbourLeft->fieldY;
            previousFieldYNeighbourRight = neighbourRight->fieldY;
            previousFieldZNeighbourLeft = neighbourLeft->fieldZ;
            previousFieldZNeighbourRight = neighbourRight->fieldZ;

            calculateGradX();
            return gradXz;
        }
    }

    double getGradYx()
    {
        if (neighbourDown->fieldX == previousFieldXNeighbourDown && neighbourUp->fieldX == previousFieldXNeighbourUp)
        {
            return gradYx;
        }
        else
        {
            previousFieldXNeighbourDown = neighbourDown->fieldX;
            previousFieldXNeighbourUp = neighbourUp->fieldX;
            previousFieldYNeighbourDown = neighbourDown->fieldY;
            previousFieldYNeighbourUp = neighbourUp->fieldY;
            previousFieldZNeighbourDown = neighbourDown->fieldZ;
            previousFieldZNeighbourUp = neighbourUp->fieldZ;

            calculateGradY();
            return gradYx;
        }
    }

    double getGradYy()
    {
        if (neighbourDown->fieldX == previousFieldXNeighbourDown && neighbourUp->fieldX == previousFieldXNeighbourUp)
        {
            return gradYy;
        }
        else
        {
            previousFieldXNeighbourDown = neighbourDown->fieldX;
            previousFieldXNeighbourUp = neighbourUp->fieldX;
            previousFieldYNeighbourDown = neighbourDown->fieldY;
            previousFieldYNeighbourUp = neighbourUp->fieldY;
            previousFieldZNeighbourDown = neighbourDown->fieldZ;
            previousFieldZNeighbourUp = neighbourUp->fieldZ;

            calculateGradY();
            return gradYy;
        }
    }

    double getGradYz()
    {
        if (neighbourDown->fieldX == previousFieldXNeighbourDown && neighbourUp->fieldX == previousFieldXNeighbourUp)
        {
            return gradYz;
        }
        else
        {
            previousFieldXNeighbourDown = neighbourDown->fieldX;
            previousFieldXNeighbourUp = neighbourUp->fieldX;
            previousFieldYNeighbourDown = neighbourDown->fieldY;
            previousFieldYNeighbourUp = neighbourUp->fieldY;
            previousFieldZNeighbourDown = neighbourDown->fieldZ;
            previousFieldZNeighbourUp = neighbourUp->fieldZ;

            calculateGradY();
            return gradYz;
        }
    }

    double getGradZx()
    {
        if (neighbourIn->fieldX == previousFieldXNeighbourIn && neighbourOut->fieldX == previousFieldXNeighbourOut)
        {
            return gradZx;
        }
        else
        {
            previousFieldXNeighbourIn = neighbourIn->fieldX;
            previousFieldXNeighbourOut = neighbourOut->fieldX;
            previousFieldYNeighbourIn = neighbourIn->fieldY;
            previousFieldYNeighbourOut = neighbourOut->fieldY;
            previousFieldZNeighbourIn = neighbourIn->fieldZ;
            previousFieldZNeighbourOut = neighbourOut->fieldZ;

            calculateGradZ();
            return gradZx;
        }
    }

    double getGradZy()
    {
        if (neighbourIn->fieldX == previousFieldXNeighbourIn && neighbourOut->fieldX == previousFieldXNeighbourOut)
        {
            return gradZy;
        }
        else
        {
            previousFieldXNeighbourIn = neighbourIn->fieldX;
            previousFieldXNeighbourOut = neighbourOut->fieldX;
            previousFieldYNeighbourIn = neighbourIn->fieldY;
            previousFieldYNeighbourOut = neighbourOut->fieldY;
            previousFieldZNeighbourIn = neighbourIn->fieldZ;
            previousFieldZNeighbourOut = neighbourOut->fieldZ;

            calculateGradZ();
            return gradZy;
        }
    }

    double getGradZz()
    {
        if (neighbourIn->fieldX == previousFieldXNeighbourIn && neighbourOut->fieldX == previousFieldXNeighbourOut)
        {
            return gradZz;
        }
        else
        {
            previousFieldXNeighbourIn = neighbourIn->fieldX;
            previousFieldXNeighbourOut = neighbourOut->fieldX;
            previousFieldYNeighbourIn = neighbourIn->fieldY;
            previousFieldYNeighbourOut = neighbourOut->fieldY;
            previousFieldZNeighbourIn = neighbourIn->fieldZ;
            previousFieldZNeighbourOut = neighbourOut->fieldZ;

            calculateGradZ();
            return gradZz;
        }
    }

	double getDiv() { return getGradXx() + getGradYy() + getGradZz(); }

    double getCurlX() { return getGradYz() - getGradZy(); }

	double getCurlY() { return getGradZx() - getGradXz(); }

	double getCurlZ() { return getGradXy() - getGradYx(); }

    double getLaplacianX()
    {
        if (previousFieldXNeighbourRight == neighbourRight->fieldX &&
            previousFieldXNeighbourLeft == neighbourLeft->fieldX &&
            previousFieldXNeighbourUp == neighbourUp->fieldX &&
            previousFieldXNeighbourDown == neighbourDown->fieldX &&
            previousFieldXNeighbourOut == neighbourOut->fieldX &&
            previousFieldXNeighbourIn == neighbourIn->fieldX &&
            previousFieldX == fieldX)
        {
            return lapX;  
        }

        previousFieldXNeighbourRight = neighbourRight->fieldX;
        previousFieldXNeighbourLeft = neighbourLeft->fieldX;
        previousFieldXNeighbourUp = neighbourUp->fieldX;
        previousFieldXNeighbourDown = neighbourDown->fieldX;
        previousFieldXNeighbourOut = neighbourOut->fieldX;
        previousFieldXNeighbourIn = neighbourIn->fieldX;
        previousFieldX = fieldX;

        lapX = (previousFieldXNeighbourRight + previousFieldXNeighbourLeft +
            previousFieldXNeighbourUp + previousFieldXNeighbourDown +
            previousFieldXNeighbourOut + previousFieldXNeighbourIn -
            6.0 * previousFieldX) * invH2;

        return lapX;
    }

    double getLaplacianY()
    {
        if (previousFieldYNeighbourRight == neighbourRight->fieldY &&
            previousFieldYNeighbourLeft == neighbourLeft->fieldY &&
            previousFieldYNeighbourUp == neighbourUp->fieldY &&
            previousFieldYNeighbourDown == neighbourDown->fieldY &&
            previousFieldYNeighbourOut == neighbourOut->fieldY &&
            previousFieldYNeighbourIn == neighbourIn->fieldY &&
            previousFieldY == fieldY)
        {
            return lapY;
        }

        previousFieldYNeighbourRight = neighbourRight->fieldY;
        previousFieldYNeighbourLeft = neighbourLeft->fieldY;
        previousFieldYNeighbourUp = neighbourUp->fieldY;
        previousFieldYNeighbourDown = neighbourDown->fieldY;
        previousFieldYNeighbourOut = neighbourOut->fieldY;
        previousFieldYNeighbourIn = neighbourIn->fieldY;
        previousFieldY = fieldY;

        lapY = (previousFieldYNeighbourRight + previousFieldYNeighbourLeft +
            previousFieldYNeighbourUp + previousFieldYNeighbourDown +
            previousFieldYNeighbourOut + previousFieldYNeighbourIn -
            6.0 * previousFieldY) * invH2;

        return lapY;
    }

    double getLaplacianZ()
    {
        if (previousFieldZNeighbourRight == neighbourRight->fieldZ &&
            previousFieldZNeighbourLeft == neighbourLeft->fieldZ &&
            previousFieldZNeighbourUp == neighbourUp->fieldZ &&
            previousFieldZNeighbourDown == neighbourDown->fieldZ &&
            previousFieldZNeighbourOut == neighbourOut->fieldZ &&
            previousFieldZNeighbourIn == neighbourIn->fieldZ &&
            previousFieldZ == fieldZ)
        {
            return lapZ;
        }

        previousFieldZNeighbourRight = neighbourRight->fieldZ;
        previousFieldZNeighbourLeft = neighbourLeft->fieldZ;
        previousFieldZNeighbourUp = neighbourUp->fieldZ;
        previousFieldZNeighbourDown = neighbourDown->fieldZ;
        previousFieldZNeighbourOut = neighbourOut->fieldZ;
        previousFieldZNeighbourIn = neighbourIn->fieldZ;
        previousFieldZ = fieldZ;

        lapZ = (previousFieldZNeighbourRight + previousFieldZNeighbourLeft +
            previousFieldZNeighbourUp + previousFieldZNeighbourDown +
            previousFieldZNeighbourOut + previousFieldZNeighbourIn -
            6.0 * previousFieldZ) * invH2;

        return lapZ;
    }


	double fieldX = 0, fieldY = 0, fieldZ = 0;
	double dxdt = 0, dydt = 0, dzdt = 0;
	double d2xdt2 = 0, d2ydt2 = 0, d2zdt2 = 0;

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

    void calculateGradX()
    {
        gradXx = (previousFieldXNeighbourRight - previousFieldXNeighbourLeft) * inv2H;
        gradXy = (previousFieldYNeighbourRight - previousFieldYNeighbourLeft) * inv2H;
        gradXz = (previousFieldZNeighbourRight - previousFieldZNeighbourLeft) * inv2H;
    }

    void calculateGradY()
    {
        gradYx = (previousFieldXNeighbourUp - previousFieldXNeighbourDown) * inv2H;
        gradYy = (previousFieldYNeighbourUp - previousFieldYNeighbourDown) * inv2H;
        gradYz = (previousFieldZNeighbourUp - previousFieldZNeighbourDown) * inv2H;
    }

    void calculateGradZ()
    {
        gradZx = (previousFieldXNeighbourOut - previousFieldXNeighbourIn) * inv2H;
        gradZy = (previousFieldYNeighbourOut - previousFieldYNeighbourIn) * inv2H;
        gradZz = (previousFieldZNeighbourOut - previousFieldZNeighbourIn) * inv2H;
    }

    void calculateddt()
    {
		dxdt = 0.01 * getLaplacianX();
        //d2xdt2 = 0.01 * getLaplacianX();
		// dxdt *= 0.99; 
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