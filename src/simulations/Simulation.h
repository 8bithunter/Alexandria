#ifndef SIMULATION_H
#define SIMULATION_H

#include <string>

// Forward declare GLuint
using GLuint = unsigned int;

// Shared simulation context
struct SimulationContext {
    int resolution;
    GLuint ssbo[2];           // Ping-pong buffers for field data
    GLuint ssboDiv;           // Divergence buffer (for Navier-Stokes)
    GLuint ssboPres[2];       // Pressure buffers (for Navier-Stokes)

    // Uniform locations (cached)
    int uRes, uInvH2, uDiffusion, uDt, uMode;
    int uRotation, uFieldMode, uFieldRes;
    int uTxtOrigin, uTxtSize, uTxtAdvance, uTxtFont, uTxtChars, uTxtColor;
    int uRectOrigin, uRectSize, uRectColor;

    // Derived
    float invH2;
    float targetStepDt;

    int groups;               // Compute groups for dispatch
    int currentBuffer;        // Current ping-pong buffer index
    int pressureBuffer;         // Current pressure buffer index
};

// Base simulation interface
class Simulation {
public:
    virtual ~Simulation() = default;

    // Initialize with context
    virtual void init(SimulationContext& ctx) = 0;

    // Update simulation by simBudget time, advancing simTime
    virtual void update(float simBudget, float& simTime) = 0;

    // Reset field to initial conditions
    virtual void reset() = 0;

    // Metadata
    virtual const char* getName() const = 0;
    virtual int getMode() const = 0;

    // Default timestep for this simulation
    virtual float getDefaultTimestep() const = 0;

    // Check if initialized
    virtual bool isInitialized() const = 0;
};

#endif // SIMULATION_H
