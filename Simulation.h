#ifndef SIMULATION_H
#define SIMULATION_H

// Forward declare GLuint to avoid including glad here
typedef unsigned int GLuint;

// Simulation state container for OpenGL resources
struct SimulationState {
    GLuint ssbo[2];
    GLuint ssboDiv;
    GLuint ssboPres[2];

    int current = 0;
    int pressCur = 0;

    int groups = 0;

    // Uniform locations
    int uResU = 0;
    int uInvH2U = 0;
    int uDiffusionU = 0;
    int uDtU = 0;
    int uComputeModeU = 0;
    int uFieldModeU = 0;
    int uFieldResU = 0;
    int uRotationU = 0;
    int uTxtOrigin = 0;
    int uTxtSize = 0;
    int uTxtAdvance = 0;
    int uTxtFont = 0;
    int uTxtChars = 0;
    int uTxtColor = 0;
    int uRectOrigin = 0;
    int uRectSize = 0;
    int uRectColor = 0;

    // Parameters
    int N = 0;
    float invH2 = 0.0f;
    float TARGET_STEP_DT = 0.0f;
};

// Interface for all simulation types
class ISimulation {
public:
    virtual ~ISimulation() {}

    // Initialize with resolution
    virtual void init(int resolution) = 0;

    // Update one frame
    virtual void update(SimulationState& state, float& simTime, float simBudget) = 0;

    // Reset simulation
    virtual void reset() = 0;

    // Get/set resolution
    virtual int getResolution() const = 0;
    virtual void setResolution(int resolution) = 0;

    // Get metadata
    virtual const char* getName() const = 0;
    virtual int getMode() const = 0;

    // Get timestep
    virtual float getTimestep() const = 0;

    // Access state
    virtual SimulationState& getState() = 0;
    virtual const SimulationState& getState() const = 0;
    virtual bool isInitialized() const = 0;
};

// Factory functions
ISimulation* createSimulation(int mode, int resolution);
ISimulation* createDiffusionSimulation(int resolution);
ISimulation* createWaveSimulation(int resolution);
ISimulation* createFluidSimulation(int resolution);
ISimulation* createSchrodingerSimulation(int resolution);

#endif // SIMULATION_H
