#ifndef SIM_WAVE_H
#define SIM_WAVE_H

#include "../Simulation.h"

// Wave equation simulation: ∂²u/∂t² = c² ∇²u
// Uses leapfrog integration with velocity damping
class SimWave : public Simulation {
public:
    SimWave();
    ~SimWave() = default;

    void init(SimulationContext& ctx) override;
    void update(float simBudget, float& simTime) override;
    void reset() override;

    const char* getName() const override { return "Wave"; }
    int getMode() const override { return 1; }
    float getDefaultTimestep() const override { return timestep; }
    bool isInitialized() const override { return initialized; }

private:
    void calculateTimestep();

private:
    SimulationContext* context;
    float accumulator;
    float waveSpeed;           // c in wave equation
    float timestep;
    float targetTimestep;
    float damping;             // Velocity damping factor
    bool initialized;
};

#endif // SIM_WAVE_H
