#ifndef SIM_DIFFUSION_H
#define SIM_DIFFUSION_H

#include "../Simulation.h"

// Diffusion simulation: ∂u/∂t = D ∇²u
class SimDiffusion : public Simulation {
public:
    SimDiffusion();
    ~SimDiffusion() = default;

    void init(SimulationContext& ctx) override;
    void update(float simBudget, float& simTime) override;
    void reset() override;

    const char* getName() const override { return "Diffusion"; }
    int getMode() const override { return 0; }
    float getDefaultTimestep() const override { return timestep; }
    bool isInitialized() const override { return initialized; }

private:
    void calculateTimestep();

private:
    SimulationContext* context;
    float accumulator;
    float diffusionCoefficient;
    float timestep;
    float targetTimestep;
    bool initialized;
};

#endif // SIM_DIFFUSION_H
