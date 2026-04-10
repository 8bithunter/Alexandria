#ifndef SIM_SCHRODINGER_H
#define SIM_SCHRODINGER_H

#include "../Simulation.h"

// Schrödinger equation: i ∂ψ/∂t = –½ ∇²ψ
// Uses split-step method with alternating real/imaginary updates
class SimSchrodinger : public Simulation {
public:
    SimSchrodinger();
    ~SimSchrodinger() = default;

    void init(SimulationContext& ctx) override;
    void update(float simBudget, float& simTime) override;
    void reset() override;

    const char* getName() const override { return "Schrodinger"; }
    int getMode() const override { return 2; } // Multiple modes used internally
    float getDefaultTimestep() const override { return timestep; }
    bool isInitialized() const override { return initialized; }

private:
    void calculateTimestep();

private:
    SimulationContext* context;
    float accumulator;
    float planckScale;         // Scale factor for quantum effects
    float timestep;
    float targetTimestep;
    bool initialized;
    bool realPass;             // Whether we're doing real or imaginary update
};

#endif // SIM_SCHRODINGER_H
