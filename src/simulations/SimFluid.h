#ifndef SIM_FLUID_H
#define SIM_FLUID_H

#include "../Simulation.h"

// Navier-Stokes fluid simulation
// Currently implements: semi-Lagrangian advection + viscosity (simplified)
// TODO: Add divergence computation, pressure solve, and projection
//       for full incompressible Navier-Stokes
class SimFluid : public Simulation {
public:
    SimFluid();
    ~SimFluid() = default;

    void init(SimulationContext& ctx) override;
    void update(float simBudget, float& simTime) override;
    void reset() override;

    const char* getName() const override { return "Fluid"; }
    int getMode() const override { return 2; }
    float getDefaultTimestep() const override { return timestep; }
    bool isInitialized() const override { return initialized; }

private:
    void calculateTimestep();

private:
    SimulationContext* context;
    float accumulator;
    float kinematicViscosity;  // ν
    float timestep;
    float targetTimestep;
    bool initialized;
};

#endif // SIM_FLUID_H
