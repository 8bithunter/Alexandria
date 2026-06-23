#ifndef SIM_FLUID_H
#define SIM_FLUID_H

#include "Simulation.h"

// Navier-Stokes fluid simulation
// Implements full incompressible Navier-Stokes equations:
// ∂u/∂t = -(u·∇)u + ν∇²u - (1/ρ)∇p + f
// ∇·u = 0 (incompressibility constraint)
//
// Uses operator splitting:
// 1. Advection: semi-Lagrangian
// 2. Diffusion: implicit viscosity
// 3. Pressure projection: divergence → pressure solve → velocity correction
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
    void computeDivergenceAndPressure(float dt);

private:
    SimulationContext* context;
    float accumulator;
    float kinematicViscosity; // ν - kinematic viscosity
    float density;            // ρ - fluid density (proportionality constant)
    float timestep;
    float targetTimestep;
    bool initialized;

    // Pressure solver parameters
    static constexpr int MAX_PRESSURE_ITERATIONS = 50;
    static constexpr float PRESSURE_TOLERANCE = 1e-5f;
};

#endif // SIM_FLUID_H
