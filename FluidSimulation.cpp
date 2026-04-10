#include "Simulation.h"
#include <glad/glad.h>
#include <cmath>

// Fluid/Navier-Stokes simulation implementation
class FluidSimulation : public ISimulation {
private:
    int m_resolution = 200;
    float m_accumulator = 0.0f;
    float m_kinematicViscosity = 0.001f; // ν for Navier-Stokes
    float m_simulationDt = 0.0f;
    float m_targetStepDt = 0.0f;
    static constexpr int JACOBI_ITERS = 20;

    void calculateTimestep() {
        // Use the target step dt from state
        m_simulationDt = m_targetStepDt;
    }

public:
    FluidSimulation(int resolution = 200) {
        m_resolution = resolution;
    }

    int getMode() const override { return 2; }
    const char* getName() const override { return "Fluid"; }

    void init(int resolution) override {
        m_resolution = resolution;
        calculateTimestep();
    }

    void update(SimulationState& state, float& simTime, float simBudget) override {
        // Grab parameters from state
        if (m_targetStepDt == 0.0f && state.TARGET_STEP_DT > 0) {
            m_targetStepDt = state.TARGET_STEP_DT;
            calculateTimestep();
        }

        // Full incompressible Navier-Stokes
        m_accumulator += simBudget;

        while (m_accumulator >= m_simulationDt) {
            // Always bind ssboDiv to slot 2
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, state.ssboDiv);

            // (a) Advect + viscous diffusion
            glUniform1i(state.uComputeModeU, 2);
            glUniform1f(state.uDiffusionU, m_kinematicViscosity);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, state.ssbo[state.current]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, state.ssbo[1 - state.current]);
            glDispatchCompute(state.groups, state.groups, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            state.current = 1 - state.current; // ssbo[current] now holds u*

            // (b) Divergence ∇·u*
            glUniform1i(state.uComputeModeU, 7);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, state.ssbo[state.current]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, state.ssboDiv);
            glDispatchCompute(state.groups, state.groups, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, state.ssboDiv); // restore slot 2

            // (c) Jacobi pressure solve ∇²p = ∇·u*/dt
            glUniform1i(state.uComputeModeU, 5);
            for (int j = 0; j < JACOBI_ITERS; ++j) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, state.ssboPres[state.pressCur]);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, state.ssboPres[1 - state.pressCur]);
                glDispatchCompute(state.groups, state.groups, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                state.pressCur = 1 - state.pressCur;
            }

            // (d) Pressure projection u = u* – dt ∇p
            glUniform1i(state.uComputeModeU, 6);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, state.ssbo[state.current]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, state.ssbo[1 - state.current]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, state.ssboPres[state.pressCur]);
            glDispatchCompute(state.groups, state.groups, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            state.current = 1 - state.current; // ssbo[current] holds div-free velocity

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, state.ssboDiv);

            m_accumulator -= m_simulationDt;
            simTime += m_simulationDt;
        }
    }

    void reset() override {
        m_accumulator = 0.0f;
    }

    int getResolution() const override {
        return m_resolution;
    }

    void setResolution(int resolution) override {
        if (m_resolution != resolution) {
            m_resolution = resolution;
            calculateTimestep();
        }
    }

    SimulationState& getState() override {
        static SimulationState dummy; // Should not be called
        return dummy;
    }

    const SimulationState& getState() const override {
        static SimulationState dummy; // Should not be called
        return dummy;
    }

    bool isInitialized() const override {
        return m_resolution > 0;
    }

    float getTimestep() const override {
        return m_simulationDt;
    }
};

// Factory function
ISimulation* createFluidSimulation(int resolution) {
    ISimulation* sim = new FluidSimulation(resolution);
    sim->init(resolution);
    return sim;
}
