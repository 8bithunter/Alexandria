#include "Simulation.h"
#include <glad/glad.h>
#include <cmath>
#include <algorithm>

// Diffusion simulation implementation
class DiffusionSimulation : public ISimulation {
private:
    int m_resolution = 200;
    float m_accumulator = 0.0f;
    float m_diffusionCoefficient = 0.01f; // Diffusion coefficient
    float m_simulationDt = 0.0f;
    float m_targetStepDt = 0.0f;

    void calculateTimestep() {
        float h = 1.0f / m_resolution;
        float maxDt = (h * h) / (4.0f * m_diffusionCoefficient) * 0.9f;
        m_simulationDt = std::min(maxDt, m_targetStepDt);
    }

public:
    DiffusionSimulation(int resolution = 200) {
        m_resolution = resolution;
    }

    int getMode() const override { return 0; }
    const char* getName() const override { return "Diffusion"; }

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

        // Diffusion
        m_accumulator += simBudget;
        glUniform1f(state.uDiffusionU, m_diffusionCoefficient);
        glUniform1i(state.uComputeModeU, 0);
        glUniform1f(state.uDtU, m_simulationDt);

        while (m_accumulator >= m_simulationDt) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, state.ssbo[state.current]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, state.ssbo[1 - state.current]);
            glDispatchCompute(state.groups, state.groups, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            state.current = 1 - state.current;
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
        return m_resolution > 0 && m_simulationDt > 0.0f;
    }

    float getTimestep() const override {
        return m_simulationDt;
    }
};

// Factory function
ISimulation* createDiffusionSimulation(int resolution) {
    ISimulation* sim = new DiffusionSimulation(resolution);
    sim->init(resolution);
    return sim;
}
