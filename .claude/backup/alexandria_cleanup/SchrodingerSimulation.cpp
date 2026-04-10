#include "Simulation.h"
#include <glad/glad.h>
#include <cmath>

// Schrödinger simulation implementation
class SchrodingerSimulation : public ISimulation {
private:
    int m_resolution = 200;
    float m_accumulator = 0.0f;
    float m_planckScale = 0.01f; // Scale factor for Schrödinger equation
    float m_simulationDt = 0.0f;

    void calculateTimestep() {
        float h = 1.0f / m_resolution;
        m_simulationDt = 0.45f * h * h; // Fixed timestep for Schrödinger
    }

public:
    SchrodingerSimulation(int resolution = 200) {
        m_resolution = resolution;
    }

    int getMode() const override { return 3; }
    const char* getName() const override { return "Schrodinger"; }

    void init(int resolution) override {
        m_resolution = resolution;
        calculateTimestep();
    }

    void update(SimulationState& state, float& simTime, float simBudget) override {
        // Schrödinger (Strang split: Re then Im)
        m_accumulator += simBudget;

        while (m_accumulator >= m_simulationDt) {
            // Real part update (mode 3)
            glUniform1i(state.uComputeModeU, 3);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, state.ssbo[state.current]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, state.ssbo[1 - state.current]);
            glUniform1f(state.uDtU, m_simulationDt);
            glUniform1f(state.uDiffusionU, m_planckScale);
            glDispatchCompute(state.groups, state.groups, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            state.current = 1 - state.current;

            // Imaginary part update (mode 4)
            glUniform1i(state.uComputeModeU, 4);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, state.ssbo[state.current]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, state.ssbo[1 - state.current]);
            glUniform1f(state.uDtU, m_simulationDt);
            glUniform1f(state.uDiffusionU, m_planckScale);
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
ISimulation* createSchrodingerSimulation(int resolution) {
    ISimulation* sim = new SchrodingerSimulation(resolution);
    sim->init(resolution);
    return sim;
}
