#include "SimWave.h"
#include <glad/glad.h>
#include <cmath>
#include "SimulationRegistry.h"

SimWave::SimWave() : context(nullptr), accumulator(0.0f),
    waveSpeed(1.0f), timestep(0.0f), targetTimestep(0.0f),
    damping(0.999f), initialized(false) {
}

void SimWave::init(SimulationContext& ctx) {
    context = &ctx;
    calculateTimestep();
    accumulator = 0.0f;
    initialized = true;
}

void SimWave::calculateTimestep() {
    if (context) {
        targetTimestep = context->targetStepDt;
        timestep = targetTimestep;
    }
}

void SimWave::update(float simBudget, float& simTime) {
    if (!context || !initialized) return;

    if (targetTimestep == 0.0f && context->targetStepDt > 0) {
        targetTimestep = context->targetStepDt;
        calculateTimestep();
    }

    accumulator += simBudget;

    glUniform1f(context->uDiffusion, waveSpeed * waveSpeed); // c²
    glUniform1i(context->uMode, 1); // Wave mode

    while (accumulator >= timestep) {
        glUniform1f(context->uDt, timestep);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, context->ssbo[context->currentBuffer]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, context->ssbo[1 - context->currentBuffer]);
        glDispatchCompute(context->groups, context->groups, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        accumulator -= timestep;
        simTime += timestep;
        context->currentBuffer = 1 - context->currentBuffer;
    }
}

void SimWave::reset() {
    if (!context) return;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, context->ssbo[0]);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, context->ssbo[1]);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);

    accumulator = 0.0f;
    context->currentBuffer = 0;
}

REGISTER_SIMULATION(2, "Wave", SimWave)
