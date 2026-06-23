#include "SimSchrodinger.h"
#include <glad/glad.h>
#include <cmath>
#include "SimulationRegistry.h"

SimSchrodinger::SimSchrodinger() : context(nullptr), accumulator(0.0f),
    planckScale(1.0f), timestep(0.0f), targetTimestep(0.0f),
    initialized(false), realPass(true) {
}

void SimSchrodinger::init(SimulationContext& ctx) {
    context = &ctx;
    calculateTimestep();
    accumulator = 0.0f;
    realPass = true;
    initialized = true;
}

void SimSchrodinger::calculateTimestep() {
    if (context) {
        targetTimestep = context->targetStepDt;
        timestep = targetTimestep;
    }
}

void SimSchrodinger::update(float simBudget, float& simTime) {
    if (!context || !initialized) return;

    if (targetTimestep == 0.0f && context->targetStepDt > 0) {
        targetTimestep = context->targetStepDt;
        calculateTimestep();
    }

    accumulator += simBudget;

    // Two passes: real then imaginary
    while (accumulator >= timestep * 0.5f) {
        float dt = timestep * 0.5f; // Each pass uses half timestep

        if (realPass) {
            // Pass A: ∂Re/∂t = +½ ∇²Im
            glUniform1f(context->uDt, dt);
            glUniform1i(context->uMode, 3); // Mode 3: real update
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, context->ssbo[context->currentBuffer]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, context->ssbo[1 - context->currentBuffer]);
            glDispatchCompute(context->groups, context->groups, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        } else {
            // Pass B: ∂Im/∂t = -½ ∇²Re
            glUniform1f(context->uDt, dt);
            glUniform1i(context->uMode, 4); // Mode 4: imaginary update
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, context->ssbo[context->currentBuffer]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, context->ssbo[1 - context->currentBuffer]);
            glDispatchCompute(context->groups, context->groups, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            accumulator -= timestep;
            simTime += timestep;
            context->currentBuffer = 1 - context->currentBuffer;
        }

        realPass = !realPass; // Alternate between real and imaginary passes
    }
}

void SimSchrodinger::reset() {
    if (!context) return;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, context->ssbo[0]);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, context->ssbo[1]);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);

    accumulator = 0.0f;
    context->currentBuffer = 0;
    realPass = true;
}

REGISTER_SIMULATION(4, "Schrodinger", SimSchrodinger)
