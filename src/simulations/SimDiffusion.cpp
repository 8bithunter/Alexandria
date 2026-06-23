#include "SimDiffusion.h"
#include <glad/glad.h>
#include <cmath>
#include "SimulationRegistry.h"

SimDiffusion::SimDiffusion() : context(nullptr), accumulator(0.0f),
    diffusionCoefficient(0.01f), timestep(0.0f), targetTimestep(0.0f), initialized(false) {
}

void SimDiffusion::init(SimulationContext& ctx) {
    context = &ctx;
    calculateTimestep();
    accumulator = 0.0f;
    initialized = true;
}

void SimDiffusion::calculateTimestep() {
    if (context)
    {
        targetTimestep = context->targetStepDt;
        timestep = targetTimestep;
    }
}

void SimDiffusion::update(float simBudget, float& simTime) {
    if (!context || !initialized) return;

    // Grab parameters from context if needed
    if (targetTimestep == 0.0f && context->targetStepDt > 0) {
        targetTimestep = context->targetStepDt;
        calculateTimestep();
    }

    // Update accumulator
    accumulator += simBudget;

    // Set uniforms
    glUniform1f(context->uDiffusion, diffusionCoefficient);
    glUniform1i(context->uMode, 0); // Diffusion mode

    // Update simulation
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

void SimDiffusion::reset() {
    if (!context) return;

    // Clear both buffers to zero
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, context->ssbo[0]);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, context->ssbo[1]);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);

    accumulator = 0.0f;
    context->currentBuffer = 0;
}

// Explicit instantiation with registration
REGISTER_SIMULATION(1, "Diffusion", SimDiffusion)
