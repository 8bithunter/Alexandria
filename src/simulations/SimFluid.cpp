#include "SimFluid.h"
#include <glad/glad.h>
#include <cmath>

SimFluid::SimFluid() : context(nullptr), accumulator(0.0f),
    kinematicViscosity(0.001f), timestep(0.0f), targetTimestep(0.0f),
    initialized(false) {
}

void SimFluid::init(SimulationContext& ctx) {
    context = &ctx;
    calculateTimestep();
    accumulator = 0.0f;
    initialized = true;

    // TODO: Initialize divergence and pressure buffers for full Navier-Stokes
    // Currently implements simplified semi-Lagrangian advection + viscosity
}

void SimFluid::calculateTimestep() {
    if (context) {
        targetTimestep = context->targetStepDt;
        timestep = targetTimestep;
    }
}

void SimFluid::update(float simBudget, float& simTime) {
    if (!context || !initialized) return;

    if (targetTimestep == 0.0f && context->targetStepDt > 0) {
        targetTimestep = context->targetStepDt;
        calculateTimestep();
    }

    accumulator += simBudget;

    glUniform1f(context->uDiffusion, kinematicViscosity);
    glUniform1i(context->uMode, 2); // Fluid advection mode

    // Semi-Lagrangian advection + viscous diffusion
    // TODO: Add full Navier-Stokes with divergence/pressure projection:
    // 1. Compute advected velocity
    // 2. Compute divergence → ssboDiv
    // 3. Jacobi pressure solve → ssboPres[]
    // 4. Project velocity (subtract pressure gradient)

    // Currently only does advection + diffusion (simplified)
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

void SimFluid::reset() {
    if (!context) return;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, context->ssbo[0]);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, context->ssbo[1]);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);

    // TODO: Clear divergence and pressure buffers when added
    accumulator = 0.0f;
    context->currentBuffer = 0;
}

REGISTER_SIMULATION(3, "Fluid", SimFluid)
