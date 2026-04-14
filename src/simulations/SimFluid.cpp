#include "SimFluid.h"
#include <glad/glad.h>
#include <cmath>

SimFluid::SimFluid() : context(nullptr), accumulator(0.0f),
kinematicViscosity(0.0001f), density(1.0f), timestep(0.0f), targetTimestep(0.0f),
initialized(false) {
}

void SimFluid::init(SimulationContext& ctx) {
    context = &ctx;
    calculateTimestep();
    accumulator = 0.0f;
    initialized = true;

    // Full Navier-Stokes implementation with density and pressure projection
    // Uses operator splitting: advection → diffusion → pressure projection
    // Divergence and pressure stored in temporary buffer components
}

void SimFluid::calculateTimestep() {
    if (context) {
        targetTimestep = context->targetStepDt;
        timestep = targetTimestep;
    }
}

void SimFluid::computeDivergenceAndPressure(float dt) {
    if (!context || !initialized) return;

    // Step 1: Compute divergence
    glUniform1i(context->uMode, 4); // Divergence computation mode
    glUniform1f(context->uDt, dt);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, context->ssbo[context->currentBuffer]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, context->ssbo[1 - context->currentBuffer]);
    glDispatchCompute(context->groups, context->groups, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BUFFER_BARRIER_BIT);
    context->currentBuffer = 1 - context->currentBuffer;

    // Step 2: Solve pressure Poisson equation (∇²p = divergence)
    // Store pressure in FZ component during solve
    int iteration = 0;
    bool converged = false;
    while (iteration < MAX_PRESSURE_ITERATIONS && !converged) {
        glUniform1i(context->uMode, 5); // Pressure solve mode (Jacobi iteration)
        glUniform1f(context->uDt, dt);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, context->ssbo[context->currentBuffer]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, context->ssbo[1 - context->currentBuffer]);
        glDispatchCompute(context->groups, context->groups, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BUFFER_BARRIER_BIT);
        context->currentBuffer = 1 - context->currentBuffer;

        // TODO: Could add convergence check here by reading back divergence residual
        // For now, fixed iteration count or early exit
        iteration++;
        if (iteration >= MAX_PRESSURE_ITERATIONS) {
            converged = true; // Max iterations reached
        }
    }

    // Step 3: Project velocity (subtract pressure gradient)
    glUniform1i(context->uMode, 6); // Pressure projection mode
    glUniform1f(context->uDensity, density); // Use density as proportionality constant
    glUniform1f(context->uDt, dt);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, context->ssbo[context->currentBuffer]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, context->ssbo[1 - context->currentBuffer]);
    glDispatchCompute(context->groups, context->groups, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BUFFER_BARRIER_BIT);
}

void SimFluid::update(float simBudget, float& simTime) {
    if (!context || !initialized) return;

    if (targetTimestep == 0.0f && context->targetStepDt > 0) {
        targetTimestep = context->targetStepDt;
        calculateTimestep();
    }

    accumulator += simBudget;

    // Full Navier-Stokes pipeline:
    // 1. Semi-Lagrangian advection + diffusion
    // 2. Pressure projection (divergence → pressure solve → velocity correction)
    while (accumulator >= timestep) {
        glUniform1f(context->uDiffusion, kinematicViscosity);
        glUniform1f(context->uDensity, density);
        glUniform1i(context->uMode, 2); // Fluid advection mode
        glUniform1f(context->uDt, timestep);

        // First pass: advection + diffusion
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, context->ssbo[context->currentBuffer]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, context->ssbo[1 - context->currentBuffer]);
        glDispatchCompute(context->groups, context->groups, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BUFFER_BARRIER_BIT);
        context->currentBuffer = 1 - context->currentBuffer;

        // Second pass: compute divergence, solve pressure, project velocity
        computeDivergenceAndPressure(timestep);

        accumulator -= timestep;
        simTime += timestep;
    }
}

void SimFluid::reset() {
    if (!context) return;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, context->ssbo[0]);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, context->ssbo[1]);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);

    accumulator = 0.0f;
    context->currentBuffer = 0;
}

REGISTER_SIMULATION(3, "Fluid", SimFluid)
