// =============================================================================
// Simulation Runner Interface
// =============================================================================
#ifndef SIMULATION_RUNNER_H
#define SIMULATION_RUNNER_H

// Run a specific simulation mode
// mode: 0=diffusion, 1=wave, 2=fluid, 3=schrodinger
// Returns 0 on success, -1 on failure
int runSimulation(int mode);

#endif // SIMULATION_RUNNER_H