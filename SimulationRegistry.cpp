#include "Simulation.h"

// Forward declarations - these are implemented in the respective .cpp files
ISimulation* createDiffusionSimulation(int resolution);
ISimulation* createWaveSimulation(int resolution);
ISimulation* createFluidSimulation(int resolution);
ISimulation* createSchrodingerSimulation(int resolution);

// Master factory function
ISimulation* createSimulation(int mode, int resolution) {
    switch (mode) {
        case 0: return createDiffusionSimulation(resolution);
        case 1: return createWaveSimulation(resolution);
        case 2: return createFluidSimulation(resolution);
        case 3: return createSchrodingerSimulation(resolution);
        default: return nullptr;
    }
}
