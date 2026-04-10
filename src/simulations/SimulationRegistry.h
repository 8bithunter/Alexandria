#ifndef SIMULATION_REGISTRY_H
#define SIMULATION_REGISTRY_H

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include "Simulation.h"

// Singleton registry for simulation types
class SimulationRegistry {
public:
    using FactoryFunc = std::function<std::unique_ptr<Simulation>(SimulationContext&)>;

    static SimulationRegistry& get();

    // Register a new simulation type
    void registerSimulation(int mode, const char* name, FactoryFunc factory);

    // Factory method - create a simulation by mode number
    std::unique_ptr<Simulation> create(int mode, SimulationContext& ctx);

    // Get the name for a mode
    const char* getName(int mode) const;

    // List all available simulations (mode number, name)
    std::vector<std::pair<int, const char*>> list() const;

private:
    SimulationRegistry() = default;
    SimulationRegistry(const SimulationRegistry&) = delete;
    SimulationRegistry& operator=(const SimulationRegistry&) = delete;

private:
    std::unordered_map<int, std::pair<std::string, FactoryFunc>> registry;
};

// Macro for automatic registration
// Usage: REGISTER_SIMULATION(0, "Diffusion", SimDiffusion)
#define REGISTER_SIMULATION(mode, name, class_name)                    \
    static struct class_name##Registrar {                           \
        class_name##Registrar() {                                     \
            SimulationRegistry::get().registerSimulation(           \
                mode, name, [](SimulationContext& ctx) {          \
                    auto sim = std::make_unique<class_name>();      \
                    sim->init(ctx);                                  \
                    return sim;                                      \
                });                                                  \
        }                                                             \
    } class_name##Registrar_instance;

#endif // SIMULATION_REGISTRY_H
