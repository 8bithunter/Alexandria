#include "SimulationRegistry.h"

SimulationRegistry& SimulationRegistry::get() {
    static SimulationRegistry instance;
    return instance;
}

void SimulationRegistry::registerSimulation(int mode, const char* name, FactoryFunc factory) {
    registry[mode] = std::make_pair(std::string(name), factory);
}

std::unique_ptr<Simulation> SimulationRegistry::create(int mode, SimulationContext& ctx) {
    auto it = registry.find(mode);
    if (it == registry.end()) {
        return nullptr;
    }
    return it->second.second(ctx);
}

const char* SimulationRegistry::getName(int mode) const {
    auto it = registry.find(mode);
    if (it == registry.end()) {
        return nullptr;
    }
    return it->second.first.c_str();
}

std::vector<std::pair<int, const char*>> SimulationRegistry::list() const {
    std::vector<std::pair<int, const char*>> result;
    for (const auto& entry : registry) {
        result.push_back({entry.first, entry.second.first.c_str()});
    }
    return result;
}
