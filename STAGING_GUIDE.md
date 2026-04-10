# 📝 Git Staging Guide - Before You Commit

## ✅ Files TO STAGE (Add to Commit)

### Core Refactored Files
- MainGPU.cpp - REFACTORED with shader loading ✓
- ConsoleLauncher.cpp - Minor updates (title change) ✓
- ShaderLoader.cpp/.h - NEW shader loader ✓

### Shaders (NEW)
- shaders/compute.glsl ✓
- shaders/field.vert/.frag ✓
- shaders/text.vert/.frag ✓
- shaders/rect.vert/.frag ✓

### OOP Architecture (NEW)
- src/simulations/Simulation.h ✓
- src/simulations/SimulationRegistry.h/.cpp ✓
- src/simulations/SimDiffusion.h/.cpp ✓
- src/simulations/SimWave.h/.cpp ✓
- src/simulations/SimFluid.h/.cpp ✓
- src/simulations/SimSchrodinger.h/.cpp ✓

## 🚫 Files to IGNORE (Don't Commit)

### Backups (Will undelete if needed)
- MainGPU-legacy-backup.cpp ← This is a backup, not needed in repo
- Any *.tmp files

### Build Artifacts
- .vs/ directory
- x64/ directory
- *.tlog files

### Old Dead Code (Already deleted, should stay deleted)
- DiffusionSimulation.cpp
- WaveSimulation.cpp
- FluidSimulation.cpp
- SchrodingerSimulation.cpp
- FieldVertex.cpp
- Simulation.h (old)
- SimulationRegistry.cpp
- SimulationRunner.h (old)

## 🎯 Summary

**To Add:** 1 new directory (shaders/), 1 new directory (src/simulations/), ~15 new files, 2 modified files  **To Remove:** 7 old simulation files (already deleted) **To Ignore:** 1 backup file

**Total Impact:** +13 files, -7 files, net +6 files (all new/refactored code)

## ✅ Ready Status: YES

**You can safely commit these changes:**
```bash
git add MainGPU.cpp ConsoleLauncher.cpp
git add ShaderLoader.cpp ShaderLoader.h
git add shaders/
git add src/simulations/
git add -u  # Add modified tracked files
git restore --staged MainGPU-legacy-backup.cpp  # Exclude backup
```

