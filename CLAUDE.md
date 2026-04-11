# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alexandria is a GPU-accelerated physics simulation engine using OpenGL compute shaders. It simulates various physics phenomena (diffusion, wave propagation, fluid dynamics, quantum mechanics) through a unified architecture with real-time visualization.

**Technology Stack:**
- C++20 with Visual Studio/MSBuild
- OpenGL 4.3+ compute shaders
- GLFW3 for window management
- Compiler: ClangCL (Debug) / v145 (Release)

## Build Commands

### Visual Studio (primary method)
```bash
# Open in Visual Studio
# Devenv Alexandria.sln
# Build → Build Solution (F7)
# Debug configuration uses ClangCL
```

### MSBuild (command line)
```bash
# Debug build (x64, ClangCL)
msbuild Alexandria.sln /p:Configuration=Debug /p:Platform=x64

# Release build (x64)
msbuild Alexandria.sln /p:Configuration=Release /p:Platform=x64
```

**Note:** This project uses Visual Studio project files (.vcxproj) not CMake. There is no CMakeLists.txt.

## Architecture

### Directory Structure
```
alexandria/
├── MainGPU.cpp                  # Main application with rendering loop
├── ConsoleLauncher.cpp         # Console entry point with CLI
├── ShaderLoader.h/.cpp         # Runtime shader file loader
├── glad.c                      # OpenGL function loader
├── Libraries/                  # External dependencies
│   ├── include/                # GLFW, GL headers
│   └── lib/                    # glfw3.lib
├── shaders/                    # GLSL shader files
│   ├── compute.glsl           # Main physics compute shader
│   ├── field.vert/.frag        # Field visualization shaders
│   ├── text.vert/.frag         # Bitmap text rendering
│   └── rect.vert/.frag         # Rectangle rendering
└── src/simulations/           # OOP simulation architecture
    ├── Simulation.h/.cpp      # Base abstract class
    ├── SimulationRegistry.h/.cpp  # Factory/registry pattern
    ├── SimDiffusion.h/.cpp    # Diffusion simulation
    ├── SimWave.h/.cpp         # Wave equation simulation
    ├── SimFluid.h/.cpp        # Fluid dynamics
    └── SimSchrodinger.h/.cpp  # Quantum mechanics
```

### Simulation Architecture

The simulation system uses a polymorphic design pattern:

1. **Simulation base class** - Pure abstract interface for all simulations
2. **SimulationRegistry** - Singleton factory pattern with automatic registration via macros
3. **Compute shader architecture** - All physics calculations run on GPU via OpenGL compute shaders (mode 0-4)

**Simulation Modes:**
- 0: Diffusion (∂u/∂t = D ∇²u)
- 1: Wave (∂²u/∂t² = c² ∇²u)
- 2: Fluid (semi-Lagrangian advection + viscosity)
- 3: Schrödinger (i ∂ψ/∂t = –½ ∇²ψ)

**Key Design Principles:**
- Mode numbers must stay 0-3 internally (UI is 1-4)
- All simulations use **ping-pong SSBO buffers** for stateful GPU computations
- Physics timestep is fixed; accumulator handles variable frame rates
- Compute shaders dispatched in work groups of 16x16

### Runtime Configuration

**Global Parameters (set via console or CLI):**
- `RES`: Simulation grid resolution (default: 200, max: 1000)
- `DIFFUSION`: Coefficient for diffusion/viscosity (default: 0.01)

**Command Line:**
```bash
# Direct mode selection
Project3.exe 0              # Run diffusion mode

# Named simulation
Project3.exe run diffusion  # Alternative syntax

# Interactive console
Project3.exe                # Console mode with "run <sim>" commands
```

## GPU Buffer Management

The engine uses OpenGL SSBOs (Shader Storage Buffer Objects):

1. **Main simulation SSBOs**: 2 buffers for ping-pong pattern
   - `ssbo[0]`, `ssbo[1]` in SimulationContext
   - `currentBuffer` tracks which is read vs write

2. **Fluid-specific SSBOs**:
   - `ssboDiv` - Divergence buffer
   - `ssboPres[2]` - Pressure solve ping-pong buffers

3. **Buffer utilization by mode**:
   - Single-pass: Diffusion, Wave, Schrödinger (uses 2 main buffers)
   - Multi-pass: Fluid (splits compute into multiple passes)

4. **Texture unit bindings** (compute shader):
   - `tex0`/`tex1` - Alternate field buffers
   - `texDiv` - Divergence
   - `texPres` - Pressure

## Shader System

**Shader Loading:**
- All shaders load from `shaders/` directory at runtime
- Compute shaders use `#version 430 core`
- Visualization shaders use `#version 330 core`
- Shader compilation errors print to console and trigger window close

**Compute Shader Modes (switch statement):**
- 0: Diffusion step
- 1: Wave step
- 2: Fluid advection
- 3: Fluid divergence
- 4: Fluid pressure solve
- 5: Fluid projection
- 6-7: Schrödinger real/imaginary updates

**Visualization Pipeline:**
1. Render text overlay (HUD info)
2. Render field using screen quad
3. Color mapping in fragment shader

## Input/Output

**Controls (when running a simulation):**
- `1-4`: Switch simulation mode
- `Space`: Pause/resume
- `R`: Reset field
- `LEFT-CLICK+DRAG`: Paint excitation
- `RIGHT-CLICK+DRAG`: Rotate view
- `Scroll`: Zoom
- `Ctrl+Scroll`: Change paint strength
- `Alt+Scroll`: Change brush radius
- `Shift+Scroll`: Camera up/down
- `ESC`: Return to console

**Console Commands:**
- `run <diffusion|wave|fluid|schrodinger>`: Start simulation
- `setresolution <N>`: Set grid resolution
- `setcoefficient <value>`: Set physics coefficient
- `help`: Show help
- `cls`: Clear screen
- `exit`: Exit program

## Important Notes

**Working Directory:** All shader paths are relative to the executable's working directory. Ensure you run from the project root where the `shaders/` directory exists.

**Dependencies:** The `Libraries/` directory must be present with GLFW headers and libraries. The project expects:
- `Libraries/include/GLFW/glfw3.h`
- `Libraries/include/glad/glad.h`
- `Libraries/lib/glfw3.lib`

**Development Workflow:**
1. Edit shader files in `shaders/` directory - no recompilation needed
2. Modify simulation classes in `src/simulations/`
3. Build via Visual Studio or MSBuild
4. Run executable from project root to ensure shaders load

**Performance:** The simulation runs entirely on GPU. Resolution directly impacts VRAM and compute time. The default 200x200 grid provides good performance.

**Branching Model:** This appears to use a "main" branch for development (not "master"). Check branch context when making changes.

## Integration Status

As of recent refactoring (see PHASE_1-7_COMPLETE.md):
- **Shaders extracted** to `.glsl` files - working ✓
- **ShaderLoader** implemented and operational ✓
- **OOP architecture** complete with 4 simulation classes ✓
- **SimulationRegistry** factory pattern functional ✓
- **Procedural code** still exists in MainGPU (legacy mode switching)
- **Integration point**: runSimulation() needs updating to use activeSimulation->update()

The structural foundation is complete. Full OOP integration requires replacing the mode-switching logic with polymorphic dispatch through SimulationRegistry.
