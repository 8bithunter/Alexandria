// =============================================================================
// Console Launcher for Field Simulations
// =============================================================================

#include <iostream>
#include <string>
#include <cstring>

// Forward declaration
int runSimulation(int mode);

static void printBanner() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                                ║\n";
    std::cout << "║   _____/\\\\\\\\\\_____/\\\\\\\\\\\\\\\_____/\\\\\\\\\\\\\\\_____/\\\    ║\n";
    std::cout << "║   ___/\\\\\\\\\\\\\\\\\\\\\\\__/\\\\\\\\\\\\\\\\\\\\\\\__      ║\n";
    std::cout << "║   __/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\____\/\\\\\\\\\\\\\\\\\\\\\\\\\\    ║\n";
    std::cout << "║   _\/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\____\/\\\\\\\\\\\\\\\\\\\\\\      ║\n";
    std::cout << "║   \/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\____\/\\\\\\\\\\\\\\\        ║\n";
    std::cout << "║   \/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\////////////\\\\\\\____\/\\\\\\\\\\\         ║\n";
    std::cout << "║   \/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\____\/\\\\\\\          ║\n";
    std::cout << "║   \/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\____\/\\\           ║\n";
    std::cout << "║   _\/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\____\/_           ║\n";
    std::cout << "║   __\\/________\\/_____\\/////////____\\/////////____\\/____\\/___         ║\n";
    std::cout << "║                                                                ║\n";
    std::cout << "║   GPU Field Simulation Console                               ║\n";
    std::cout << "║   Type 'run <diffusion|wave|fluid|schrodinger>' to start       ║\n";
    std::cout << "║   Type 'help' to see available commands                        ║\n";
    std::cout << "║                                                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

static void printHelp() {
    std::cout << "Available commands:\n";
    std::cout << "  run <sim>   Launch a simulation by name\n";
    std::cout << "              Names: diffusion, wave, fluid, schrodinger\n";
    std::cout << "  help        Show this help message\n";
    std::cout << "  exit        Exit the launcher\n";
    std::cout << "  cls         Clear the screen\n";
    std::cout << "\n";
    std::cout << "While in a simulation:\n";
    std::cout << "  ESC         Return to this console\n";
    std::cout << "  1-4         Switch simulation mode\n";
    std::cout << "  Space       Pause/resume\n";
    std::cout << "  R           Reset field\n";
    std::cout << "  Mouse       Paint/rotate/zoom (see window for details)\n";
    std::cout << "\n";
}

static void printModeBanner() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Valid simulation modes:                                      ║\n";
    std::cout << "║   diffusion  - Diffusion equation: ∂u/∂t = D ∇²u             ║\n";
    std::cout << "║   wave       - Wave equation: ∂²u/∂t² = c² ∇²u               ║\n";
    std::cout << "║   fluid      - Semi-Lagrangian advection + viscosity         ║\n";
    std::cout << "║   schrodinger- Schrödinger: i ∂ψ/∂t = –½ ∇²ψ                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    // Command line mode: run specific simulation
    if (argc == 2) {
        // Direct mode number
        int mode = atoi(argv[1]);
        if (mode >= 0 && mode <= 3) {
            std::cout << "Launching simulation mode " << mode << "...\n";
            return runSimulation(mode);
        }
    }
    else if (argc >= 3 && strcmp(argv[1], "run") == 0) {
        // Named simulation mode
        int mode = -1;
        if (strcmp(argv[2], "diffusion") == 0) mode = 0;
        else if (strcmp(argv[2], "wave") == 0) mode = 1;
        else if (strcmp(argv[2], "fluid") == 0) mode = 2;
        else if (strcmp(argv[2], "schrodinger") == 0 || strcmp(argv[2], "schro") == 0) mode = 3;

        if (mode >= 0) {
            std::cout << "Launching " << argv[2] << " simulation...\n";
            return runSimulation(mode);
        }
        else {
            std::cerr << "Unknown simulation: " << argv[2] << "\n";
            std::cerr << "Use: run <diffusion|wave|fluid|schrodinger>\n";
            std::cerr << "Or: <0|1|2|3> for direct mode\n";
            return 1;
        }
    }

    // Interactive console mode
    printBanner();

    while (true) {
        std::cout << "> ";
        std::string command;
        std::getline(std::cin, command);

        // Trim leading/trailing spaces
        while (!command.empty() && (command[0] == ' ' || command[0] == '\t'))
            command.erase(0, 1);
        while (!command.empty() && (command[command.length()-1] == ' ' || command[command.length()-1] == '\t'))
            command.pop_back();

        if (command.empty()) {
            continue;
        }
        else if (command == "exit" || command == "quit" || command == "q") {
            break;
        }
        else if (command == "help" || command == "h" || command == "?") {
            printHelp();
        }
        else if (command == "banner" || command == "b") {
            printBanner();
        }
        else if (command == "modes" || command == "m") {
            printModeBanner();
        }
        else if (command == "cls" || command == "clear") {
            #ifdef _WIN32
            system("cls");
            #else
            system("clear");
            #endif
            printBanner();
            std::cout << "Console cleared. Type 'help' for commands.\n";
        }
        else if (command.find("run ") == 0) {
            std::string sim = command.substr(4);

            // Trim again in case there were multiple spaces
            while (!sim.empty() && (sim[0] == ' ' || sim[0] == '\t'))
                sim.erase(0, 1);
            while (!sim.empty() && (sim[sim.length()-1] == ' ' || sim[sim.length()-1] == '\t'))
                sim.pop_back();

            if (sim == "diffusion") {
                runSimulation(0);
                printBanner();
            }
            else if (sim == "wave") {
                runSimulation(1);
                printBanner();
            }
            else if (sim == "fluid") {
                runSimulation(2);
                printBanner();
            }
            else if (sim == "schrodinger" || sim == "schro" || sim == "quantum") {
                runSimulation(3);
                printBanner();
            }
            else if (sim == "modes" || sim == "mode") {
                printModeBanner();
            }
            else if (sim.empty()) {
                std::cout << "Error: No simulation specified.\n";
                std::cout << "Usage: run <diffusion|wave|fluid|schrodinger>\n";
                std::cout << "Type 'run modes' to see mode numbers.\n";
            }
            else {
                std::cout << "Unknown simulation: " << sim << "\n";
                std::cout << "Available: diffusion, wave, fluid, schrodinger\n";
                std::cout << "Type 'run modes' to see all modes.\n";
            }
        }
        else if (!command.empty()) {
            std::cout << "Unknown command: " << command << "\n";
            std::cout << "Type 'help' for available commands\n";
        }
    }

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Goodbye! ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    return 0;
}
