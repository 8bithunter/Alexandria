// =============================================================================
// Console Launcher for Field Simulations
// =============================================================================

#include <iostream>
#include <string>
#include <cstring>

// Forward declarations
int runSimulation(int mode);
void setResolution(int resolution);
void setCoefficient(float coefficient);

static void printBanner() {
    std::cout << "\n";
    std::cout << "___/\\\\\\\\\\\\\\\\\\_____/\\\\\\\\\\\\______________________________________________________________________/\\\\\\_____________________________________________\n"        
                 "__/\\\\\\\\\\\\\\\\\\\\\\\\\\__\\////\\\\\\_____________________________________________________________________\\/\\\\\\____________________________________________\n"       
                 "__/\\\\\\/////////\\\\\\____\\/\\\\\\_____________________________________________________________________\\/\\\\\\_________________/\\\\\\______________________\n"      
                 "__\\/\\\\\\_______\\/\\\\\\____\\/\\\\\\________/\\\\\\\\\\\\\\\\___/\\\\\\____/\\\\\\__/\\\\\\\\\\\\\\\\\\_____/\\\\/\\\\\\\\\\\\__________\\/\\\\\\___/\\\\/\\\\\\\\\\\\\\__\\///___/\\\\\\\\\\\\\\\\\\_________\n"     
                 "___\\/\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\____\\/\\\\\\______/\\\\\\/////\\\\\\_\\///\\\\\\/\\\\\\/__\\////////\\\\\\___\\/\\\\\\////\\\\\\____/\\\\\\\\\\\\\\\\\\__\\/\\\\\\/////\\\\\\__/\\\\\\_\\////////\\\\\\_______\n"    
                 "____\\/\\\\\\/////////\\\\\\____\\/\\\\\\_____/\\\\\\\\\\\\\\\\\\\\\\____\\///\\\\\\/______/\\\\\\\\\\\\\\\\\\\\__\\/\\\\\\__\\//\\\\\\__/\\\\\\////\\\\\\__\\/\\\\\\___\\///__\\/\\\\\\___/\\\\\\\\\\\\\\\\\\\\_____\n"   
                 "_____\\/\\\\\\_______\\/\\\\\\____\\/\\\\\\____\\//\\\\///////______/\\\\\\/\\\\\\____/\\\\\\/////\\\\\\__\\/\\\\\\___\\/\\\\\\_\\/\\\\\\__\\/\\\\\\__\\/\\\\\\_________\\/\\\\\\__/\\\\\\/////\\\\\\____\n"  
                 "______\\/\\\\\\_______\\/\\\\\\__/\\\\\\\\\\\\\\\\\\__\\//\\\\\\\\\\\\\\\\\\\\__/\\\\\\/\\///\\\\\\_\\//\\\\\\\\\\\\\\\\/\\\\_\\/\\\\\\___\\/\\\\\\_\\//\\\\\\\\\\\\\\/\\\\_\\/\\\\\\_________\\/\\\\\\_\\//\\\\\\\\\\\\\\\\/\\\\__\n" 
                 "_______\\///________\\///__\\/////////____\\//////////__\\///____\\///___\\////////\\//__\\///____\\///___\\///////\\//__\\///__________\\///___\\////////\\//__\n \n";
    std::cout << "------------------------------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "--------------------------------------------------------======= Physics  Library =======--------------------------------------------------------\n";
    std::cout << "------------------------------------------------------------------------------------------------------------------------------------------------\n\n";
    std::cout << "Type 'run <sim>' to start       \n";
    std::cout << "Type 'help' to see available commands                        \n";
    std::cout << "                                                                \n";
    std::cout << "\n";
}

static void printHelp() {
    std::cout << "Available commands:\n";
    std::cout << "  run <sim>   Launch a simulation by name\n";
    std::cout << "              Names: diffusion, wave, fluid, schrodinger\n";
    std::cout << "  setresolution <number> Set simulation resolution\n";
    std::cout << "  setcoefficient <number> Set diffusion coefficient\n";
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
    else if (command.find("setresolution ") == 0) {
        std::string valueStr = command.substr(14);
        if (!valueStr.empty()) {
            try {
                int value = std::stoi(valueStr);
                setResolution(value);
            }
            catch (...) {
                std::cout << "Invalid resolution value. Please provide a number.";
            }
        }
        else {
            std::cout << "Usage: setresolution <number>";
        }
    }
    else if (command.find("setcoefficient ") == 0) {
        std::string valueStr = command.substr(15);
        if (!valueStr.empty()) {
            try {
                float value = std::stof(valueStr);
                setCoefficient(value);
            }
            catch (...) {
                std::cout << "Invalid coefficient value. Please provide a number.";
            }
        }
        else {
            std::cout << "Usage: setcoefficient <number>";
        }
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
