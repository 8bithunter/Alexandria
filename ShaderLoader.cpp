#include "ShaderLoader.h"
#include <fstream>
#include <iostream>

std::string loadShaderFile(const char* filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filename << std::endl;
        return "";
    }

    // Get file size
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string buffer;
    buffer.resize(size);

    if (!file.read(&buffer[0], size)) {
        std::cerr << "Failed to read shader file: " << filename << std::endl;
        return "";
    }

    return buffer;
}
