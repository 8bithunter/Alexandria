#ifndef SHADER_LOADER_H
#define SHADER_LOADER_H

#include <string>

// Load shader source code from file
// Returns empty string on failure (check with .empty())
std::string loadShaderFile(const char* filename);

#endif // SHADER_LOADER_H
