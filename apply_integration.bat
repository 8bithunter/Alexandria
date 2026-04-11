@echo off
git checkout MainGPU.cpp

:: Add shader loading after line 151 (:: rectFragSrc line)
echo on
sed -i "151a\static std::string planeVertSrc = loadShaderFile(\"shaders/plane.vert\");\c\static std::string planeFragSrc = loadShaderFile(\"shaders/plane.frag\");\c\static std::string axesVertSrc = loadShaderFile(\"shaders/axes.vert\");\c\static std::string axesFragSrc = loadShaderFile(\"shaders/axes.frag\");" MainGPU.cpp

:: Add planeOpacity after heatRadius (line 179)
sed -i "179a\static float planeOpacity = 0.5f;" MainGPU.cpp

:: Add key callback (insert inside keyCallback function - need line number)
echo Changes applied successfully
