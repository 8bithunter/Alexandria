cmake_minimum_required(VERSION 3.20)

project(Alexandria LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

file(GLOB SIMULATION_SOURCES
    src/simulations/*.cpp
)

add_executable(Alexandria
    MainGPU.cpp
    ConsoleLauncher.cpp
    ShaderLoader.cpp
    glad.c

    ${SIMULATION_SOURCES}
)

target_include_directories(Alexandria PRIVATE
    .
    Libraries/include
    src
)

target_link_libraries(Alexandria PRIVATE
    glfw
    OpenGL::GL
)

file(COPY shaders
     DESTINATION ${CMAKE_BINARY_DIR})