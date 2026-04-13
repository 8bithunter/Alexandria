#!/usr/bin/env bash
# Script to insert geometry code into MainGPU.cpp

set -e

MAINFILE="C:/Users/eight/source/repos/alexandria/MainGPU.cpp"
TEMPFILE="${MAINFILE}.tmp"

# Split at line 504 (before UI layout)
head -503 "${MAINFILE}" > "${TEMPFILE}"

cat >> "${TEMPFILE}" << 'GEOMETRY'

// ── Topomap geometry ───────────────────────────────────────────────────────
float topomapVerts[] = {
    -1, -1, 0, 0,
     1, -1, 1, 0,
    -1,  1, 0, 1,
     1, -1, 1, 0,
     1,  1, 1, 1,
    -1,  1, 0, 1
};
unsigned int topomapVAO, topomapVBO;
glGenVertexArrays(1, &topomapVAO); glGenBuffers(1, &topomapVBO);
glBindVertexArray(topomapVAO);
glBindBuffer(GL_ARRAY_BUFFER, topomapVBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(topomapVerts), topomapVerts, GL_STATIC_DRAW);
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0); // position
glEnableVertexAttribArray(0);
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float))); // texcoord
glEnableVertexAttribArray(1);
glBindVertexArray(0);

// ── 3D Axes geometry ──────────────────────────────────────────────────────
struct AxesVert { float x, y, z; float r, g, b; };
std::vector<AxesVert> axesVerts;
float axisLength = 2.0f;
// X axis (red)
axesVerts.push_back({0, 0, 0, 1, 0, 0});
axesVerts.push_back({axisLength, 0, 0, 1, 0, 0});
// Y axis (green)
axesVerts.push_back({0, 0, 0, 0, 1, 0});
axesVerts.push_back({0, axisLength, 0, 0, 1, 0});
// Z axis (blue)
axesVerts.push_back({0, 0, 0, 0, 0, 1});
axesVerts.push_back({0, 0, axisLength, 0, 0, 1});
// Add tick marks every 0.1 units
for (int i = 1; i <= 20; ++i) {
    float t = i * 0.1f;
    // X axis ticks
    axesVerts.push_back({t, -0.05f, 0, 0.5f, 0, 0});
    axesVerts.push_back({t, 0.05f, 0, 0.5f, 0, 0});
    // Y axis ticks
    axesVerts.push_back({-0.05f, t, 0, 0, 0.5f, 0});
    axesVerts.push_back({0.05f, t, 0, 0, 0.5f, 0});
    // Z axis ticks
    axesVerts.push_back({-0.05f, 0, t, 0, 0, 0.5f});
    axesVerts.push_back({0.05f, 0, t, 0, 0, 0.5f});
}
unsigned int axesVAO, axesVBO;
glGenVertexArrays(1, &axesVAO); glGenBuffers(1, &axesVBO);
glBindVertexArray(axesVAO);
glBindBuffer(GL_ARRAY_BUFFER, axesVBO);
glBufferData(GL_ARRAY_BUFFER, axesVerts.size() * sizeof(AxesVert), axesVerts.data(), GL_STATIC_DRAW);
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(AxesVert), (void*)0); // position
glEnableVertexAttribArray(0);
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(AxesVert), (void*)(3 * sizeof(float))); // color
glEnableVertexAttribArray(1);
glBindVertexArray(0);
GEOMETRY

tail -n +504 "${MAINFILE}" >> "${TEMPFILE}"

mv "${TEMPFILE}" "${MAINFILE}"

echo "Geometry sections inserted successfully!"
