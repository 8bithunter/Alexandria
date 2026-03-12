#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include "FieldVertex.cpp"

// ─── Rotation state (accessed in GLFW callbacks) ─────────────────────────────
struct RotationState
{
    bool   dragging = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    float  yaw = 0.0f;    // rotates left/right (around Y)
    float  pitch = 0.0f;    // rotates up/down   (around X)
    float  sensitivity = 0.005f;
};

static RotationState rot;

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (action == GLFW_PRESS)
        {
            rot.dragging = true;
            glfwGetCursorPos(window, &rot.lastMouseX, &rot.lastMouseY);
        }
        else if (action == GLFW_RELEASE)
        {
            rot.dragging = false;
        }
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (!rot.dragging) return;

    float dx = static_cast<float>(xpos - rot.lastMouseX);
    float dy = static_cast<float>(ypos - rot.lastMouseY);

    rot.yaw += dx * rot.sensitivity;
    rot.pitch += dy * rot.sensitivity;

    // Clamp pitch so the view never flips past vertical
    const float halfPi = 1.5707963f;
    if (rot.pitch > halfPi) rot.pitch = halfPi;
    if (rot.pitch < -halfPi) rot.pitch = -halfPi;

    rot.lastMouseX = xpos;
    rot.lastMouseY = ypos;
}

// ─── Build Ry(yaw) * Rx(pitch) as a column-major 4×4 matrix ─────────────────
void buildRotationMatrix(float yaw, float pitch, float* mat)
{
    float cy = cosf(yaw), sy = sinf(yaw);
    float cx = cosf(pitch), sx = sinf(pitch);

    // Column 0
    mat[0] = cy;    mat[1] = 0.0f; mat[2] = -sy;    mat[3] = 0.0f;
    // Column 1
    mat[4] = sy * sx; mat[5] = cx;   mat[6] = cy * sx; mat[7] = 0.0f;
    // Column 2
    mat[8] = sy * cx; mat[9] = -sx;   mat[10] = cy * cx; mat[11] = 0.0f;
    // Column 3 (no translation)
    mat[12] = 0.0f;   mat[13] = 0.0f;  mat[14] = 0.0f;   mat[15] = 1.0f;
}

// ─── Shaders ──────────────────────────────────────────────────────────────────
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
uniform mat4 uRotation;
out vec3 vColor;
void main()
{
    vColor = aColor;
    gl_Position = uRotation * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main()
{
    FragColor = vec4(vColor, 1.0);
}
)";

/*
int main()
{
    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 800, "Field Simulation", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD\n";
        return -1;
    }

    glViewport(0, 0, 800, 800);
    glEnable(GL_DEPTH_TEST);   // needed for correct 3-D overlap when rotated

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Cache uniform location — never needs to be looked up again
    int rotUniformLoc = glGetUniformLocation(shaderProgram, "uRotation");

    unsigned int gridVAO, gridVBO;
    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);

    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    double lastTime = glfwGetTime();

    int    resolution = 127;
    double spacingD = (resolution > 1) ? (2.0 / (resolution - 1)) : 0.0;
    float  spacingF = static_cast<float>(spacingD);

    std::vector<FieldVertex*> grid(resolution * resolution);
    auto G = [&](int i, int j) -> FieldVertex*& { return grid[i * resolution + j]; };

    for (int i = 0; i < resolution; ++i)
        for (int j = 0; j < resolution; ++j)
        {
            double x = -1.0 + j * spacingD;
            double y = -1.0 + i * spacingD;
            double initFieldX = (x == 0.0 && y == 0.0) ? 100.0 : -100.0;
            G(i, j) = new FieldVertex(x, y, 0.0, spacingD, initFieldX, 0.0, 0.0);
        }

    for (int i = 0; i < resolution; ++i)
        for (int j = 0; j < resolution; ++j)
        {
            FieldVertex* v = G(i, j);
            v->neighbourUp = (i + 1 < resolution) ? G(i + 1, j) : G(0, j);
            v->neighbourDown = (i - 1 >= 0) ? G(i - 1, j) : G(resolution - 1, j);
            v->neighbourLeft = (j - 1 >= 0) ? G(i, j - 1) : G(i, resolution - 1);
            v->neighbourRight = (j + 1 < resolution) ? G(i, j + 1) : G(i, 0);
            v->neighbourOut = v;
            v->neighbourIn = v;
        }

    // Pre-compute vertex positions (never change)
    std::vector<float> posX(resolution), posY(resolution);
    for (int k = 0; k < resolution; ++k)
        posX[k] = posY[k] = -1.0f + k * spacingF;

    const int cellCount = (resolution - 1) * (resolution - 1);
    std::vector<float> meshData(cellCount * 2 * 3 * 6);

    // Allocate GPU buffer once
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, meshData.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    float rotMatrix[16];

    while (!glfwWindowShouldClose(window))
    {
        double currentTime = glfwGetTime();
        double dt = currentTime - lastTime;
        lastTime = currentTime;
        if (dt > 0.01) dt = 0.01;

        // ── Simulation ──────────────────────────────────────────────────────
        for (int k = 0; k < resolution * resolution; ++k)
        {
            FieldVertex* v = grid[k];
            v->calculateddt();
            v->updateVisuals();
            v->updateField(dt);
        }

        // ── Build mesh ──────────────────────────────────────────────────────
        int idx = 0;
        for (int i = 0; i < resolution - 1; ++i)
        {
            for (int j = 0; j < resolution - 1; ++j)
            {
                FieldVertex* bl = G(i, j);
                FieldVertex* br = G(i, j + 1);
                FieldVertex* tl = G(i + 1, j);
                FieldVertex* tr = G(i + 1, j + 1);

                float r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4;
                bl->getColorFloat(r1, g1, b1);
                br->getColorFloat(r2, g2, b2);
                tl->getColorFloat(r3, g3, b3);
                tr->getColorFloat(r4, g4, b4);

                float cr = 0.25f * (r1 + r2 + r3 + r4);
                float cg = 0.25f * (g1 + g2 + g3 + g4);
                float cb = 0.25f * (b1 + b2 + b3 + b4);

                float bx = posX[j], by = posY[i];
                float rx = posX[j + 1], ty = posY[i + 1];

                // Read Z height from each corner's field value.
                // The scale factor maps field magnitude into NDC-friendly range —
                // tweak it to taste, or swap `v->z` for whichever member you prefer
                float zbl = static_cast<float>(bl->getZ());
                float zbr = static_cast<float>(br->getZ());
                float ztl = static_cast<float>(tl->getZ());
                float ztr = static_cast<float>(tr->getZ());

                // Triangle 1  (bl, br, tl)
                meshData[idx++] = bx; meshData[idx++] = by; meshData[idx++] = zbl;
                meshData[idx++] = cr; meshData[idx++] = cg; meshData[idx++] = cb;
                meshData[idx++] = rx; meshData[idx++] = by; meshData[idx++] = zbr;
                meshData[idx++] = cr; meshData[idx++] = cg; meshData[idx++] = cb;
                meshData[idx++] = bx; meshData[idx++] = ty; meshData[idx++] = ztl;
                meshData[idx++] = cr; meshData[idx++] = cg; meshData[idx++] = cb;
                // Triangle 2  (br, tr, tl)
                meshData[idx++] = rx; meshData[idx++] = by; meshData[idx++] = zbr;
                meshData[idx++] = cr; meshData[idx++] = cg; meshData[idx++] = cb;
                meshData[idx++] = rx; meshData[idx++] = ty; meshData[idx++] = ztr;
                meshData[idx++] = cr; meshData[idx++] = cg; meshData[idx++] = cb;
                meshData[idx++] = bx; meshData[idx++] = ty; meshData[idx++] = ztl;
                meshData[idx++] = cr; meshData[idx++] = cg; meshData[idx++] = cb;
            }
        }

        // ── Render ──────────────────────────────────────────────────────────
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        buildRotationMatrix(rot.yaw, rot.pitch, rotMatrix);
        glUniformMatrix4fv(rotUniformLoc, 1, GL_FALSE, rotMatrix);

        glBindVertexArray(gridVAO);
        glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
            meshData.size() * sizeof(float),
            meshData.data());

        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(meshData.size() / 6));

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    for (int k = 0; k < resolution * resolution; ++k)
        delete grid[k];

    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
    glfwTerminate();
    return 0;
}
*/