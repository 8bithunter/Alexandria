#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include "FieldVertex.cpp"

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
out vec3 vColor;
void main()
{
    vColor = aColor;
    gl_Position = vec4(aPos, 1.0);
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

int main()
{
    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 800, "Field Simulation", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD\n";
        return -1;
    }

    glViewport(0, 0, 800, 800);

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

    int resolution = 128;
    double spacing = (resolution > 1) ? (2.0 / (resolution - 1)) : 0.0;

    std::vector<std::vector<FieldVertex*>> grid(resolution, std::vector<FieldVertex*>(resolution));

    for (int i = 0; i < resolution; ++i)
    {
        for (int j = 0; j < resolution; ++j)
        {
            double x = -1.0 + j * spacing;
            double y = -1.0 + i * spacing;

            double initFieldX = (x > 0.0 && y > 0.0) ? 100.0 : -100;

            grid[i][j] = new FieldVertex(x, y, 0.0, spacing, initFieldX, 0.0, 0.0);
        }
    }

    for (int i = 0; i < resolution; ++i)
    {
        for (int j = 0; j < resolution; ++j)
        {
            FieldVertex* v = grid[i][j];

			// maps the edges to itsself
            /*
            v->neighbourUp = (i + 1 < resolution) ? grid[i + 1][j] : v;
            v->neighbourDown = (i - 1 >= 0) ? grid[i - 1][j] : v;
            v->neighbourLeft = (j - 1 >= 0) ? grid[i][j - 1] : v;
            v->neighbourRight = (j + 1 < resolution) ? grid[i][j + 1] : v;
            */

			//maps the edges to the opposite edge, creating a toroidal topology
            
            v->neighbourUp = (i + 1 < resolution) ? grid[i + 1][j] : grid[0][j];
            v->neighbourDown = (i - 1 >= 0) ? grid[i - 1][j] : grid[resolution - 1][j];
            v->neighbourLeft = (j - 1 >= 0) ? grid[i][j - 1] : grid[i][resolution - 1];
            v->neighbourRight = (j + 1 < resolution) ? grid[i][j + 1] : grid[i][0];
            
            v->neighbourOut = v;
            v->neighbourIn = v;

        }
    }

    while (!glfwWindowShouldClose(window))
    {
        double currentTime = glfwGetTime();
        double dt = currentTime - lastTime;
        lastTime = currentTime;

        if (dt > 0.01)
            dt = 0.01; // SUPER DUPER IMPORTANT FOR STABILIY DONT REMOVE PRETTY PLEASE

        for (int i = 0; i < resolution; ++i)
        {
            for (int j = 0; j < resolution; ++j)
            {
                FieldVertex* v = grid[i][j];

                v->calculateddt();
                v->updateColour();

				v->updateField(dt);
            }
        }

        std::vector<float> meshData;

        for (int i = 0; i < resolution - 1; ++i)
        {
            for (int j = 0; j < resolution - 1; ++j)
            {
                FieldVertex* bl = grid[i][j];
                FieldVertex* br = grid[i][j + 1];
                FieldVertex* tl = grid[i + 1][j];
                FieldVertex* tr = grid[i + 1][j + 1];

                float r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4;
                bl->getColorFloat(r1, g1, b1);
                br->getColorFloat(r2, g2, b2);
                tl->getColorFloat(r3, g3, b3);
                tr->getColorFloat(r4, g4, b4);

                float cr = 0.25f * (r1 + r2 + r3 + r4);
                float cg = 0.25f * (g1 + g2 + g3 + g4);
                float cb = 0.25f * (b1 + b2 + b3 + b4);

                float blx = -1.0f + j * spacing;
                float bly = -1.0f + i * spacing;
                float brx = -1.0f + (j + 1) * spacing;
                float bry = bly;
                float tlx = blx;
                float tly = -1.0f + (i + 1) * spacing;
                float trx = brx;
                float try_ = tly;
                float z = 0.0f;

                meshData.insert(meshData.end(), {
                    blx,bly,z, cr,cg,cb,
                    brx,bry,z, cr,cg,cb,
                    tlx,tly,z, cr,cg,cb
                    });

                meshData.insert(meshData.end(), {
                    brx,bry,z, cr,cg,cb,
                    trx,try_,z, cr,cg,cb,
                    tlx,tly,z, cr,cg,cb
                    });
            }
        }

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(gridVAO);
        glBindBuffer(GL_ARRAY_BUFFER, gridVBO);

        glBufferData(GL_ARRAY_BUFFER,
            meshData.size() * sizeof(float),
            meshData.data(),
            GL_DYNAMIC_DRAW);

        glDrawArrays(GL_TRIANGLES, 0, meshData.size() / 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    for (int i = 0; i < resolution; ++i)
        for (int j = 0; j < resolution; ++j)
            delete grid[i][j];

    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);

    glfwTerminate();
    return 0;
}