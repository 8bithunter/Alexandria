
    // ── UI layout ─────────────────────────────────────────────────────────────
    const float PX = 2.0f / 800.0f;
    const float CHAR_W = 20.0f * PX;
    const float CHAR_H = 30.0f * PX;
    const float CHAR_ADV = 22.0f * PX;
    const float MARGIN = 12.0f * PX;
    const float PAD = 6.0f * PX;
    const float TEXT_X = -1.0f + MARGIN;
    const float TEXT_Y = 1.0f - MARGIN;

    const int groups = (N + 15) / 16;
    int   current = 0;
    float simTime = 0.0f;
    double lastTime = glfwGetTime();
    float accumDiff = 0.0f;
    float accumWave = 0.0f;
    float accumSchrod = 0.0f;
    float accumFluid = 0.0f;

    // ── Main loop ─────────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(window) && !g_returnToConsole)
    {
        // ── Poll events FIRST so heatDX/heatDY are fresh before painting ──────
     // Optional: Check for ESC key directly (in case callback missed it)
     if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
       glfwSetWindowShouldClose(window, GLFW_TRUE);
       g_returnToConsole = true;
     }
        glfwPollEvents();

        // ── Wall-clock delta time ─────────────────────────────────────────────
        double nowTime = glfwGetTime();
        float  realDt = (float)(nowTime - lastTime);
        lastTime = nowTime;
        float simBudget = std::min(realDt, 4.0f / 60.0f) * SIM_SPEED;

        // ── Reset ─────────────────────────────────────────────────────────────
        if (resetRequested) {
            resetRequested = false;
            simTime = 0.0f; current = 0;
            accumDiff = accumWave = accumSchrod = accumFluid = 0.0f;
            for (int b = 0; b < 2; ++b) {
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[b]);
                glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        // ── Build MVP ─────────────────────────────────────────────────────────
        float R[16], T[16], P[16], TR[16], MVP[16];
        buildGlobeMatrix(rot.spin, rot.pitch, R);
        buildTranslation(0.0f, cameraY, -zoom, T);
        buildPerspective(3.14159265f / 3.0f, 1.0f, 0.01f, 100.0f, P);
        matMul(T, R, TR);
        matMul(P, TR, MVP);

        // ── Paint excitation ──────────────────────────────────────────────────
        if (heatActive) {
            float worldX, worldY;
            int cx = N / 2, cy = N / 2;
            if (unprojectToField(heatCurX, heatCurY, MVP, worldX, worldY)) {
                cx = (int)((worldX + 1.0f) * 0.5f * N);
                cy = (int)((worldY + 1.0f) * 0.5f * N);
            }

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[current]);

            if (simulationMode == 2) {
                if (fabsf(heatDX) > 0.1f || fabsf(heatDY) > 0.1f) {
                    // Unproject current and previous screen positions → world-space drag vector.
                    // This naturally accounts for camera spin and pitch.
                    float wx0, wy0, wx1, wy1;
                    bool ok0 = unprojectToField(heatCurX, heatCurY, MVP, wx0, wy0);
                    bool ok1 = unprojectToField(heatCurX - heatDX, heatCurY - heatDY, MVP, wx1, wy1);
                    if (ok0 && ok1) {
                        float velX = (wx0 - wx1) * heatValue * 0.1f;
                        float velY = (wy0 - wy1) * heatValue * 0.1f;
                        float vel[2] = { velX, velY };
                        for (int dy = -heatRadius; dy <= heatRadius; ++dy)
                            for (int dx = -heatRadius; dx <= heatRadius; ++dx) {
                                if (dx * dx + dy * dy > heatRadius * heatRadius) continue;
                                int gi = cy + dy, gj = cx + dx;
                                if (gi < 0 || gi >= N || gj < 0 || gj >= N) continue;
                                GLintptr off = (GLintptr)((gi * N + gj) * STRIDE) * sizeof(float);
                                glBufferSubData(GL_SHADER_STORAGE_BUFFER, off, 2 * sizeof(float), vel);
                            }
                    }
                }
            }
            else 
            {
                // Scalar modes: paint heatValue into FX.
                for (int dy = -heatRadius; dy <= heatRadius; ++dy)
                    for (int dx = -heatRadius; dx <= heatRadius; ++dx) {
                        if (dx * dx + dy * dy > heatRadius * heatRadius) continue;
                        int gi = cy + dy, gj = cx + dx;
                        if (gi < 0 || gi >= N || gj < 0 || gj >= N) continue;
                        GLintptr off = (GLintptr)((gi * N + gj) * STRIDE) * sizeof(float);
                        glBufferSubData(GL_SHADER_STORAGE_BUFFER, off, sizeof(float), &heatValue);
                    }
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        // Delta consumed — clear for next frame.
        heatDX = 0.0f;
        heatDY = 0.0f;

        // ── Compute pass ──────────────────────────────────────────────────────
        if (!paused) {
            glUseProgram(computeProg);
            glUniform1i(uResU, N);
            glUniform1f(uInvH2U, invH2);

            if (simulationMode == 0) {
                accumDiff += simBudget;
                glUniform1f(uDiffusionU, DIFFUSION);
                glUniform1i(uComputeModeU, 0);
                glUniform1f(uDtU, subDtDiff);
                while (accumDiff >= subDtDiff) {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
                    glDispatchCompute(groups, groups, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    current = 1 - current; accumDiff -= subDtDiff; simTime += subDtDiff;
                }
            }
            else if (simulationMode == 1) {
                accumWave += simBudget;
                glUniform1f(uDiffusionU, DIFFUSION);
                glUniform1i(uComputeModeU, 1);
                glUniform1f(uDtU, subDtWave);
                while (accumWave >= subDtWave) {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
                    glDispatchCompute(groups, groups, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    current = 1 - current; accumWave -= subDtWave; simTime += subDtWave;
                }
            }
            else if (simulationMode == 2) {
                accumFluid += simBudget;
                glUniform1f(uDiffusionU, 0.001f);   // kinematic viscosity
                glUniform1i(uComputeModeU, 2);
                glUniform1f(uDtU, subDtFluid);
                while (accumFluid >= subDtFluid) {
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
                    glDispatchCompute(groups, groups, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    current = 1 - current; accumFluid -= subDtFluid; simTime += subDtFluid;
                }
            }
            else {
                accumSchrod += simBudget;
                glUniform1f(uDiffusionU, DIFFUSION);
                glUniform1f(uDtU, subDtSchrod);
                while (accumSchrod >= subDtSchrod) {
                    glUniform1i(uComputeModeU, 3);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
                    glDispatchCompute(groups, groups, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    current = 1 - current;
                    glUniform1i(uComputeModeU, 4);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1 - current]);
                    glDispatchCompute(groups, groups, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    current = 1 - current; accumSchrod -= subDtSchrod; simTime += subDtSchrod;
                }
            }
        }

        // ── Field render ──────────────────────────────────────────────────────
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glDepthMask(simulationMode == 2 ? GL_FALSE : GL_TRUE); for transparency stuff
        glDepthMask(GL_TRUE);

        glUseProgram(fieldProg);
        glUniformMatrix4fv(uRotationU, 1, GL_FALSE, MVP);
        glUniform1i(uFieldModeU, simulationMode);
        glUniform1i(uFieldResU, N);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
        glBindVertexArray(fieldVAO);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)mesh.size());

        glDepthMask(GL_TRUE);

        // ── 2-D overlay ───────────────────────────────────────────────────────
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // — Timer —
        uint32_t charBuf[16];
        int numChars = formatSimTime(simTime, charBuf, 16);
        float textW = CHAR_W + (numChars - 1) * CHAR_ADV;

        glUseProgram(rectProg);
        glUniform2f(uRectOrigin, TEXT_X - PAD, TEXT_Y + PAD);
        glUniform2f(uRectSize, textW + 2 * PAD, CHAR_H + 2 * PAD);
        glUniform4f(uRectColor, 0, 0, 0, 0.65f);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glUseProgram(textProg);
        glUniform2f(uTxtOrigin, TEXT_X, TEXT_Y);
        glUniform2f(uTxtSize, CHAR_W, CHAR_H);
        glUniform1f(uTxtAdvance, CHAR_ADV);
        glUniform1uiv(uTxtFont, FONT_COUNT, FONT);
        glUniform1uiv(uTxtChars, numChars, charBuf);
        glUniform4f(uTxtColor, 1, 1, 1, 1);
        glBindVertexArray(quadVAO);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, numChars);

        // — Mode label —
        uint32_t modeBuf[3];
        int   numMode = formatModeLabel(simulationMode, modeBuf);
        float modeW = CHAR_W + (numMode - 1) * CHAR_ADV;
        float modeY = TEXT_Y - CHAR_H - PAD - MARGIN;

        glUseProgram(rectProg);
        glUniform2f(uRectOrigin, TEXT_X - PAD, modeY + PAD);
        glUniform2f(uRectSize, modeW + 2 * PAD, CHAR_H + 2 * PAD);
        glUniform4f(uRectColor, 0, 0, 0, 0.65f);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glUseProgram(textProg);
        glUniform2f(uTxtOrigin, TEXT_X, modeY);
        glUniform2f(uTxtSize, CHAR_W, CHAR_H);
        glUniform1f(uTxtAdvance, CHAR_ADV);
        glUniform1uiv(uTxtFont, FONT_COUNT, FONT);
        glUniform1uiv(uTxtChars, numMode, modeBuf);
        if (simulationMode == 0) glUniform4f(uTxtColor, 1.0f, 0.65f, 0.0f, 1.0f);
        else if (simulationMode == 1) glUniform4f(uTxtColor, 0.0f, 1.0f, 1.0f, 1.0f);
        else if (simulationMode == 2) glUniform4f(uTxtColor, 0.2f, 1.0f, 0.4f, 1.0f);
        else                        glUniform4f(uTxtColor, 1.0f, 0.3f, 1.0f, 1.0f);
        glBindVertexArray(quadVAO);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, numMode);

        glDisable(GL_BLEND);
        glfwSwapBuffers(window);
        // NOTE: glfwPollEvents() is at the TOP of the loop so heatDX/heatDY
        // are always fresh (non-zero) by the time painting runs.
    }

    glDeleteBuffers(2, ssbo);
    glDeleteBuffers(1, &fieldVBO);  glDeleteVertexArrays(1, &fieldVAO);
    glDeleteBuffers(1, &quadVBO);   glDeleteVertexArrays(1, &quadVAO);
    glDeleteProgram(computeProg);  glDeleteProgram(fieldProg);
    glDeleteProgram(textProg);     glDeleteProgram(rectProg);
    glfwTerminate();
    return 0;
}