#!/usr/bin/env bash
# Script to insert rendering code into MainGPU.cpp

set -e

MAINFILE="C:/Users/eight/source/repos/alexandria/MainGPU.cpp"
TEMPFILE="${MAINFILE}.tmp"

# Split AT line 731 (before glUseProgram(fieldProg) which is line 732)
# We want to insert after line 730 (after glDepthMask) and before 732
head -730 "${MAINFILE}" > "${TEMPFILE}"

cat >> "${TEMPFILE}" << 'RENDERCODE'

// ── Topographical map ────────────────────────────────────────────────────
glUseProgram(topomapProg);
glUniformMatrix4fv(uTMRotationU, 1, GL_FALSE, MVP);
glUniform1i(uTMResU, N);
glUniform1i(uTMModeU, simulationMode);
glUniform1f(uTMOpacityU, 0.8f);
glUniform1f(uTMYOffsetU, -1.5f);
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
glBindVertexArray(topomapVAO);
glDrawArrays(GL_TRIANGLES, 0, 6);

RENDERCODE

# Now lines 731-732 from original become after the insertion
tail -n +731 "${MAINFILE}" >> "${TEMPFILE}"

mv "${TEMPFILE}" "${MAINFILE}"

echo "Rendering section inserted successfully!"
