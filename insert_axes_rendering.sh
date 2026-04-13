#!/usr/bin/env bash
# Script to insert axes rendering code

set -e

MAINFILE="C:/Users/eight/source/repos/alexandria/MainGPU.cpp"
TEMPFILE="${MAINFILE}.tmp"

# Insert after line 752 (after glDepthMask(GL_TRUE))
head -752 "${MAINFILE}" > "${TEMPFILE}"

cat >> "${TEMPFILE}" << 'AXESRENDER'

// ── 3D axes ────────────────────────────────────────────────────────────────
glUseProgram(axesProg);
glUniformMatrix4fv(uAxesRotationU, 1, GL_FALSE, MVP);
glUniform3f(uAxesOriginU, -0.9f, -1.2f, -0.9f);
glUniform1f(uAxesLengthU, 2.0f / zoom);
glBindVertexArray(axesVAO);
glDrawArrays(GL_LINES, 0, 126);
AXESRENDER

tail -n +753 "${MAINFILE}" >> "${TEMPFILE}"

mv "${TEMPFILE}" "${MAINFILE}"

echo "Axes rendering section inserted successfully!"
