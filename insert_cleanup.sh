#!/usr/bin/env bash
# Script to insert cleanup code

set -e

MAINFILE="C:/Users/eight/source/repos/alexandria/MainGPU.cpp"
TEMPFILE="${MAINFILE}.tmp"

# Insert before glfwTerminate (which is at the end after all glDeleteProgram calls)
# Find line with glfwTerminate by counting down from end
tail -5 "${MAINFILE}" | head -1 > /dev/null  # Just to check structure

# Insert before the last line that has glfwTerminate
# First find the line number of glfwTerminate
TERMINATE_LINE=$(grep -n "glfwTerminate" "${MAINFILE}" | tail -1 | cut -d: -f1)
INSERT_BEFORE=$((TERMINATE_LINE - 1))

head -${INSERT_BEFORE} "${MAINFILE}" > "${TEMPFILE}"

cat >> "${TEMPFILE}" << 'CLEANUP'

// ── New resources cleanup ───────────────────────────────────────────────────
glDeleteBuffers(1, &topomapVBO);
glDeleteVertexArrays(1, &topomapVAO);
glDeleteBuffers(1, &axesVBO);
glDeleteVertexArrays(1, &axesVAO);
glDeleteProgram(topomapProg);
glDeleteProgram(axesProg);
CLEANUP

tail -n +${TERMINATE_LINE} "${MAINFILE}" >> "${TEMPFILE}"

mv "${TEMPFILE}" "${MAINFILE}"

echo "Cleanup section inserted successfully!"
