glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[current]);
glBindVertexArray(planeVAO);
for (int plane = 0; plane < 3; ++plane) {
glUniform1i(uPlaneType, plane);
glUniform1f(uPlaneSliceValue, 0.0f);
glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}
glBindVertexArray(0);
glDisable(GL_BLEND);
