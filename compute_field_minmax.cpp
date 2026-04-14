// Insert this function before the main update loop:
// Function to compute min/max field values from SSBO
void computeFieldMinMax(unsigned int ssbo, int N, float& minVal, float& maxVal, int mode) {
    float* fieldData = new float[N * N * 10]; // STRIDE = 10
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    void* ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (ptr) {
        memcpy(fieldData, ptr, N * N * 10 * sizeof(float));
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        
        minVal = FLT_MAX;
        maxVal = -FLT_MAX;
        
        for (int i = 0; i < N * N; i++) {
            float val;
            if (mode >= 2) {
                // Fluid/Schrodinger: magnitude of complex field
                float re = fieldData[i * 10 + 0];
                float im = fieldData[i * 10 + 1];
                val = sqrtf(re*re + im*im);
            } else {
                // Diffusion/Wave: real field
                val = fieldData[i * 10 + 0];
            }
            
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    }
    delete[] fieldData;
}
