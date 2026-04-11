#!/bin/bash
git checkout MainGPU.cpp 2>/dev/null

# Add planeOpacity after heatRadius (line 180)
sed -i '180a\static float planeOpacity = 0.5f;' MainGPU.cpp

# Add plane rendering loop (line 790 - tri count to match XYZ)
sed -i '/for.*plane.*0.*2/d' MainGPU.cpp
sed -i '789a\n        for (int plane = 0; plane < 3; ++plane) {\c\        glUniform1i(uPlaneType, plane);\c\        glUniform1f(uPlaneSliceValue, 0.0f);\c\        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);\c\        }' MainGPU.cpp

echo "Plane integration fixes applied"
