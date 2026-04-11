#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aPlaneType;
layout(location = 2) in float aSliceCoord;

uniform mat4 uRotation;
uniform int uRes;
uniform int uMode;
uniform float uOpacity;
uniform float uSliceValue;
uniform int uPlaneType;

out float vPlaneType;
out vec2 vTexCoord;
out float vSliceCoord;

void main()
{
    vec3 pos = aPos;

    // For slices, adjust position based on brush coordinate
    if (uPlaneType == 1) { // XZ plane - slice at Y = brush Y
        pos.y = uSliceValue;
    } else if (uPlaneType == 2) { // YZ plane - slice at X = brush X
        pos.x = uSliceValue;
    }

    gl_Position = uRotation * vec4(pos, 1.0);

    // Calculate texture coordinates based on plane type
    // Use vector swizzle to avoid conditional branching
    vec3 normalizedPos = pos / float(uRes);
    vTexCoord = (uPlaneType == 0) ? normalizedPos.xy :
                (uPlaneType == 1) ? normalizedPos.xz :
                                    normalizedPos.yz;

    vPlaneType = float(uPlaneType);
    vSliceCoord = pos[(uPlaneType == 0) ? 2 : (uPlaneType == 1) ? 1 : 0];
}
