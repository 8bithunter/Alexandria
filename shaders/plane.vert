#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aPlaneType;
layout(location = 2) in float aSliceCoord;

uniform mat4 uRotation;
uniform int uRes;
uniform int uMode;
uniform float uOpacity; // New opacity uniform
uniform float uSliceValue; // Brush position for slices
uniform int uPlaneType; // 0=XY, 1=XZ, 2=YZ

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

    // Pass texture coordinates based on plane type
    if (uPlaneType == 0) { // XY plane
        vTexCoord = pos.xy / float(uRes);
    } else if (uPlaneType == 1) { // XZ plane
        vTexCoord = vec2(pos.x / float(uRes), pos.z / float(uRes));
    } else { // YZ plane
        vTexCoord = vec2(pos.y / float(uRes), pos.z / float(uRes));
    }

    vPlaneType = float(uPlaneType);
    vSliceCoord = (uPlaneType == 0) ? pos.z : ((uPlaneType == 1) ? pos.y : pos.x);
}
