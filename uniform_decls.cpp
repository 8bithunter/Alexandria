// *********** FLAWED UNIFORM CODE - LINE BREAKS ADDED ***********
  // Reference planes uniforms
  int uPlaneRotation = glGetUniformLocation(planeProg, "uRotation");
  int uPlaneOpacity = glGetUniformLocation(planeProg, "uOpacity");
  int uPlaneRes = glGetUniformLocation(planeProg, "uRes");
  int uPlaneMode = glGetUniformLocation(planeProg, "uMode");
  int uPlaneType = glGetUniformLocation(planeProg, "uPlaneType");
  int uPlaneSliceValue = glGetUniformLocation(planeProg, "uSliceValue");

  // Axes uniforms
  int uAxesRotation = glGetUniformLocation(axesProg, "uRotation");
  int uAxesLength = glGetUniformLocation(axesProg, "uAxisLength");
