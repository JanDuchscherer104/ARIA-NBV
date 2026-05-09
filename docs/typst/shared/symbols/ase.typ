#let ase = (
    // GT mesh / surface.
    mesh: $cal(M)^"GT"$,
    // Target-specific GT surface / mesh crop.
    mesh_target: $cal(M)_e^"GT"$,
    // GT mesh faces / triangles.
    faces: $cal(F)^"GT"$,
    // Trajectory
    traj: $bold(T)_"rig"^"w" (t)$,
    // Final trajectory pose
    traj_final: $bold(T)_"rig"^"w" (T)$,
    // Semi-dense point set.
    points_semi: $cal(P)_t^"semi"$,
  )
