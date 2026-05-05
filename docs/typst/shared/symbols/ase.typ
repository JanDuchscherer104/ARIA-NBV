#let ase = (
    // GT mesh
    mesh: $bold(cal(M))_"GT"$,
    // Target-specific GT surface / mesh crop.
    mesh_target: $M_e$,
    // GT mesh faces / triangles.
    faces: $bold(cal(F))_"GT"$,
    // Trajectory
    traj: $bold(T)_"rig"^"w" (t)$,
    // Final trajectory pose
    traj_final: $bold(T)_"rig"^"w" (T)$,
    // Semi-dense PC
    points_semi: $bold(cal(P))_t$,
  )
