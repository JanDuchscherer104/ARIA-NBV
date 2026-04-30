#let obs = (
    // Logged RGB image stream.
    img_rgb: $bold(I)^"rgb"$,
    // Optional grayscale image stream (used by Hestia-style formulations).
    img_gray: $bold(I)^"gray"$,
    // Depth image / rendered depth observation.
    depth: $bold(D)$,
    // Pose stream along the trajectory.
    pose: $bold(X)$,
    // Pose / camera metadata bundle.
    meta: $bold(M)$,
    // Semidense point-cloud observation stream.
    points_semi: $bold(cal(P))^"semi"$,
    // Counterfactual / rendered geometry point-cloud stream.
    points_cf: $bold(cal(P))^"cf"$,
    // Geometry / voxel-grid observation bundle.
    grid: $bold(G)$,
    // Generic visibility / directional-observability cue.
    vis: $bold(V)$,
    // Target / look-at latent.
    lookat: $bold(L)$,
    // Cumulative face visibility tensor (Hestia-style).
    face_vis: $bold(F)$,
    // Instantaneous face visibility tensor (Hestia-style).
    face_vis_step: $bold(f)$,
    // Voxel center position.
    voxel_center: $bold(p)_v$,
    // Face normal vector.
    face_normal: $bold(n)$,
  )
