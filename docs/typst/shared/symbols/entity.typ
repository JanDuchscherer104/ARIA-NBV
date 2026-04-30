#import "../terms.typ": RRI

#let entity = (
    // Entity set (objects of interest).
    E: $cal(E)$,
    // Entity-weight vector; use components as `#(symb.entity.w)_e`.
    w: $bold(w)$,
    // Mixing weight for the scene-level term.
    lambda_scene: $lambda_"scene"$,
    // Weighted objective (global + entity-specific terms).
    rri_total: $RRI_"total"$,
  )
