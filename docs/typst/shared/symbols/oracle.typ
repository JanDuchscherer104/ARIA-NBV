#import "../terms.typ": RRI

#let oracle = (
    // Point set (use subscripts for time/candidate: #(symb.oracle.points)_t, #symb.oracle.points_q).
    points: $bold(cal(P))$,
    // Candidate point cloud.
    points_q: $bold(cal(P))_q$,
    // Candidate pose set
    candidates: $bold(cal(Q))$,
    // Candidate depth maps.
    depth_q: $bold(D)_q$,
    // Pixel-wise valid mask for candidate depth maps / projections.
    // (Used e.g. for rendered depth validity and projection validity.)
    mask_q: $bold(M)_q$,
    // Candidate camera intrinsics/extrinsics (non-PyTorch3D).
    cameras_q: $bold(bold(C))_q$,
    // Direction vector (sampling).
    dir: $bold(d)$,
    // Center / translation vector.
    center: $bold(c)$,
    // Offset vector.
    offset: $bold(o)$,
    // Accuracy term (P -> M).
    acc: $cal(A)$,
    // Completeness term (M -> P).
    comp: $cal(C)$,
    // Relative Reconstruction Improvement scalar.
    rri: $RRI$,
  )
