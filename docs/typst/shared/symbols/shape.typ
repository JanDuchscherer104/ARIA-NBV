#let shape = (
    // Batch size.
    B: $B$,
    // Generic count.
    N: $N_q$,
    // Number of candidates.
    Nq: $N_q$,
    // Trajectory length / time steps.
    Tlen: $T$,
    // Point count.
    P: $P$,
    // Max points after subsampling.
    Pmax: $P_"max"$,
    // Projected points.
    Pproj: $P_"proj"$,
    // Frustum points.
    Pfr: $P_"fr"$,
    // Feature dimension (generic).
    D: $D$,
    // Height.
    H: $H$,
    // Width.
    Wdim: $W$,
    // Image height/width (pixel space).
    Himg: $H_"img"$,
    Wimg: $W_"img"$,
    // Voxel grid size.
    Vvox: $V$,
    // Global pooling dim.
    Gpool: $G_"pool"$,
    // Global projection dim.
    Gproj: $G_"proj"$,
    // Semidense projection grid size.
    Gsem: $G_"sem"$,
    // Mesh vertex count.
    M: $M$,
    // Ordinal bins.
    K: $K$,
    // Per-point semidense feature dimension (e.g., XYZ + extras).
    Csem: $C_"sem"$,
    // Feature channel / embedding dimensions.
    Fin: $F_"in"$,
    // Scene-field channel dimension.
    Ffield: $F_"field"$,
    Fpose: $F_"pose"$,
    Fpe: $F_"pe"$,
    Fq: $F_q$,
    Fg: $F_g$,
    Ftau: $F_tau$,
    Fproj: $F_"proj"$,
    Fcnn: $F_"cnn"$,
    Ftok: $F_"tok"$,
    Ffr: $F_"fr"$,
    Fpt: $F_"pt"$,
    Faux: $F_"aux"$,
    Fhead: $F_"head"$,
    Fhid: $F_"hid"$,
  )
