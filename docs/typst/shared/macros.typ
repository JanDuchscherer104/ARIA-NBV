// Shared macros and symbols for NBV project
// This file can be imported in both presentations and papers

// ============================================================================
// Text Styling Macros
// ============================================================================

/// Emphasize text in primary color (similar to current emph but explicit)
#let emph-color(body) = text(fill: rgb("fc5555"), body)

/// Italic text
#let textit(body) = text(style: "italic", body)

/// Bold italic text
#let textbf-it(body) = text(weight: "bold", style: "italic", body)

/// Bold text (for completeness)
#let textbf(body) = text(weight: "bold", body)

/// Colored bold text
#let emph-bold(body) = text(fill: rgb("fc5555"), weight: "bold", body)

/// Colored italic text
#let emph-it(body) = text(fill: rgb("fc5555"), style: "italic", body)

/// Monospace/code inline
#let code-inline(body) = text(font: "DejaVu Sans Mono", size: 0.9em, body)

// ============================================================================
// Abbreviations (short + long forms)
// ============================================================================
// Short and long forms are kept together for consistent usage in prose.

/// Relative Reconstruction Improvement
#let RRI = "RRI"
#let RRI_full = "Relative Reconstruction Improvement"

/// Coverage Ratio
#let CR = "CR"
#let CR_full = "Coverage Ratio"

/// Chamfer Distance
#let CD = "CD"
#let CD_full = "Chamfer Distance"

/// Next-Best-View
#let NBV = "NBV"
#let NBV_full = "Next-Best-View"

/// Ground Truth
#let GT = "GT"
#let GT_full = "Ground Truth"

/// Degrees of Freedom
#let DoF = "DoF"
#let DoF_full = "Degrees of Freedom"

/// 6 Degrees of Freedom
#let SixDoF = "6DoF"
#let SixDoF_full = "Six Degrees of Freedom"

/// 5 Degrees of Freedom
#let FiveDoF = "5DoF"
#let FiveDoF_full = "Five Degrees of Freedom"

/// Area Under Curve
#let AUC = "AUC"
#let AUC_full = "Area Under Curve"

/// Point Cloud
#let PC = "PC"
#let PC_full = "Point Cloud"

/// Multi-view Stereo
#let MVS = "MVS"
#let MVS_full = "Multi-view Stereo"

/// Simultaneous Localization and Mapping
#let SLAM = "SLAM"
#let SLAM_full = "Simultaneous Localization and Mapping"

/// Occupancy Grid
#let OccGrid = "Occupancy Grid"
#let OccGrid_full = "Occupancy Grid"

/// Aria Synthetic Environments
#let ASE = "ASE"
#let ASE_full = "Aria Synthetic Environments"

/// Egocentric Foundation Model 3D
#let EFM3D = "EFM3D"
#let EFM3D_full = "Egocentric Foundation Models for 3D understanding"

/// Egocentric Voxel Lifting
#let EVL = "EVL"
#let EVL_full = "Egocentric Voxel Lifting"

/// Aria Digital Twin
#let ADT = "ADT"
#let ADT_full = "Aria Digital Twin"

/// Aria Everyday Objects
#let AEO = "AEO"
#let AEO_full = "Aria Everyday Objects"

/// Scene Script Structured Language
#let SSL = "SSL"
#let SSL_full = "Scene Script Structured Language"

// ============================================================================
// Paper notation (symbols used throughout the Typst paper)
// ============================================================================
// Note: These are *labels* that can be injected into math via `#` interpolation
// inside `$ ... $`. We keep them centralized here to enforce consistent
// notation across sections.

/// Frame labels used in transform subscripts (T_{A<-B}).
#let fr_world = "w"
#let fr_rig = "rig"
#let fr_rig_ref = "r"
#let fr_cam = "q"
#let fr_voxel = "voxel"

/// Common short-hands for sets, tensors, and dimensions used in equations.
///
/// Access with `#symb.group.key` inside math (e.g., `$#(symb.oracle.points)_t$`).
/// Use `#(symb.oracle.points)_t` when applying scripts to avoid spacing issues.
#let symb = (
  frame: (
    // World frame label.
    w: $w$,
    // Rig frame label.
    r: $r$,
    // Camera frame label.
    c: $c$,
    // Candidate camera frame label.
    cq: $c_q$,
    // Voxel frame label.
    v: $v$,
    // Sampling frame label (gravity-aligned shell).
    s: $s$,
  ),
  ase: (
    // GT mesh
    mesh: $bold(cal(M))_"GT"$,
    // GT mesh faces / triangles.
    faces: $bold(cal(F))_"GT"$,
    // Trajectory
    traj: $bold(T)_"rig"^"w" (t)$,
    // Final trajectory pose
    traj_final: $bold(T)_"rig"^"w" (T)$,
    // Semi-dense PC
    points_semi: $bold(cal(P))_t$,
  ),
  oracle: (
    // Point set (use subscripts for time/candidate: #(symb.oracle.points)_t, #symb.oracle.points_q).
    points: $bold(cal(P))$,
    // Candidate point cloud.
    points_q: $bold(cal(P))_q$,
    // Candidate pose set
    candidates: $bold(cal(Q))$,
    // Candidate depth maps.
    depth_q: $bold(D)_q$,
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
  ),
  vin: (
    // Loss symbol.
    loss: $cal(L)$,
    // Pose embedding.
    pose_emb: $bold(e)$,
    // Token / feature vector.
    token: $bold(x)$,
    // Positional encoding.
    pos: $bold(p)$,
    // Global context vector.
    global: $bold(g)$,
    // Attention query.
    query: $bold(q)$,
    // Attention key.
    key: $bold(k)$,
    // Attention value.
    value: $bold(v)$,
    // SE(3) transform.
    T: $bold(T)$,
    // Weight matrix.
    W: $bold(W)$,
    // FiLM scale.
    gamma: $bold(gamma)$,
    // FiLM shift.
    beta: $bold(beta)$,
  ),
  shape: (
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
    // Voxel grid size.
    Vvox: $V$,
    // Global pooling dim.
    Gpool: $G_"pool"$,
    // Global projection dim.
    Gproj: $G_"proj"$,
    // Mesh vertex count.
    M: $M$,
    // Ordinal bins.
    K: $K$,
    // Per-point semidense feature dimension (e.g., XYZ + extras).
    Csem: $C_"sem"$,
    // Feature channel / embedding dimensions.
    Fin: $F_"in"$,
    Fpose: $F_"pose"$,
    Fpe: $F_"pe"$,
    Fq: $F_q$,
    Fg: $F_g$,
    Ftau: $F_tau$,
    Fproj: $F_"proj"$,
    Ftok: $F_"tok"$,
    Ffr: $F_"fr"$,
    Fpt: $F_"pt"$,
    Faux: $F_"aux"$,
    Fhead: $F_"head"$,
    Fhid: $F_"hid"$,
  ),
)

/// SE(3) transform from frame `B` to frame `A` (i.e., `A <- B`).
///
/// Note: Using a function avoids needing whitespace when applying scripts to
/// interpolated symbols (e.g., `$#(symb.vin.T)_A$` would require a space).
#let T(A, B) = $#symb.vin.T^(#A)_(#B)$

// ============================================================================
// Common Math Expressions (nested dictionary)
// ============================================================================

// Nested dictionary of all common equations (shared across paper + slides)
#let eqs = (
  rri: (
    cd: $
      "CD"(#symb.oracle.points, #symb.ase.mesh) =
      #symb.oracle.acc (#symb.oracle.points, #symb.ase.mesh) + #symb.oracle.comp (#symb.oracle.points, #symb.ase.mesh)
    $,
    acc: $
      #symb.oracle.acc (#symb.oracle.points, #symb.ase.mesh) =
      (1)/(||#symb.oracle.points||) sum_(bold(p) in #symb.oracle.points) min_(bold(f) in #symb.ase.faces) d(bold(p), bold(f))^2
    $,
    comp: $
      #symb.oracle.comp (#symb.oracle.points, #symb.ase.mesh) =
      (1)/(||#symb.ase.faces||) sum_(bold(f) in #symb.ase.faces) min_(bold(p) in #symb.oracle.points) d(bold(p), bold(f))^2
    $,
    union: $
      #(symb.oracle.points) _(t union q) = #(symb.oracle.points) _t union #symb.oracle.points_q
    $,
    rri: $
      "RRI"(q) =
      ("CD"(#(symb.oracle.points) _t, #symb.ase.mesh) - "CD"(#(symb.oracle.points) _t union #symb.oracle.points_q, #symb.ase.mesh))
      / ("CD"(#(symb.oracle.points) _t, #symb.ase.mesh) + epsilon)
    $,
    greedy: $ q_star = op("argmax", limits: #true)_(q in #symb.oracle.candidates) "RRI"(q) $,
  ),
  coverage: (
    ratio: $ "CR"_t = (tilde(N)_t) / (N^*) dot 100% $,
    weight: $ w_i(t) = (1 - lambda_t) + lambda_t (f + (1 - f) c_i^p) $,
    weighted_loss: $ cal(L) = (sum_i w_i(t) ell_i) / (sum_i w_i(t)) $,
  ),
  binning: (
    // Empirical quantile edges (equal-mass bins) for discretizing oracle RRI.
    edges: $
      e_k = "Quantile"( {r_n}, k/K),
      quad k in {1, dots, K-1}
    $,
    // Ordinal class index via edge counting (equivalent to `torch.bucketize`).
    label: $
      y(r) = sum_(k=1)^(K-1) bb(1)[r > e_k],
      quad y(r) in {0, dots, K-1}
    $,
    // CORAL level targets derived from ordinal labels.
    levels: $
      t_k = bb(1)[y > k],
      quad k in {0, dots, K-2}
    $,
  ),
  coral: (
    loss: $
      cal(L)_"coral"(y, bold(p))
      = - sum_(k=0)^(K-2) (t_k "log"(p_k) + (1 - t_k) "log"(1 - p_k))
    $,
    marginals: $ pi_k = p_(k-1) - p_k, quad p_(-1) = 1, quad p_(K-1) = 0 $,
    expected: $ hat(r) = sum_(k=0)^(K-1) pi_k dot u_k $,
    // Fraction of rank-order violations in the cumulative probabilities.
    violation: $
      v =
      (1) / (K - 2)
      sum_(k=0)^(K-3) bb(1)[p_(k+1) > p_k]
    $,
    rel_random: $ cal(L)_("rel") = cal(L)_("coral") / ((K - 1) "log"(2)) $,
  ),
  vin: (
    // Optional auxiliary regression combined with the CORAL loss.
    loss_total: $ #symb.vin.loss = #(symb.vin.loss)_"coral" + lambda dot #(symb.vin.loss)_"reg" $,
  ),
  metrics: (
    spearman: $ rho = "corr"("rank"(hat(r)_i), "rank"(r_i)) $,
    topk_acc: $ "TopKAcc"(k) = (1) / N sum_i bb(1)[y_i in "TopK"(bold(pi)_i, k)] $,
    confusion: $ C_(a,b) = |{i : y_i = a, hat(y)_i = b}| $,
    candidate_validity: $ m_i = bb(1)["finite"] dot bb(1)[v_i > 0] dot bb(1)[v_i^("sem") > 0] $,
    grad_norm: $ ||nabla_theta cal(L)||_2 $,
  ),
  features: (
    film: $ bold(g)_i^("film") = (1 + bold(gamma)_i) dot.o bold(g)_i + bold(beta)_i $,
    semidense_validity: $
      m_(i,j)
      =
      bb(1)["finite"] dot bb(1)[z_(i,j) > 0] dot
      bb(1)[0 <= u_(i,j) < W_i] dot bb(1)[0 <= v_(i,j) < H_i]
    $,
    semidense_visibility: $
      v_i^("sem")
      = (sum_j w_(i,j) m_(i,j)) / (sum_j w_(i,j) f_(i,j))
    $,
  ),
  action: (
    space: $ cal(A) = bb(R)^3 times S O(2) $,
  ),
)

// ============================================================================
// Utility Functions
// ============================================================================

/// Create a highlighted inline term
#let term(body) = text(weight: "semibold", body)

/// Create a filename/path reference
#let filepath(body) = raw(body, lang: none)

/// Create a citation-style reference
#let paperref(title, authors) = [
  #emph[#title] by #authors
]
