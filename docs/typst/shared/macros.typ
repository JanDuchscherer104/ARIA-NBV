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

#let excl = "#text(size: 22pt)[_!_]"

// ============================================================================
// Paper notation (symbols used throughout the Typst paper)
// ============================================================================
// Note: These are *labels* that can be injected into math via `#` interpolation
// inside `$ ... $`. We keep them centralized here to enforce consistent
// notation across sections.

/// Frame labels used in transform subscripts (T_{A<-B}).
// Use the canonical `#symb.frame.*` entries below.

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
  ),
  obs: (
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
  ),
  entity: (
    // Entity set (objects of interest).
    E: $cal(E)$,
    // Entity-weight vector; use components as `#(symb.entity.w)_e`.
    w: $bold(w)$,
    // Mixing weight for the scene-level term.
    lambda_scene: $lambda_"scene"$,
    // Weighted objective (global + entity-specific terms).
    rri_total: $RRI_"total"$,
  ),
  rl: (
    // RL state / observation / reward / return.
    s: $s$,
    o: $o$,
    a: $a$,
    r: $r$,
    G: $G$,
    Q: $Q$,
    V: $V$,
    pi: $pi$,
    A: $A$,
    delta: $delta$,
    rho: $rho$,
    z: $z$,
    // Pose / persistent memory / optional entity memory / budget.
    x: $bold(x)$,
    m: $bold(m)$,
    e: $bold(e)$,
    b: $b$,
    // History bundles for the two state formulations.
    hist_ego: $cal(O)^"ego"_(1:t)$,
    hist_cf: $(cal(O)^"ego", cal(O)^"cf")_(1:t)$,
  ),
  vin: (
    // Loss symbol.
    loss: $cal(L)$,
    // Pose embedding.
    pose_emb: $bold(E)_q$,
    // Token / feature vector.
    token: $bold(x)$,
    // Pooled voxel token feature (used in pose-conditioned global attention).
    vox_tok: $bold(t)$,
    // Positional encoding.
    pos: $bold(p)$,
    // Global context vector.
    global: $bold(g)$,
    // Attention query.
    query: $bold(Q)_q$,
    // Attention key.
    key: $bold(K)$,
    // Attention value.
    value: $bold(V)$,
    // SE(3) transform.
    T: $bold(T)$,
    // Weight matrix.
    W: $bold(W)$,
    // FiLM scale.
    gamma: $bold(gamma)$,
    // FiLM shift.
    beta: $bold(beta)$,
    // Semi-dense observation count (track length).
    n_obs: $n_"obs"$,
    // Max observation count used for normalization.
    n_obs_max: $n_"obs"^"max"$,
    // Inverse-distance std (sigma_rho) from semi-dense tracks.
    inv_dist_std: $sigma_(rho)$,
    // Min / p95 of inverse-distance std for normalization.
    inv_dist_std_min: $sigma_(rho,"min")$,
    inv_dist_std_p95: $sigma_(rho,"p95")$,
    // Distance std (sigma_d) for completeness.
    dist_std: $sigma_d$,
    // Continuous oracle RRI (per candidate).
    rri: $r$,
    // Predicted RRI proxy (expected CORAL value).
    rri_hat: $hat(r)$,
    // Coverage / visibility fraction (generic).
    cov_frac: $c$,
    // Coverage-based weight.
    cov_weight: $w$,
    // Coverage-weight blend strength.
    cov_strength: $lambda$,
    // Auxiliary regression weight.
    aux_weight: $lambda_"reg"$,
    // Voxel coverage proxy (candidate-level).
    voxel_valid: $v$,
    // Semi-dense visibility proxy (candidate-level).
    sem_valid: $v^("sem")$,
    // Semi-dense projection statistics (per candidate).
    sem_proj: $bold(s)_"proj"$,
    // Semi-dense grid CNN features (per candidate).
    sem_grid: $bold(s)_"grid"$,
    // Trajectory pooled feature (optional).
    traj_feat: $bold(f)_"traj"$,
    // Trajectory context per candidate (optional attention).
    traj_ctx: $bold(c)_"traj"$,
    // Candidate validity mask.
    cand_valid: $m$,
    // -----------------------------------------------------------------------
    // EVL voxel-field features (as used by VINv3)
    //
    // Notes:
    // - The voxel grid lives in the voxel frame #symb.frame.v and is anchored
    //   at the final rig pose #symb.ase.traj_final (gravity-aligned).
    // - We keep notation head-centric: occupancy / evidence / counts, matching
    //   `aria_nbv/aria_nbv/vin/model_v3.py`.
    // -----------------------------------------------------------------------
    // Occupancy prediction (sigmoid) on the voxel grid.
    occ_pr: $bold(V)_"occ"^"pr"$,
    // Occupied evidence (voxelized surface evidence from inputs).
    occ_in: $bold(V)_"surf"^"in"$,
    // Free-space evidence (optional; if provided by EVL).
    free_in: $bold(V)_"free"^"in"$,
    // Observation counts / coverage proxy per voxel.
    counts: $bold(V)_"count"^"in"$,
    // Normalized counts (log1p normalized).
    counts_norm: $bold(V)_"count"^"norm"$,
    // Centerness prediction (geometric prior).
    cent_pr: $bold(V)_"cent"^"pr"$,
    // Centerness after NMS (used in VINv3 field bundle).
    cent_pr_nms: $bold(V)_"cent"^"pr,nms"$,
    // Observed mask (counts > 0).
    observed: $bold(V)_"obs"$,
    // Unknown mask (1 - counts_norm).
    unknown: $bold(V)_"unk"$,
    // New-surface prior (unknown ⊙ occ_pr).
    new_surface_prior: $bold(V)_"new"$,
    // VIN scene field after lightweight 3D projection/assembly (multi-channel).
    field_v: $bold(F)_v$,
    // Per-candidate voxel features sampled/pooled from the scene field.
    field_q: $bold(F)_q^("vox")$,
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
    strength_linear: $
      lambda_t = lambda_0 + (lambda_T - lambda_0) dot (t / T)
    $,
    strength_cosine: $
      lambda_t = lambda_T + (lambda_0 - lambda_T) dot (1 + "cos"(pi t / T)) / 2
    $,
  ),
  binning: (
    // Empirical quantile edges (equal-mass bins) for discretizing oracle RRI.
    edges: $
      e_k = "Quantile"( {r_i}_(i=1)^N, k/K),
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
      cal(L)_"coral" (y, bold(p))
      = - sum_(k=0)^(K-2) (t_k "log"(p_k) + (1 - t_k) "log"(1 - p_k))
    $,
    balanced_bce: $
      cal(L)_"bal"
      = -(1)/(K-1) sum_(k=0)^(K-2)
      (w_k t_k "log"(p_k) + (1 - t_k) "log"(1 - p_k))
    $,
    balanced_bce_weight: $ w_k = (1 - pi_k^("th")) / pi_k^("th") $,
    focal: $
      cal(L)_"focal"
      = -(1)/(K-1) sum_(k=0)^(K-2)
      alpha_(t,k) (1 - p_(t,k))^gamma "log"(p_(t,k))
    $,
    focal_defs: $
      p_(t,k) = p_k t_k + (1 - p_k) (1 - t_k),
      quad alpha_(t,k) = alpha t_k + (1 - alpha) (1 - t_k)
    $,
    marginals: $ pi_k = p_(k-1) - p_k, quad p_(-1) = 1, quad p_(K-1) = 0 $,
    expected: $ hat(r) = sum_(k=0)^(K-1) pi_k dot u_k $,
    bin_values: $
      u_0 in bb(R),
      quad u_k = u_0 + sum_(j=1)^k op("softplus")(delta_j)
    $,
    // Fraction of rank-order violations in the cumulative probabilities.
    violation: $
      v =
      (1) / (K - 2)
      sum_(k=0)^(K-3) bb(1)[p_(k+1) > p_k]
    $,
    rel_random: $ cal(L)_("rel") = cal(L)_("coral") / ((K - 1) "log"(2)) $,
  ),
  vin: (
    // Observation-count normalization used as a voxel-coverage proxy.
    counts_norm: $
      #symb.vin.counts_norm
      = ("log"(1 + #symb.vin.counts)) / ("log"(1 + "max"(#symb.vin.counts)))
    $,
    // Unknown mask + new-surface prior derived from counts and occupancy.
    new_surface_prior: $
      #symb.vin.unknown = 1 - #symb.vin.counts_norm,
      quad #symb.vin.new_surface_prior = #symb.vin.unknown dot.op #symb.vin.occ_pr
    $,
    // Optional auxiliary regression combined with the CORAL loss.
    loss_total: $ #symb.vin.loss = #(symb.vin.loss) _"coral" + lambda dot #(symb.vin.loss) _"reg" $,
    aux_reg_mse: $
      #(symb.vin.loss) _"reg"
      = (1)/(N) sum_i (#(symb.vin.rri_hat) _i - #(symb.vin.rri) _i)^2
    $,
    aux_reg_huber: $
      #(symb.vin.loss) _"reg"
      = (1)/(N) sum_i "Huber"_1(#(symb.vin.rri_hat) _i - #(symb.vin.rri) _i)
    $,
    huber: $
      "Huber"_1(e) = { 0.5 e^2 "if" |e| <= 1; |e| - 0.5 "otherwise" }
    $,
    aux_weight: $
      lambda_"reg" (t)
      = max(lambda_0 dot gamma^t, lambda_"min")
    $,
  ),
  metrics: (
    spearman: $
      rho = "corr"("rank"(#(symb.vin.rri_hat) _i), "rank"(#(symb.vin.rri) _i))
    $,
    topk_acc: $ "TopKAcc"(k) = (1) / N sum_i bb(1)[y_i in "TopK"(bold(pi)_i, k)] $,
    confusion: $ C_(a,b) = |{i : y_i = a, hat(y)_i = b}| $,
    label_hist: $ h_k = |{i : y_i = k}| $,
    candidate_validity: $
      #(symb.vin.cand_valid) _i
      =
      bb(1)["finite"]
      dot bb(1)[#(symb.vin.voxel_valid) _i > 0]
      dot bb(1)[#(symb.vin.sem_valid) _i > 0]
    $,
    rri_mean: $ bar(#symb.vin.rri) = (1)/(N) sum_i #(symb.vin.rri) _i $,
    pred_rri_mean: $ bar(#symb.vin.rri_hat) = (1)/(N) sum_i #(symb.vin.rri_hat) _i $,
    bias2: $ "bias"^2 = ((1)/(N) sum_i (#(symb.vin.rri_hat) _i - #(symb.vin.rri) _i))^2 $,
    variance: $
      "var"
      =
      (1)/(N) sum_i (#(symb.vin.rri_hat) _i - #(symb.vin.rri) _i)^2
      - ((1)/(N) sum_i (#(symb.vin.rri_hat) _i - #(symb.vin.rri) _i))^2
    $,
    mean: $ bar(x) = (1)/(N) sum_i x_i $,
    std: $ sigma_x = sqrt((1)/(N) sum_i (x_i - bar(x))^2) $,
    voxel_valid_mean: $ bar(#symb.vin.voxel_valid) = (1)/(N) sum_i #(symb.vin.voxel_valid) _i $,
    voxel_valid_std: $
      sigma_(#symb.vin.voxel_valid) = sqrt((1)/(N) sum_i (#(symb.vin.voxel_valid) _i - bar(#symb.vin.voxel_valid))^2)
    $,
    sem_valid_mean: $ bar(#symb.vin.sem_valid) = (1)/(N) sum_i #(symb.vin.sem_valid) _i $,
    sem_valid_std: $
      sigma_(#symb.vin.sem_valid) = sqrt((1)/(N) sum_i (#(symb.vin.sem_valid) _i - bar(#symb.vin.sem_valid))^2)
    $,
    candidate_valid_frac: $ (1)/(N) sum_i #(symb.vin.cand_valid) _i $,
    cov_weight_mean: $ bar(#symb.vin.cov_weight) = (1)/(N) sum_i #(symb.vin.cov_weight) _i $,
    drop_nonfinite_logits_frac: $
      (sum_i bb(1)["finite"(#(symb.vin.rri) _i)] dot bb(1)["nonfinite"(bold(ell)_i)])
      / (sum_i bb(1)["finite"(#(symb.vin.rri) _i)])
    $,
    skip_nonfinite_logits: $
      bb(1)[sum_i bb(1)["finite"(#(symb.vin.rri) _i)] > 0 dot sum_i #(symb.vin.cand_valid) _i = 0]
    $,
    skip_no_valid: $ bb(1)[sum_i bb(1)["finite"(#(symb.vin.rri) _i)] = 0] $,
    grad_norm: $ ||nabla_theta cal(L)||_2 $,
  ),
  features: (
    film: $
      #(symb.vin.global) _i^("film")
      = (1 + #(symb.vin.gamma) _i) dot.op #(symb.vin.global) _i + #(symb.vin.beta) _i
    $,
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
  rl: (
    mdp: $
      cal(M) = (cal(S), cal(A), P, #symb.rl.r, gamma)
    $,
    hist_ego: $
      #symb.rl.hist_ego
      =
      (
        #symb.obs.img_rgb,
        #symb.obs.pose,
        #symb.obs.points_semi,
        #(symb.vin.field_v)^"ego"
      ) _(1:t)
    $,
    hist_cf: $
      #symb.rl.hist_cf
      =
      (
        #symb.rl.hist_ego,
        (
          #(symb.obs.depth)^"cf",
          #(symb.obs.vis)^"cf",
          #symb.obs.points_cf
        ) _(1:t)
      )
    $,
    state_ego: $
      #(symb.rl.s) _t^"ego"
      =
      (
        #symb.rl.hist_ego,
        #(symb.rl.x) _t,
        #(symb.ase.points_semi) _t,
        #(symb.vin.field_v) _t,
        #(symb.rl.e) _t,
        #(symb.rl.b) _t
      )
    $,
    state_cf: $
      #(symb.rl.s) _t^"cf"
      =
      (
        #symb.rl.hist_cf,
        #(symb.rl.x) _t,
        #(symb.rl.m) _t,
        #(symb.rl.e) _t,
        #(symb.rl.b) _t
      )
    $,
    obs_render: $
      #(symb.rl.o) _(t+1)
      =
      cal(G)(#symb.ase.mesh, #(symb.rl.x) _(t+1))
    $,
    memory_update: $
      #(symb.rl.m) _(t+1)
      =
      cal(U)(
        #(symb.rl.m) _t,
        #(symb.rl.o) _(t+1),
        #(symb.rl.x) _(t+1)
      )
    $,
    reward_log: $
      #(symb.rl.r) _t
      =
      "log"("CD"(#(symb.oracle.points) _t, #symb.ase.mesh) + epsilon)
      -
      "log"("CD"(#(symb.oracle.points) _(t+1), #symb.ase.mesh) + epsilon)
    $,
    reward_geom: $
      #(symb.rl.r) _t^"geom"
      =
      "log"("CD"(#(symb.oracle.points) _t, #symb.ase.mesh) + epsilon)
      -
      "log"("CD"(#(symb.oracle.points) _(t+1), #symb.ase.mesh) + epsilon)
      -
      alpha bb(1)["collision"(#(symb.rl.a) _t)]
      -
      beta c(#(symb.rl.a) _t)
    $,
    planner: $
      #(symb.rl.a) _t^star
      =
      "arg max"_(#(symb.rl.a) _(t:t+H-1))
      sum_(k=0)^(H-1) gamma^k #(symb.rl.r) _(t+k)
    $,
    q_backup: $
      y_t^Q
      =
      #(symb.rl.r) _t
      +
      gamma #(symb.rl.V) ( #(symb.rl.s) _(t+1) )
    $,
    iql_q_loss: $
      cal(L)_(#(symb.rl.Q))^"IQL"
      =
      ( #(symb.rl.Q) ( #(symb.rl.s) _t, #(symb.rl.a) _t ) - y_t^Q )^2
    $,
    cql_loss: $
      cal(L)_(#(symb.rl.Q))^"CQL"
      =
      (1)/(2) ( #(symb.rl.Q) ( #(symb.rl.s) _t, #(symb.rl.a) _t ) - y_t^Q )^2
      +
      alpha (
        "logsumexp"_(a in cal(A)) #(symb.rl.Q) ( #(symb.rl.s) _t, a )
        -
        #(symb.rl.Q) ( #(symb.rl.s) _t, #(symb.rl.a) _t )
      )
    $,
    return_lambda: $
      #(symb.rl.G) _t^lambda
      =
      (1-lambda) sum_n lambda^(n-1) G_t^(n)
    $,
    leq_loss: $
      cal(L)_(#(symb.rl.V))^"LEQ"
      =
      rho_(tau)(
        #(symb.rl.V) ( #(symb.rl.s) _t ) - #(symb.rl.G) _t^lambda
      )
    $,
    gae: $
      #(symb.rl.A) _t^"GAE"
      =
      sum_(l=0)^(L-1) (gamma lambda)^l #(symb.rl.delta) _(t+l)
    $,
    ppo_clip: $
      cal(L)_(#(symb.rl.pi))^"PPO"
      =
      bb(E)[
        "min"(
          #(symb.rl.rho) _t #(symb.rl.A) _t,
          "clip"(#(symb.rl.rho) _t, 1-epsilon, 1+epsilon) #(symb.rl.A) _t
        )
      ]
    $,
    hier_policy: $
      #(symb.rl.z) _t ~ #(symb.rl.pi) _("hi")(z ; #(symb.rl.s) _t),
      quad
      #(symb.rl.a) _t ~ #(symb.rl.pi) _("lo")(a ; #(symb.rl.s) _t, #(symb.rl.z) _t)
    $,
  ),
  action: (
    space: $ cal(A) = bb(R)^3 times S O(2) $,
  ),
  entity: (
    objective: $
      RRI_"total"(q)
      =
      sum_(e in #symb.entity.E)
      #(symb.entity.w) _e dot #(symb.oracle.rri) _e
      +
      #symb.entity.lambda_scene dot #symb.oracle.rri
    $,
  ),
)

// ============================================================================
// Utility Functions
// ============================================================================

/// Create a highlighted inline term
#let term(body) = text(weight: "semibold", body)

/// Create a filename/path reference
#let filepath(body) = raw(body, lang: none)

/// Link to a file in the GitHub repo (shows only the filename).
#let gh(path) = {
  let base = path.split("/").last()
  link("https://github.com/JanDuchscherer104/NBV/blob/main/" + path)[#code-inline(base)]
}

/// Create a citation-style reference
#let paperref(title, authors) = [
  #emph[#title] by #authors
]
