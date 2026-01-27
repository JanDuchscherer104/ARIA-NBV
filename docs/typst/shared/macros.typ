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
// Common Math Expressions
// ============================================================================

/// RRI formula in display math
#let rri-formula = $
  "RRI"(q) = (CD(cal(R)_"base", cal(R)_"GT") - CD(cal(R)_("base" union q), cal(R)_"GT")) / (CD(cal(R)_"base", cal(R)_"GT"))
$

/// Coverage ratio formula
#let coverage-ratio-formula = $
  "CR"_t = (tilde(N)_t) / (N^*) dot 100%
$

/// Action space definition
#let action-space = $
  cal(A) = bb(R)^3 times S O(2)
$

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
/// Access with `#s.key` inside math (e.g., `$#(s.points)_t$`).
/// Use `#(s.key)_t` when applying scripts to avoid spacing issues.
#let s = (
  points: $cal(P)$,
  mesh: $cal(M)_"GT"$,
  faces: $cal(F)_"GT"$,
  loss: $cal(L)$,
  candidates: $cal(Q)$,
  depth: $bold(D)$,
  dir: $bold(d)$,
  center: $bold(c)$,
  offset: $bold(o)$,
  acc: $cal(A)$,
  comp: $cal(C)$,

  // Attention and feature symbols (all tensors/vectors; already bold).
  pose_emb: $bold(e)$,
  token: $bold(x)$,
  pos: $bold(p)$,
  global: $bold(g)$,
  query: $bold(q)$,
  key: $bold(k)$,
  value: $bold(v)$,

  // Dimension symbols used in architecture diagrams and shape annotations.
  B: $B$,
  N: $N$,
  Tlen: $T$,
  P: $P$,
  Pproj: $P_"proj"$,
  Pfr: $P_"fr"$,
  D: $D$,
  H: $H$,
  Wdim: $W$,
  Vvox: $V$,
  Gpool: $G_"pool"$,
  Gproj: $G_"proj"$,
  M: $M$,
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

  // Transform / weight symbols.
  T: $bold(T)$,
  W: $bold(W)$,
  gamma: $bold(gamma)$,
  beta: $bold(beta)$,
)

/// SE(3) transform from frame `B` to frame `A` (i.e., `A <- B`).
///
/// Note: Using a function avoids needing whitespace when applying scripts to
/// interpolated symbols (e.g., `$#(s.T)_A$` would require a space).
#let T(A, B) = $#s.T^(#A)_(#B)$

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
