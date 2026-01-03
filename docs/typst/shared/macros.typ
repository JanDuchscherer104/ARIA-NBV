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
#let fr_world = "world"
#let fr_rig = "rig"
#let fr_rig_ref = "rig_ref"
#let fr_cam = "cam"
#let fr_voxel = "voxel"

/// Common short-hands for sets and tensors used in equations.
///
/// - Point set: `#sym_points`
/// - Mesh surface: `#sym_mesh`
/// - Loss: `#sym_loss`
#let sym_points = $cal(P)$
#let sym_mesh = $cal(M)$
#let sym_loss = $cal(L)$
#let sym_candidates = $cal(Q)$
#let sym_depth = $bold(D)$
#let sym_dir = $bold(d)$
#let sym_center = $bold(c)$
#let sym_offset = $bold(o)$

/// Attention and feature symbols (all tensors/vectors; already bold).
#let sym_pose_emb = $bold(e)$
#let sym_token = $bold(x)$
#let sym_pos = $bold(p)$
#let sym_global = $bold(g)$
#let sym_query = $bold(q)$
#let sym_key = $bold(k)$
#let sym_value = $bold(v)$

/// Transform / weight symbols.
/// SE(3) transform symbol (matrix).
#let sym_T = $bold(T)$
/// Linear projection matrix symbol.
#let sym_W = $bold(W)$
/// FiLM scale/bias symbols (vectors).
#let sym_gamma = $bold(γ)$
#let sym_beta = $bold(β)$

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
