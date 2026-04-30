= Aria-VIN-NBV Architecture

#import "../../shared/macros.typ": *

This section describes the currently implemented VINv3 candidate scorer that
we train against oracle RRI labels. The design emphasizes explicit
_view-conditioned_ evidence and a lightweight CORAL head, with EVL providing the
voxel backbone @EFM3D-straub2024.

== Core design and optional ablations

To avoid ambiguity between what is implemented in VINv3 and what is exploratory, we
separate the model into (i) a stable core that is always present and (ii)
optional modules evaluated via ablations.

- *Core*: EVL voxel evidence, pose encoding, pose-conditioned global context,
  per-candidate semi-dense projection statistics (optionally a projection-grid
  CNN), and a final MLP + CORAL head.
- *View-conditioned evidence*: candidate-specific cues obtained by projecting
  both semi-dense points and pooled voxel centers into each candidate view.
- *Optional ablations*: trajectory-context encoder and alternative feature
  fusion mechanisms are not part of this section, even though they might have been part of our "baseline" model.

We denote the core _per-candidate_ features as the pose embedding
$#(symb.vin.pose_emb)$, global context $#(symb.vin.global) _q$, and semidense
projection evidence $#(symb.vin.sem_proj) _q$ augmented by
$#(symb.vin.sem_grid) _q$, fused by an MLP + CORAL head (Section *Feature fusion
and CORAL head*).

#figure(
  placement: auto,
  image("/figures/app-paper/vin-geom-oc_pr-candfrusta-semi-dense.png", width: 100%),
  caption: [Superposition of VINv3 inputs: EVL occupancy prior, candidate frusta, and semi-dense points.],
) <fig:vin-inputs>

== Backbone and scene field

EVL lifts multi-view observations into a local voxel grid centered near the
latest rig pose and outputs multiple dense heads (occupancy probability,
centerness, free-space evidence, and observation counts) @EFM3D-straub2024. We
assemble an input voxel field $#(symb.vin.field_v)^("in") in bb(R)^(#symb.shape.B times #symb.shape.Fin times #symb.shape.Vvox times #symb.shape.Vvox times #symb.shape.Vvox)$
by concatenating selected EVL head channels plus derived cues. In the current
VINv3 baseline, the default channel set is
$(#symb.vin.occ_pr, #symb.vin.occ_in, #symb.vin.counts_norm, #symb.vin.cent_pr)$,
with optional inclusion of #symb.vin.free_in, #symb.vin.unknown, and
#symb.vin.new_surface_prior via configuration. In our EVL configuration the
voxel grid uses $#symb.shape.Vvox=48$ (roughly a 4 m cube), and the observation
evidence channels #symb.vin.occ_in and #symb.vin.counts are derived from the
semi-dense SLAM points.

We normalize observation counts via a per-snippet log1p normalization:

#block[#align(center)[#eqs.vin.counts_norm]]

where the maximum is taken over the voxel grid (per batch element). We then
derive the unknown mask and a new-surface prior as

#block[#align(center)[#eqs.vin.new_surface_prior]]

When EVL does not provide free-space evidence, we derive it as
$ #symb.vin.free_in = #symb.vin.observed dot.o (1 - #symb.vin.occ_in) $
(observed mask times empty occupancy). The final scene field is then
$#symb.vin.field_v = phi("Conv"_(1 times 1 times 1)(#(symb.vin.field_v)^("in"))) in bb(R)^(#symb.shape.B times #symb.shape.Ffield times #symb.shape.Vvox times #symb.shape.Vvox times #symb.shape.Vvox)$,
implemented as `Conv3d + GroupNorm + GELU` @GroupNorm-wu2018.
The EVL output summary used to form the scene field is shown in @fig:evl-summary,
and representative input-channel slices are provided in @fig:evl-field-slices.


#figure(
  placement: auto,
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 10pt,
    image("/figures/app-paper/field_occ_in.png", width: 100%),
    image("/figures/app-paper/field_occ_pr.png", width: 100%),
    image("/figures/app-paper/field_counts_norm.png", width: 100%),
  ),
  caption: [Scene-field input channels: occ_input, occ_pr, and normalized observation counts.],
) <fig:evl-field-slices>

// NOTE: #symb.vin.counts_norm is log-scaled and normalized per snippet, so its
// appearance is dominated by the local trajectory coverage pattern.

== Pose encoding

Each candidate pose is expressed in the reference rig frame via
$#T(symb.frame.r, symb.frame.cq) = #T(symb.frame.w, symb.frame.r)^(-1) dot #T(symb.frame.w, symb.frame.cq)$.
We encode translation and rotation using a 6D rotation representation
@zhou2019continuity and Learnable Fourier Features (LFF) @LFF-li2021.
For each candidate $i$ the pose encoder yields:

- a low-dimensional pose vector in $bb(R)^9$ (translation + rotation-6D),
- an embedding $#(symb.vin.pose_emb) _q in bb(R)^(#symb.shape.Fpose)$,
- the candidate center $#(symb.oracle.center) _q in bb(R)^3$ in the rig frame.
// TODO(paper-cleanup): Clarify which quantities are tensors vs. scalars and use `bold(...)`
// accordingly (e.g., pose vector vs. embedding). Prefer referencing #symb definitions.

Figure @fig:vin-pose-encoder summarizes the pose-encoding branch and its
output contract.

#figure(
  placement: auto,
  image("/figures/diagrams/vin_nbv/mermaid/pose_encoder.png", width: 50%),
  caption: [Pose encoding branch: rig-relative candidate pose #T(symb.frame.r, symb.frame.cq) mapped to an embedding $(#symb.vin.pose_emb)_q$.],
) <fig:vin-pose-encoder>

== Global context via pose-conditioned attention

We pool #symb.vin.field_v onto a coarse grid of size
$#symb.shape.Gpool times #symb.shape.Gpool times #symb.shape.Gpool$, yielding
$#symb.shape.Gpool^3$ voxel tokens #symb.vin.vox_tok. For each voxel $j$, the voxel-center position
$#symb.vin.pos _j$ is used both to compute positional keys and to parameterize the
voxel-projection branch. Each token is augmented with an LFF positional encoding
of its voxel center expressed in the rig frame.

Candidate pose embeddings act as queries in a multi-head cross-attention block
@Transformer-vaswani2017. Let $#(symb.vin.vox_tok) _j$ denote the pooled voxel token
feature at voxel center $#(symb.vin.pos) _j$ (both defined in the reference rig
frame). Cross-attention then produces candidate-conditioned mixtures of voxel
tokens:

#block[
  #align(center)[
    $
      #(symb.vin.query) = #(symb.vin.W) _Q dot #(symb.vin.pose_emb), \
      #(symb.vin.key) _j = #(symb.vin.W) _K dot #(symb.vin.vox_tok) _j + phi(#(symb.vin.pos) _j), \
      #(symb.vin.value) _j = #(symb.vin.W) _V dot #(symb.vin.vox_tok) _j
    $
  ]
]

#block[
  #align(center)[
    $
      #symb.vin.global _q = "MHCA"(#symb.vin.query, #(symb.vin.key), #(symb.vin.value))
    $
  ]
]
The attention output yields a global context vector $#symb.vin.global _q in bb(R)^(#symb.shape.Fg)$
for each candidate. We apply residual connections and an MLP block for
stability.

Figure @fig:vin-global-pool illustrates how the pose-conditioned pooling and
the voxel-projection FiLM modulation combine into the final global feature
$#(symb.vin.global) _q$.

#figure(
  placement: auto,
  image("/figures/diagrams/vin_nbv/mermaid/global_pool.png", width: 100%),
  caption: [Pose-conditioned global pooling with voxel-projection FiLM: pooled voxel tokens attend to pose queries, then $(#symb.vin.global)_q$ is modulated by candidate-specific voxel projection statistics.],
) <fig:vin-global-pool>

== Coverage proxies and candidate validity

Candidate shells can place views outside EVL's local voxel extent. We therefore
compute a per-candidate voxel-evidence fraction $(#symb.vin.voxel_valid)_q$ by
sampling the normalized observation-count field #symb.vin.counts_norm at the
candidate camera center and applying in-bounds and finiteness checks. Together
with the semi-dense visibility fraction $(#symb.vin.sem_valid)_q$, we define the
conservative candidate-validity mask #symb.vin.cand_valid as:

#block[#align(center)[#eqs.metrics.candidate_validity]]

== Voxel projection FiLM

To modulate global features with candidate-dependent evidence, we pool voxel
centers to the same coarse grid used in global attention ($(#symb.shape.Gpool)^3$ points)
via adaptive average pooling over the voxel-center grid (with symmetric
center-cropping if the grid is padded), project those centers into each
candidate view, and summarize their screen-space coverage and depth statistics.
Because voxel centers have no per-point reliability metadata, the projection
statistics use uniform weights. A linear FiLM head predicts
per-channel $(#(symb.vin.gamma) _q, #(symb.vin.beta) _q)$ from these projection statistics and
applies

#block[#align(center)[#eqs.features.film]]

This provides lightweight view-conditioned modulation without introducing
additional token-level attention @perez2017filmvisualreasoninggeneral.

== Semi-dense projection statistics

To inject view-dependent evidence beyond the EVL voxel extent, we project the
semi-dense SLAM points into each candidate view using the same screen-space
camera model as the depth renderer. A projected point is valid if it is finite,
in front of the camera ($z > 0$), and within the image bounds. For each
candidate we compute:

- coverage ratio and empty fraction using a coarse $#symb.shape.Gsem times #symb.shape.Gsem$ image grid
  (fraction of bins hit by at least one valid point),
- visibility fraction $(#symb.vin.sem_valid)_q$ as the (optionally reliability-weighted)
  fraction of valid projections among finite projections,
- depth mean and depth standard deviation over valid points.

Reliability weights combine normalized observation count #symb.vin.n_obs and
inverse-distance uncertainty #symb.vin.inv_dist_std when available and clamp
the normalized values to $[0,1]$. The resulting projection statistics
#symb.vin.sem_proj are concatenated to the head input.
We subsample the padded semi-dense point cloud using the per-snippet
`lengths` field to avoid including invalid padding in the projection.

Figure @fig:vin-semidense-proj summarizes the semidense projection-statistics
branch, while @fig:app-paper-semidense-proj shows example diagnostic maps for
three of the per-bin statistics.

Concretely, we treat $#(symb.vin.sem_proj) _q$ as a compact scalar feature
vector of the form
$ [nu_"cov", nu_"empty", #symb.vin.sem_valid, mu_s, sigma_rho]_q $
capturing coverage/emptiness, visibility, and depth moments.

#figure(
  placement: auto,
  image("/figures/diagrams/vin_nbv/mermaid/semidense_proj.png", width: 50%),
  caption: [Semidense projection statistics branch: project semidense points into each candidate camera and summarize coverage/visibility/depth moments into $(#symb.vin.sem_proj)_q$.],
) <fig:vin-semidense-proj>

We use the per-point validity mask

#text(size: 8pt)[#eqs.features.semidense_validity]

and define the (optionally reliability-weighted) visibility fraction
$(#symb.vin.sem_valid)_q = v_q^("sem")$ as the valid fraction among finite
projections:

#block[#align(center)[#eqs.features.semidense_visibility]]

Depth mean and depth standard deviation are computed over the valid set.

// Qualitative diagnostics for the semidense projection branch (example maps).
#figure(
  placement: auto,
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 10pt,
    image("/figures/app-paper/semi-dense-counts-proj.png", width: 100%),
    image("/figures/app-paper/semi-dense-weight-proj.png", width: 100%),
    image("/figures/app-paper/semi-dense-std-proj.png", width: 100%),
  ),
  caption: [Semi-dense projection statistics (example): per-bin counts, reliability weights, and depth standard deviation.],
) <fig:app-paper-semidense-proj>

== Semi-dense projection grid CNN

We also build a per-candidate $#symb.shape.Gsem times #symb.shape.Gsem$ grid from the projected semi-dense
points, storing occupancy, depth mean, and depth standard deviation per bin.
A tiny 2D CNN encodes this grid into a compact feature vector that is appended
to the head input. This adds local view-plane structure while remaining far
lighter than a full image encoder.

The resulting grid is encoded into $(#symb.vin.sem_grid)_q$. In the current
implementation, the grid CNN uses the hard validity mask but does not apply
per-point reliability weights within bins.

In particular, we form a per-candidate grid tensor
$bold(H)_q in bb(R)^(3 times #symb.shape.Gsem times #symb.shape.Gsem)$ with
channels $[O, mu_z, sigma_z]_q$ (occupancy and depth moments), and map
it to $#symb.vin.sem_grid _q$ via a tiny CNN.

Figure @fig:vin-semidense-frustum summarizes the grid-CNN encoder.

// Diagram of the per-candidate projection-grid CNN used in VINv3.
#figure(
  placement: auto,
  image("/figures/diagrams/vin_nbv/mermaid/semidense_frustum.png", width: 35%),
  caption: [Semi-dense projection grid CNN branch (tiny CNN on a coarse $#symb.shape.Gsem times #symb.shape.Gsem$ grid).],
) <fig:vin-semidense-frustum>

// == Optional trajectory context

// An optional trajectory encoder embeds the snippet's rig pose history (expressed
// in the reference rig frame). Per-frame pose encodings are pooled and can also
// be attended by candidate pose queries to yield a candidate-specific context
// vector, providing a compact notion of motion history and improving
// disambiguation between geometrically similar candidates. This module is
// disabled by default and is evaluated as an ablation.

// Figure @fig:vin-traj-context summarizes the optional trajectory context
// branch.

// #figure(
//   placement: auto,
//   image("/figures/diagrams/vin_nbv/mermaid/trajectory.png", width: 40%),
//   caption: [Optional trajectory context: encode rig pose history and (optionally) attend by pose queries to obtain per-candidate $(#symb.vin.traj_ctx)_q$.],
// ) <fig:vin-traj-context>

== Feature fusion and CORAL head

The final per-candidate feature vector concatenates multiple candidate-specific
and snippet-level signals, e.g.

#block[
  #align(center)[
    $
      bold(h) =
      [#(symb.vin.pose_emb) ;
        #(symb.vin.global) ;
        #(symb.vin.sem_proj) ;
        #(symb.vin.sem_grid)]
    $
  ]
]

The head MLP produces $#symb.shape.K - 1$ logits per candidate. CORAL interprets these as
cumulative probabilities and yields class probabilities and an expected ordinal
score $#symb.vin.rri_hat$ (used as a ranking proxy) @CORAL-cao2019. When we
need a continuous RRI estimate, we combine the class probabilities with
representative bin values (Section *Training Objective*).

Figure @fig:vin-head summarizes the head computation and the `VinPrediction`
outputs produced by our implementation.

#figure(
  placement: auto,
  image("/figures/diagrams/vin_nbv/mermaid/head_paper.png", width: 95%),
  caption: [Scoring head + CORAL decoding: concatenate per-candidate features and predict threshold logits for ordinal RRI labels.],
) <fig:vin-head>

// In our VINv3 code, the trajectory signal used in the concatenation is a
// candidate-specific context $(#symb.vin.traj_ctx)_q$ obtained by attending candidate
// pose queries to per-frame trajectory encodings (when enabled). The pooled
// trajectory embedding (snippet-level) is currently used for diagnostics rather
// than as a direct head input.

This architecture preserves VIN-NBV's inductive bias (view-conditioned
projection features) while leveraging EVL's egocentric voxel representation and
providing explicit candidate conditioning throughout the pipeline.

// <rm>
// Run/date-specific “baseline” prose. Replace with a single “VINv3 baseline spec” table
// (dims, enabled modules, training hyperparams, seed) and avoid internal run identifiers.
*Current VINv3 baseline (Jan 2026 run).* The trajectory encoder is disabled
(`use_traj_encoder=false`), the semidense projection CNN is enabled
(`semidense_cnn_enabled=true`), and voxel-projection FiLM is enabled by
default. The effective configuration used in the offline-cache training run is:
field dim $#symb.shape.Ffield=24$, global pool grid $#symb.shape.Gpool=5$,
semidense projection grid $#symb.shape.Gsem=12$, max semidense points
$#symb.shape.Pmax=16384$, head hidden dim $#symb.shape.Fhid=192$ with 2 MLP
layers, and 15 ordinal bins.
The VINv3 head has 74,104 trainable parameters; the EVL backbone remains
frozen and is excluded from this count.
// </rm>
// TODO(paper-cleanup): Source these “effective configuration” values from a single artifact
// (W&B config export / TOML) and ensure they match the numbers shown in slides_4.typ.
