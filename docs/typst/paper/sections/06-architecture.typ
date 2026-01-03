= Aria-VIN-NBV Architecture

#import "/typst/shared/macros.typ": *

This section describes the VIN v2 architecture with a focus on theoretical
fidelity to VIN-NBV and exact alignment with the current implementation. The
model predicts ordinal RRI scores for candidate views using a frozen EVL
backbone, explicit view-conditioned evidence, and a lightweight CORAL head.

#figure(
  image("/figures/VIN-NBV_diagram.png", width: 100%),
  caption: [VIN-NBV architecture overview (baseline reference).],
) <fig:vin-nbv-diagram>

== Backbone and scene field

EVL lifts multi-view observations into a local voxel grid centered near the
latest rig pose and outputs multiple dense heads (occupancy probability,
centerness, free-space evidence, and observation counts) @EFM3D-straub2024. We
construct an input scene field $bold(F)_("in") in bb(R)^(B times C_"in" times D times H times W)$
by concatenating selected EVL head channels (implementation names:
`occ_pr`, `occ_input`, `counts`, `cent_pr`, `free_input`) plus derived features.

Let $bold(n)$ denote the per-voxel observation counts and $bold(o)^("pr")$ the
occupancy prior. We define

#block[
  #align(center)[
    $ bold(n)^("norm") = ("log"(1 + bold(n))) / ("log"(1 + "max"(bold(n)))) $
  ]
]

where `"max"` takes the maximum over the voxel grid (per batch element), and
derive

#block[
  #align(center)[
    $ bold(u) = 1 - bold(n)^("norm"), quad bold(s)^("new") = bold(u) ⊙ bold(o)^("pr") $
  ]
]

Here $bold(u)$ is a soft unknown mask and $bold(s)^("new")$ is a new-surface
prior. The projected scene field is then
$bold(F) = phi("Conv"_(1 times 1 times 1)(bold(F)_("in"))) in bb(R)^(B times C times D times H times W)$,
implemented as `Conv3d + GroupNorm + GELU`. An EVL output summary used to form
the scene field is provided in Appendix @fig:evl-summary.

== Pose encoding

Each candidate pose is expressed in the reference rig frame via
$(#sym_T)_{#fr_rig_ref <- #fr_cam} = (#sym_T)_{#fr_world <- #fr_rig_ref}^(-1) dot (#sym_T)_{#fr_world <- #fr_cam}$.
We encode translation and rotation using a 6D rotation representation and
Learnable Fourier Features (LFF) @zhou2019continuity @LFF-li2021. For each
candidate $i$ the pose encoder yields:

- raw pose vector $bold(xi)_i in bb(R)^9$ (implementation: `pose_vec`)
- pose embedding $(#sym_pose_emb)_i in bb(R)^(d_"pose")$ (implementation: `pose_enc`)
- candidate center $bold(c)_i in bb(R)^3$ in the rig frame (implementation: `candidate_center_rig_m`)

== Global context via pose-conditioned attention

We pool $bold(F)$ to a coarse grid of size $G times G times G$, yielding
$T = G^3$ tokens.
Each token is augmented with an LFF positional encoding of its voxel center in
the rig frame. Candidate pose embeddings serve as queries in a multi-head
cross-attention block:

#block[
  #align(center)[
    $ (#sym_query)_i = (#sym_W)_Q dot (#sym_pose_emb)_i $
  ]
]
#block[
  #align(center)[
    $
      (#sym_key)_j = (#sym_W)_K dot (#sym_token)_j + phi((#sym_pos)_j),
      quad (#sym_value)_j = (#sym_W)_V dot (#sym_token)_j
    $
  ]
]

The attention output yields a global context vector $(#sym_global)_i in bb(R)^C$
for each candidate (implementation: `global_feat`). We apply residual
connections and an MLP block for stability.

== Semidense projection statistics

To inject view-dependent evidence beyond the EVL voxel extent, we project
semi-dense SLAM points into each candidate view using the provided PyTorch3D
`PerspectiveCameras` (screen-space projection via `transform_points_screen`). For
candidate `q`, we compute per-view statistics:

- coverage ratio and empty fraction using a coarse `G x G` occupancy grid in the
  image plane (fraction of bins hit by at least one projected point)
- valid fraction of projected points (`z > 0` and inside image bounds)
- depth mean and depth standard deviation (optionally weighted by inverse-distance
  uncertainty when available)

These features form `sem_proj` with shape `(B, N, D_proj)`. We concatenate
`sem_proj` to the head input and optionally use FiLM modulation:

#block[
  #align(center)[
    $ (#sym_global)_i^("film") = (1 + (#sym_gamma)_i) ⊙ (#sym_global)_i + (#sym_beta)_i $
  ]
]

with per-channel $((#sym_gamma)_i, (#sym_beta)_i)$ predicted from `sem_proj`
@FiLM-perez2018.

== Frustum-aware cross-attention

We further encode view-conditioned evidence using a frustum token set. Each
projected semidense point contributes a token with features
`(x_norm, y_norm, depth_m, inv_dist_std)`. A multi-head cross-attention block
aggregates these tokens using pose embeddings as queries, producing
`sem_frustum` with shape `(B, N, C)`.

This stage approximates VIN-NBV's view-plane projection in a lightweight way:
it is explicitly candidate-conditioned while avoiding a full 2D CNN grid.

== Optional semidense and trajectory context

An optional PointNeXt-S encoder provides a global semidense embedding per
snippet @PointNeXt-qian2022. This embedding can FiLM-modulate `global_feat` or
be concatenated directly. We also include a lightweight trajectory encoder that
summarizes recent rig poses, yielding a context vector that is broadcast across
candidates.

== Feature fusion and CORAL head

The final per-candidate feature vector concatenates multiple candidate-specific
and snippet-level signals, e.g.

#block[
  #align(center)[
    $
      bold(h)_i = [(#sym_pose_emb)_i ; (#sym_global)_i ; bold(s)_i^("proj") ; bold(s)_i^("frustum") ; bold(z)^("point") ; bold(z)^("traj")]
    $
  ]
]

The head MLP produces `K-1` logits per candidate. CORAL interprets these as
cumulative probabilities and yields ordinal predictions and expected RRI
estimates @CORAL-cao2019.

This architecture preserves VIN-NBV's inductive bias (view-conditioned
projection features) while leveraging EVL's egocentric voxel representation and
providing explicit candidate conditioning throughout the pipeline.
