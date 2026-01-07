= Aria-VIN-NBV Architecture

#import "/typst/shared/macros.typ": *

This section describes the VIN v2 architecture with a focus on theoretical
fidelity to VIN-NBV and exact alignment with the current implementation. The
model predicts ordinal RRI scores for candidate views using a frozen EVL
backbone, explicit view-conditioned evidence, and a lightweight CORAL head.

#figure(
  image("/figures/VIN-NBV_diagram.png", width: 100%),
  caption: [VIN-NBV architecture overview (baseline reference; figure will be replaced by Streamlit exports).],
) <fig:vin-nbv-diagram>

#figure(
  [*Placeholder*: Insert Streamlit-exported VIN v2 architecture diagram (VIN Diagnostics → Summary tab).],
  caption: [Aria-VIN-NBV (VIN v2) architecture overview (Streamlit export pending).],
) <fig:vin-v2-diagram>

#figure(
  image("/figures/vin_v2/vin_v2_arch.png", width: 100%),
  caption: [Auto-generated VIN v2 module diagram (Graphviz; derived from the current implementation).],
) <fig:vin-v2-graphviz>

== Backbone and scene field

EVL lifts multi-view observations into a local voxel grid centered near the
latest rig pose and outputs multiple dense heads (occupancy probability,
centerness, free-space evidence, and observation counts) @EFM3D-straub2024. We
construct an input scene field $bold(F)_("in") in bb(R)^(B times C_"in" times D times H times W)$
by concatenating selected EVL head channels (occupancy prior, occupancy
evidence, observation counts, centerness, and free-space evidence) plus derived
features.

Let $bold(n)$ denote the per-voxel observation counts and $bold(o)^("pr")$ the
occupancy prior. We define

#block[
  #align(center)[
    $ bold(n)^("norm") = ("log"(1 + bold(n))) / ("log"(1 + "max"(bold(n)))) $
  ]
]

where the maximum is taken over the voxel grid (per batch element), and
derive

#block[
  #align(center)[
    $ bold(u) = 1 - bold(n)^("norm"), quad bold(s)^("new") = bold(u) dot.o bold(o)^("pr") $
  ]
]

Here $bold(u)$ is a soft unknown mask and $bold(s)^("new")$ is a new-surface
prior. The projected scene field is then
$bold(F) = phi("Conv"_(1 times 1 times 1)(bold(F)_("in"))) in bb(R)^(B times C times D times H times W)$,
implemented as `Conv3d + GroupNorm + GELU` @GroupNorm-wu2018. An EVL output summary used to form
the scene field is provided in Appendix @fig:evl-summary.

== Pose encoding

Each candidate pose is expressed in the reference rig frame via
// $(#sym_T)_{#fr_rig_ref <- #fr_cam} = (#sym_T)_{#fr_world <- #fr_rig_ref}^(-1) dot (#sym_T)_{#fr_world <- #fr_cam}$.
$#sym_T _(#fr_rig_ref^#fr_cam) = #sym_T _(#fr_world <- #fr_rig_ref)^(-1) dot #sym_T _(#fr_world <- #fr_cam)$.
We encode translation and rotation using a 6D rotation representation and
Learnable Fourier Features (LFF) @zhou2019continuity @LFF-li2021. For each
candidate $i$ the pose encoder yields:

- a low-dimensional pose vector $bold(xi)_i in bb(R)^9$ (translation + rotation-6D),
- an embedding $#sym_pose_emb _i in bb(R)^(d_"pose")$,
- the candidate center $bold(c)_i in bb(R)^3$ in the rig frame.

== Global context via pose-conditioned attention

We pool $bold(F)$ to a coarse grid of size $G times G times G$, yielding
$T = G^3$ tokens.
Each token is augmented with an LFF positional encoding of its voxel center in
the rig frame. Candidate pose embeddings serve as queries in a multi-head
cross-attention block @Transformer-vaswani2017.
// #block[
//   #align(center)[
//     $ (#sym_query)_i = (#sym_W)_Q dot (#sym_pose_emb)_i $
//   ]
// ]
// #block[
//   #align(center)[
//     $
//       (#sym_key)_j = (#sym_W)_K dot (#sym_token)_j + phi((#sym_pos)_j),
//       quad (#sym_value)_j = (#sym_W)_V dot (#sym_token)_j
//     $
//   ]
// ]
#block[
  #align(center)[
    $ #sym_query _i = #sym_W _Q dot #sym_pose_emb _i $
  ]
]
#block[
  #align(center)[
    $
      #sym_key _j = #sym_W _K dot #sym_token _j + phi(#sym_pos _j),
      quad #sym_value _j = #sym_W _V dot #sym_token _j
    $
  ]
]

The attention output yields a global context vector $(#sym_global)_i in bb(R)^C$
for each candidate. We apply residual connections and an MLP block for
stability.

== Coverage proxies and voxel gating

Candidate shells can place views outside EVL's local voxel extent. VIN v2
therefore computes a per-candidate voxel-evidence fraction by sampling the
normalized observation-count field $bold(n)^("norm")$ at the candidate camera
center in world coordinates and masking with (i) voxel in-bounds checks and
(ii) pose finiteness. This scalar serves as a lightweight reliability signal
for voxel-based features.
We denote this scalar by $v_i$ for candidate $i$.

We use $v_i$ in two complementary ways:

- *Gating*: a tiny learned sigmoid gate predicts $g_i in [0, 1]$ from
  $v_i$ and scales the global feature
  $#sym_global _i := g_i dot.o #sym_global _i$.
- *Feature*: we optionally append $(v_i, 1-v_i)$ to the final head input to
  make the head aware of evidence quality.

== Semidense projection statistics

To inject view-dependent evidence beyond the EVL voxel extent, we project
semi-dense SLAM points into each candidate view using a consistent screen-space
camera model aligned with the depth renderer. For candidate $q$, we compute
per-view statistics:

- coverage ratio and empty fraction using a coarse $G times G$ occupancy grid in the
  image plane (fraction of bins hit by at least one projected point)
- candidate visibility fraction $v_i^("sem")$: fraction of projected points
  that are valid ($z > 0$ and inside image bounds)
- depth mean and depth standard deviation (optionally weighted by inverse-distance
  uncertainty when available)

The visibility fraction $v_i^("sem")$ is a candidate-conditioned reliability
proxy that remains informative even when voxel features are weak. We concatenate
these projection statistics to the head input and use FiLM modulation:

#block[
  #align(center)[
    $ #sym_global _i^("film") = (1 + #sym_gamma _i) dot.o #sym_global _i + #sym_beta _i $
  ]
]

with per-channel $( #sym_gamma _i, #sym_beta _i)$ predicted from the projection
statistics vector @perez2017filmvisualreasoninggeneral.

== Frustum-aware cross-attention

We further encode view-conditioned evidence using a frustum token set. Each
projected semidense point contributes a token with features
$(x, y, z, nu, c)$, where $(x, y)$ are normalized image coordinates, $z$ is
metric depth, $nu$ is an inverse-distance uncertainty term, and $c$ is a
track-length feature (number of frames that observed the point), normalized
with a log scaling. A multi-head cross-attention block aggregates these tokens
using pose embeddings as queries, producing a per-candidate frustum feature.

This stage approximates VIN-NBV's view-plane projection in a lightweight way:
it is explicitly candidate-conditioned while avoiding a full 2D CNN grid.
We optionally add a learned visibility token-type embedding (valid vs invalid
projection) and optionally mask invalid tokens in attention; both toggles are
part of our ablation suite.

== Optional semidense and trajectory context

An optional PointNeXt-S encoder provides a global semidense embedding per
snippet @PointNeXt-qian2022. We treat PointNeXt as frozen and project its
pooled embedding to a fixed dimension. The encoder operates on semidense points
expressed in the reference rig frame and can ingest per-point uncertainty and
observation counts as additional channels. The resulting embedding can
FiLM-modulate the global context or be concatenated directly as a snippet-level cue.

VIN v2 also includes a lightweight trajectory encoder that embeds the snippet's
rig pose history (again expressed in the reference rig frame). We pool per-frame
trajectory embeddings and additionally apply candidate-query cross-attention to
obtain a candidate-specific context vector, providing a compact notion of
*where we came from* and improving disambiguation between geometrically similar
candidates.

== Feature fusion and CORAL head

The final per-candidate feature vector concatenates multiple candidate-specific
and snippet-level signals, e.g.

#block[
  #align(center)[
    $
      bold(h)_i =
      [#sym_pose_emb _i ;
        #sym_global _i ;
        bold(s)_i^("proj") ;
        bold(s)_i^("frustum") ;
        bold(z)^("traj") ;
        bold(z)^("point") ;
        v_i ;
        (1 - v_i)]
    $
  ]
]

The head MLP produces $K-1$ logits per candidate. CORAL interprets these as
cumulative probabilities and yields ordinal predictions and expected RRI
estimates @CORAL-cao2019.

This architecture preserves VIN-NBV's inductive bias (view-conditioned
projection features) while leveraging EVL's egocentric voxel representation and
providing explicit candidate conditioning throughout the pipeline.
