= Aria-VIN-NBV Architecture

#import "../../shared/macros.typ": *

This section describes the currently implemented VIN v3 candidate scorer that
we train against oracle RRI labels. The design emphasizes explicit
view-conditioned evidence and a lightweight CORAL head, with EVL providing the
voxel backbone @EFM3D-straub2024. We separate stable core computations from
optional ablations to keep the implementation and evaluation contract clear.

== Core design and optional ablations

To avoid ambiguity between what is implemented and what is exploratory, we
separate the model into (i) a stable core that is always present and (ii)
optional modules evaluated via ablations.

- *Core*: EVL voxel evidence, pose encoding, pose-conditioned global context,
  per-candidate semi-dense projection statistics, and a lightweight CORAL head.
- *View-conditioned evidence*: candidate-specific cues obtained by projecting
  both semi-dense points and pooled voxel centers into each candidate view.
- *Optional ablations*: trajectory-context encoder and alternative feature
  fusion mechanisms evaluated separately.

#figure(
  image("/figures/VIN-NBV_diagram.png", width: 100%),
  caption: [VIN-NBV architecture overview (baseline reference).],
) <fig:vin-nbv-diagram>

#figure(
  image("/figures/vin_v2/vin_v2_arch.png", width: 100%),
  caption: [Legacy VIN v2 module diagram (Graphviz; retained for reference).],
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
implemented as `Conv3d + GroupNorm + GELU` @GroupNorm-wu2018. An example
occupancy-prior slice from the diagnostics dashboard is shown in
@fig:evl-occ-pr.

#figure(
  image("/figures/app/scene_field_occ_pr.png", width: 100%),
  caption: [Streamlit diagnostics view of an EVL occupancy-prior slice (occ_pr) used to build the scene field.],
) <fig:evl-occ-pr>

An EVL output summary used to form the scene field is provided in Appendix @fig:evl-summary.

== Pose encoding

Each candidate pose is expressed in the reference rig frame via
$#T(fr_rig_ref, fr_cam) = #T(fr_world, fr_rig_ref)^(-1) dot #T(fr_world, fr_cam)$.
We encode translation and rotation using a 6D rotation representation
@zhou2019continuity and Learnable Fourier Features (LFF) @LFF-li2021.
The 6D choice follows the continuity analysis of @zhou2019continuity: commonly
used low-dimensional parameterizations (Euler angles, axis--angle, quaternions)
introduce discontinuities that can fragment the learning signal and lead to
unstable optimization, while a continuous Euclidean representation of $"SO"(3)$
is not possible in four or fewer dimensions. The 6D representation (two
rotation-matrix columns) is a simple continuous alternative that maps to a
valid rotation via a differentiable Gram--Schmidt-like orthogonalization,
avoiding explicit orthogonality constraints of a full $3 times 3$ matrix.
We prefer 6D over the minimal 5D continuous variant because it is simpler to
implement and performed similarly or better in practice @zhou2019continuity.
For each candidate $i$ the pose encoder yields:

- a low-dimensional pose vector $bold(xi)_i in bb(R)^9$ (translation + rotation-6D),
- an embedding $#(s.pose_emb)_i in bb(R)^(d_"pose")$,
- the candidate center $bold(c)_i in bb(R)^3$ in the rig frame.

== Global context via pose-conditioned attention

We pool $bold(F)$ to a coarse grid of size $G times G times G$, yielding
$T = G^3$ tokens.
Each token is augmented with an LFF positional encoding of its voxel center in
the rig frame. Candidate pose embeddings serve as queries in a multi-head
cross-attention block @Transformer-vaswani2017.
// #block[
//   #align(center)[
//     $ (#s.query)_i = (#s.W)_Q dot (#(s.pose)_emb)_i $
//   ]
// ]
// #block[
//   #align(center)[
//     $
//       (#s.key)_j = (#s.W)_K dot (#s.token)_j + phi((#s.pos)_j),
//       quad (#s.value)_j = (#s.W)_V dot (#s.token)_j
//     $
//   ]
// ]
#block[
  #align(center)[
    $ #(s.query)_i = #(s.W)_Q dot #(s.pose_emb)_i $
  ]
]
#block[
  #align(center)[
    $
      #(s.key)_j = #(s.W)_K dot #(s.token)_j + phi(#(s.pos)_j),
      quad #(s.value)_j = #(s.W)_V dot #(s.token)_j
    $
  ]
]

The attention output yields a global context vector $(#s.global)_i in bb(R)^C$
for each candidate. We apply residual connections and an MLP block for
stability.

== Coverage proxies and candidate validity

Candidate shells can place views outside EVL's local voxel extent. We therefore
compute a per-candidate voxel-evidence fraction $v_i$ by sampling the
normalized observation-count field $bold(n)^("norm")$ at the candidate camera
center in world coordinates and masking with (i) voxel in-bounds checks and
(ii) pose finiteness. This scalar is used to form the candidate validity mask
and as a coverage proxy for training diagnostics and optional loss reweighting.
VIN v3 no longer applies a hard gate to the global feature to avoid silently
suppressing gradients for low-coverage candidates.
Together with the semi-dense visibility fraction $v_i^("sem")$, we define
`candidate_valid` as the conjunction of finite pose, $v_i > 0$, and
$v_i^("sem") > 0$.

== Voxel projection FiLM

To modulate global features with candidate-dependent evidence, we pool voxel
centers to the same coarse grid used in global attention ($G_"pool"^3$ points),
project those centers into each candidate view, and summarize their
screen-space coverage and depth statistics. A linear FiLM head predicts
per-channel $(#(s.gamma)_i, #(s.beta)_i)$ from these projection statistics and
applies

#block[
  #align(center)[
    $ #(s.global)_i^("film") = (1 + #(s.gamma)_i) dot.o #(s.global)_i + #(s.beta)_i $
  ]
]

This provides lightweight view-conditioned modulation without introducing
additional token-level attention @perez2017filmvisualreasoninggeneral.

== Semi-dense projection statistics

To inject view-dependent evidence beyond the EVL voxel extent, we project the
semi-dense SLAM points into each candidate view using the same screen-space
camera model as the depth renderer. A projected point is valid if it is finite,
in front of the camera ($z > 0$), and within the image bounds. For each
candidate we compute:

- coverage ratio and empty fraction using a coarse $G times G$ image grid
  (fraction of bins hit by at least one valid point),
- visibility fraction $v_i^("sem")$ as the (optionally reliability-weighted)
  fraction of valid projections among finite projections,
- depth mean and depth standard deviation over valid points.

Reliability weights combine normalized observation count and inverse depth
uncertainty when available, attenuating noisy points without hard masking.
The resulting projection statistics are concatenated to the head input.
We subsample the padded semi-dense point cloud using the per-snippet
`lengths` field to avoid including invalid padding in the projection.

More explicitly, let $bold(p)_j in bb(R)^3$ denote a semidense point in the
world frame and let candidate $i$ have camera pose $#T(fr_world, fr_cam)_i$ and
intrinsics $K_i$. We compute camera-frame coordinates

#block[
  #align(center)[
    $ bold(p)^c_(i,j) = #T(fr_cam, fr_world)_i dot bold(p)_j $
  ]
]

and screen-space projection
$(u_(i,j), v_(i,j), z_(i,j)) = op("proj")(K_i, bold(p)^c_(i,j))$.
The validity mask is

#block[
  #align(center)[
    $
      m_(i,j) =
      bb(1)["finite"] dot bb(1)[z_(i,j) > 0] dot
      bb(1)[0 <= u_(i,j) < W_i] dot bb(1)[0 <= v_(i,j) < H_i]
    $
  ]
]

We form a $G times G$ occupancy grid by binning $(u, v)$ and counting valid
projections per bin. Coverage is the fraction of non-empty bins and
empty fraction is its complement. With per-point reliability weights
$w_(i,j) = a_(i,j) b_(i,j)$, where

#block[
  #align(center)[
    $
      a_(i,j) = "log"(1 + n_"obs"(i,j)) / "log"(1 + n_"obs"^"max"), quad
      b_(i,j) = "clip"((sigma_d^(-1)(i,j) - s_"min") / (s_"p95" - s_"min"), 0, 1)
    $
  ]
]

the visibility fraction is
$
  v_i^("sem") =
  (sum_j w_(i,j) m_(i,j)) / (sum_j w_(i,j) f_(i,j))
$,
with $f_(i,j) = bb(1)["finite"]$ the finite mask. Depth mean and standard deviation are computed
as weighted moments over the valid set.

== Semi-dense projection grid CNN

We also build a per-candidate $G times G$ grid from the projected semi-dense
points, storing occupancy, depth mean, and depth standard deviation per bin.
A tiny 2D CNN encodes this grid into a compact feature vector that is appended
to the head input. This adds local view-plane structure while remaining far
lighter than a full image encoder.

For each candidate $i$ and bin $b$, we compute
$O_(i,b) = bb(1)[c_(i,b) > 0]$, the per-bin mean depth $mu_(i,b)$, and
standard deviation $sigma_(i,b)$. The grid tensor
$bold(G)_i = [O_(i,*) ; mu_(i,*) ; sigma_(i,*)]$ is encoded by a small CNN to
produce $bold(s)_i^("grid")$.

== Optional trajectory context

An optional trajectory encoder embeds the snippet's rig pose history (expressed
in the reference rig frame). Per-frame pose encodings are pooled and can also
be attended by candidate pose queries to yield a candidate-specific context
vector, providing a compact notion of motion history and improving
disambiguation between geometrically similar candidates. This module is
disabled by default and is evaluated as an ablation.

== Feature fusion and CORAL head

The final per-candidate feature vector concatenates multiple candidate-specific
and snippet-level signals, e.g.

#block[
  #align(center)[
    $
      bold(h)_i =
      [#(s.pose_emb)_i ;
        #(s.global)_i ;
        bold(s)_i^("proj") ;
        bold(s)_i^("grid") ;
        bold(z)^("traj")]
    $
  ]
]

The head MLP produces $K-1$ logits per candidate. CORAL interprets these as
cumulative probabilities and yields ordinal predictions and expected RRI
estimates @CORAL-cao2019.

This architecture preserves VIN-NBV's inductive bias (view-conditioned
projection features) while leveraging EVL's egocentric voxel representation and
providing explicit candidate conditioning throughout the pipeline.
