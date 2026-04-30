= Semi-dense Frustum Pooling and View-Conditioned Tokens

#import "../../shared/macros.typ": *

// TODO(paper-cleanup): This section overlaps with Section 6’s semidense projection branch.
// Decide whether to keep it as a short “optional ablation” description here or move the full
// token/MHCA formulation to an appendix to reduce redundancy.

A key requirement for NBV is that features be explicitly conditioned on the
candidate view. In VIN v2 we approximate VIN-NBV's view-plane projection
features by projecting the current semi-dense SLAM point cloud into each
candidate camera and aggregating the resulting per-point tokens with multi-head
cross-attention.
We treat this frustum token aggregation as an optional module evaluated via
ablations. It is part of the VIN v2 feature set and is disabled in the current
VIN v3 baseline unless explicitly enabled for ablation.

Representative frusta visualizations are included in the appendix.
Representative projection diagnostics are shown in @fig:oracle-fusion-diagnostics.
// TODO(paper-cleanup): Point to the exact appendix figure(s) for frusta (the current reference
// is vague and may be stale after figure moves).

// <rm>
// Detailed VINv2 frustum-token formulation. This largely duplicates the VINv3 semidense
// projection branch in Section 6 and is not part of the VINv3 baseline. Keep only the short
// “optional ablation” description above, or move this technical detail to an appendix.
== Semi-dense projection tokens

Let $#(symb.oracle.points)_t$ be the current reconstruction point set (semi-dense
SLAM points) in
world coordinates. For each candidate camera, we project points to screen-space
using a standard pinhole camera model consistent with the depth renderer. Each
projected point contributes a token:
// TODO(paper-cleanup): “pinhole camera model” is imprecise for Aria fisheye; clarify that we
// use the PerspectiveCameras approximation (same as oracle rendering) and keep terminology
// consistent with slides_4.typ.

#block[
  #align(center)[
    $ bold(tau) = (u, v, z, sigma_(rho), n_"obs") $
  ]
]
// TODO(paper-cleanup): Align symbols with `macros.typ` (#symb.vin.inv_dist_std, #symb.vin.n_obs,
// and `bold(...)` for vector-valued tokens); consider defining token dims and normalization
// consistently with the semidense-projection equations in Section 6.

where $(u, v)$ are pixel coordinates normalized to $[-1, 1]^2$, $z$ is the
positive depth in camera space, $sigma_(rho)$ is the per-point inverse-distance
standard deviation (`inv_dist_std`), and $n_"obs"$ is the per-point observation
count (track length) normalized with a configurable log scaling (e.g.,
$n'_"obs" = log(1 + n_"obs") / log(1 + n_"obs,max")$).

A boolean mask marks valid projected points (finite, $z>0$, and inside image
bounds). We truncate to a fixed maximum number of points per candidate for
stable compute. Optionally, we add a learned token-type embedding for valid vs
invalid projections and optionally mask invalid tokens in attention.

The fraction of valid projected points provides a view-specific reliability
score and is logged as a key diagnostic signal.

== Multi-head cross-attention

The sampled tokens are projected to a fixed channel dimension and aggregated
with a multi-head attention block. Candidate pose embeddings act as queries,
while token features and normalized pixel coordinates act as keys and values.
This design provides explicit view dependence while keeping the model small
@Transformer-vaswani2017.
// </rm>
