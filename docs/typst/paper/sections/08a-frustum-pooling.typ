= Semidense Frustum Pooling and View-Conditioned Tokens

#import "/typst/shared/macros.typ": *

A key requirement for NBV is that features be explicitly conditioned on the
candidate view. In VIN v2 we approximate VIN-NBV's view-plane projection
features by projecting the current semi-dense SLAM point cloud into each
candidate camera and aggregating the resulting per-point tokens with multi-head
cross-attention.

Representative frusta visualizations are included in the appendix.

== Semidense projection tokens

Let $#sym_points _t$ be the current reconstruction point set (semi-dense
SLAM points) in
world coordinates. For each candidate camera, we project points to screen-space
using a standard pinhole camera model consistent with the depth renderer. Each
projected point contributes a token:

#block[
  #align(center)[
    $ bold(tau) = (u, v, z, nu, c) $
  ]
]

where $(u, v)$ are pixel coordinates normalized to $[-1, 1]^2$, $z$ is the
positive depth in camera space, and $nu$ is an optional per-point
inverse-distance uncertainty from the SLAM point cloud. The final feature $c$
is the per-point observation count (track length), i.e., the number of snippet
trajectory frames that observed the point, normalized with a configurable
log scaling (e.g., $c' = log(1 + c) / log(1 + c_"max")$).

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
