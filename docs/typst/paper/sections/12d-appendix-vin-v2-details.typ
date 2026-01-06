#pagebreak()

= Appendix: VIN v2 Implementation Notes

#import "/typst/shared/macros.typ": *

This appendix collects implementation-level details of the VIN v2 architecture
that are too dense for the main architecture section. The goal is to keep the
main text readable while providing a precise mapping between theory and the
current `VinModelV2` implementation.

== Pose representation and rotation-6D

VIN v2 represents candidate poses as $#sym_T _(#fr_rig_ref <- #fr_cam) in "SE"(3)$.
We encode rotation via the continuous 6D representation obtained by taking the
first two columns of the rotation matrix and flattening them into a 6-vector.
This avoids discontinuities of Euler angles and improves learning stability
@zhou2019continuity.

== Candidate-conditioned semidense visibility fraction

The semidense view conditioning uses PyTorch3D screen-space projection
(`transform_points_screen`) to decide which semidense points are visible from
candidate $i$. A point is considered valid if it projects to finite image
coordinates, has positive depth in the candidate camera frame, and lies within
image bounds. We define the candidate-conditioned visibility fraction

#block[
  #align(center)[
    $
      v_i^("sem") =
      (1)/(max(1, |#sym_points _t|))
      sum_(bold(p) in #sym_points _t) bb(1)["valid"_(i)(bold(p))]
    $
  ]
]

and refer to this scalar as `semidense_candidate_vis_frac` throughout the
paper. This is a proxy for how much semidense evidence can be reused to score
candidate $i$, even when voxel features are out-of-bounds.

== Semidense frustum tokens and masking

For the frustum attention block, we form tokens
$bold(tau)_(i,k) = (u, v, z, nu, c)$ from the projected points, where $(u, v)$
are normalized image coordinates, $z$ is depth, $nu$ is inverse-distance
uncertainty, and $c$ is per-point observation count (track length). We support
two candidate-dependent mechanisms:

- token-type embedding: add a learned embedding depending on whether the point
  is a valid projection,
- attention masking: optionally mask invalid tokens in cross-attention.

Both toggles are treated as ablation knobs to understand whether the model
benefits more from explicitly encoding *missing visibility* or from focusing
compute on the subset of valid projected points.
