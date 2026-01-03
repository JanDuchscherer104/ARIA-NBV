#pagebreak()

= Appendix: OracleRriLabeler Pipeline

#import "/typst/shared/macros.typ": *

This appendix provides a detailed, code-faithful overview of the oracle label
pipeline implemented in `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py`.
The pipeline maps a snippet (trajectory, semidense SLAM points, and GT mesh) to
per-candidate oracle RRI scores and the fitted ordinal binning used for CORAL
training.

== Pipeline overview

Given an ASE-EFM snippet with mesh supervision, `OracleRriLabeler` executes:

1. *Candidate sampling* (`CandidateViewGenerator`): sample candidate camera
   centers on a constrained spherical shell and assign orientations, then prune
   invalid candidates using rule objects.
2. *Depth rendering* (`CandidateDepthRenderer` + `Pytorch3DDepthRenderer`):
   rasterize the GT mesh from each candidate pose to obtain depth maps.
3. *Backprojection* (`build_candidate_pointclouds`): unproject valid depth
   pixels to obtain per-candidate point clouds in world coordinates and fuse
   with the semidense reconstruction point set.
4. *Oracle scoring* (`OracleRRI`): compute Chamfer-based reconstruction quality
   before/after adding each candidate and report oracle RRI.
5. *Ordinal binning* (`RriOrdinalBinner`): fit quantile edges and map continuous
   RRIs to ordinal labels and CORAL level targets.

== Candidate center sampling (hyperspherical shell)

Let $(#sym_T)_{#fr_world <- #fr_rig_ref}$ be the reference rig pose (world
coordinates). Candidate centers are sampled by drawing a direction
$(#sym_dir)_i in bb(S)^2$ and a radius $r_i in [r_"min", r_"max"]$, then
transforming the resulting offset into world coordinates:

#block[
  #align(center)[
    $
      (#sym_offset)_i = r_i (#sym_dir)_i,
      quad (#sym_center)_i = (#sym_T)_{#fr_world <- #fr_rig_ref} (#sym_offset)_i
    $
  ]
]

We draw directions either uniformly on the sphere, or from a forward-biased
Power Spherical distribution centered at the device forward axis
$(#sym_dir) ~ cal(PS)(bold(mu), kappa)$ @PowerSpherical-deCao2020. Angular caps
are enforced *without rejection* by deterministically mapping the raw samples
into the target azimuth/elevation ranges (see
`oracle_rri/oracle_rri/pose_generation/positional_sampling.py`).

== Candidate orientations

For each candidate center $(#sym_center)_i$, we construct a base camera
orientation according to the configured `ViewDirectionMode`
(`oracle_rri/oracle_rri/pose_generation/orientations.py`). The most common mode
is radial "look-away", which points the camera optical axis away from the
reference position.

Let $bold(u)_("wup")$ be the world up vector and $bold(t)_("ref")$ the reference
translation. Define the forward axis

#block[
  #align(center)[
    $
      bold(z)_i =
      ( (#sym_center)_i - bold(t)_("ref") ) / (|| (#sym_center)_i - bold(t)_("ref") ||_2 + epsilon)
    $
  ]
]

and construct a roll-stable orthonormal basis

#block[
  #align(center)[
    $
      bold(y)_i =
      ( bold(u)_("wup") - (bold(u)_("wup") dot bold(z)_i) bold(z)_i )
      / (|| bold(u)_("wup") - (bold(u)_("wup") dot bold(z)_i) bold(z)_i ||_2 + epsilon),
      quad
      bold(x)_i = bold(y)_i times bold(z)_i
    $
  ]
]

The resulting rotation is $bold(R)_i = [bold(x)_i, bold(y)_i, bold(z)_i]$ and
the candidate pose is $(#sym_T)_{#fr_world <- #fr_cam_i} = (bold(R)_i,
(#sym_center)_i)$.

*View jitter.* After base pose construction, we optionally apply bounded yaw,
pitch, and roll perturbations in the camera frame. Using yaw about +Y, pitch
about +X, and roll about +Z, we form a right-multiplicative delta rotation

#block[
  #align(center)[
    $
      bold(R)_i^("jitter") =
      bold(R)_y(delta psi_i) bold(R)_x(delta theta_i) bold(R)_z(delta phi_i)
    $
  ]
]

and update $bold(R)_i <- bold(R)_i bold(R)_i^("jitter")$.

== Candidate pruning rules

Candidates are oversampled and then filtered by modular rule objects
(`oracle_rri/oracle_rri/pose_generation/candidate_generation_rules.py`).
Let #sym_mesh be the GT mesh and $\mathcal{B}$ be the snippet occupancy AABB.

*Mesh clearance.* Reject candidates too close to the mesh:

#block[
  #align(center)[
    $
      d_i = min_(bold(m) in #sym_mesh) || (#sym_center)_i - bold(m) ||_2,
      quad
      d_i > d_"min"
    $
  ]
]

*Path collision.* Reject candidates whose straight segment from the reference to
the candidate center intersects the mesh. In the discretized variant, we sample
points along the segment and require a minimum clearance:

#block[
  #align(center)[
    $
      bold(s)_(i,k) = bold(t)_("ref") + alpha_k ((#sym_center)_i - bold(t)_("ref")),
      quad alpha_k in [0, 1],
      quad
      min_(bold(m) in #sym_mesh) || bold(s)_(i,k) - bold(m) ||_2 > delta
    $
  ]
]

*Free space.* Reject candidates outside the occupancy bounds:
$(#sym_center)_i in \mathcal{B}$.

== Depth rendering with PyTorch3D

Depth rendering is implemented in
`oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py` using PyTorch3D's
`PerspectiveCameras`, `RasterizationSettings`, and `MeshRasterizer`
@PyTorch3D-Cameras-2025. Given a candidate pose (#sym_T)_{#fr_world <- #fr_cam}
and camera intrinsics (focal length and principal point), we build a batched
`PerspectiveCameras` instance with `in_ndc=false` and rasterize the GT mesh.

*Pose conventions.* Our poses are stored as world<-camera transforms. PyTorch3D
expects a world→view mapping using row vectors:
$bold(x)_("cam") = bold(x)_("world") bold(R) + bold(T)$.
We therefore invert the pose and transpose the rotation before passing it to
PyTorch3D (see in-code comments in the renderer).

The rasterizer outputs a z-buffer $(#sym_depth)_i(u,v)$ and a face-index buffer
`pix_to_face_i(u,v)`. Valid pixels satisfy `pix_to_face >= 0` and are further
clipped to `znear < depth < zfar`.

== Backprojection and NDC alignment

We convert each depth image to a point cloud via
`_backproject_depths_p3d_batch` in
`oracle_rri/oracle_rri/rendering/candidate_pointclouds.py`. For pixel centers
$(u + 1/2, v + 1/2)$ we first map to PyTorch3D's *normalized device coordinates*
(NDC) using the convention (+X left, +Y up) and the scale
$s = min(H, W)$:

#block[
  #align(center)[
    $
      x_"ndc" = - (u + 1/2 - W/2) (2/s),
      quad
      y_"ndc" = - (v + 1/2 - H/2) (2/s)
    $
  ]
]

We then unproject in NDC space:

#block[
  #align(center)[
    $
      bold(p)_("world") =
      "unproject"((x_"ndc", y_"ndc", (#sym_depth)_i(u,v)); "from_ndc"=1)
    $
  ]
]

*Why NDC?* With `PerspectiveCameras(in_ndc=false)` and non-square images,
PyTorch3D internally uses an NDC-like normalization tied to $min(H, W)$.
Backprojecting from pixel coordinates (`from_ndc=false`) yields points in a
different screen convention than the rasterizer, which empirically leads to
backprojected points that do not lie on the rendered mesh. Converting pixels to
the same NDC convention used by the rasterizer makes depth unprojection and
rasterization consistent.

The per-candidate point set is then
$(#sym_points)_(q_i) = { bold(p)_("world") : (u,v) "valid" }$, optionally
subsampled by a stride to control point count.

== Oracle RRI computation

Let $#sym_points _t$ be the semidense SLAM reconstruction for the snippet and
$(#sym_points)_(q_i)$ the candidate point cloud rendered from the GT mesh.
Oracle RRI is computed by comparing the point-to-mesh reconstruction quality
before and after adding the candidate:

#block[
  #align(center)[
    $
      (#sym_points)_(t union q_i) = #sym_points _t union (#sym_points)_(q_i)
    $
  ]
]

#block[
  #align(center)[
    $
      "RRI"(q_i) =
      ("CD"(#sym_points _t, #sym_mesh)
      - "CD"((#sym_points)_(t union q_i), #sym_mesh))
      / ("CD"(#sym_points _t, #sym_mesh) + epsilon)
    $
  ]
]

In our implementation (`oracle_rri/oracle_rri/rri_metrics/oracle_rri.py`), the
GT mesh is cropped to a combined occupancy AABB covering semidense and
candidate points to reduce compute. Distances are evaluated with PyTorch3D
point-mesh primitives on GPU.

== Ordinal binning for CORAL

To train with CORAL, we discretize continuous RRIs into $K$ ordered bins using
empirical quantile edges (`oracle_rri/oracle_rri/rri_metrics/rri_binning.py`).
Given a stream of oracle RRIs $\{r_n\}_{n=1}^N$, we compute
$K-1$ edges at the quantiles $k/K$:

#block[
  #align(center)[
    $
      e_k = "Quantile"( {r_n}, k/K),
      quad k in {1, dots, K-1}
    $
  ]
]

The ordinal label is then the number of edges below the value (implemented via
`torch.bucketize`):

#block[
  #align(center)[
    $
      y(r) = sum_(k=1)^(K-1) 1[r > e_k],
      quad y(r) in {0, dots, K-1}
    $
  ]
]

CORAL represents the label $y$ as binary level targets
$t_k = 1[y > k]$ for $k = 0, dots, K-2$ @CORAL-cao2019. The fitted binner also
provides class priors and cumulative threshold priors
$P(y > k)$, which we use for prior-aligned bias initialization and optional
balanced threshold losses.

