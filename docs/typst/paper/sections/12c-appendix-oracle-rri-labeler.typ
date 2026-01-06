#pagebreak()

= Appendix: OracleRriLabeler Pipeline

#import "/typst/shared/macros.typ": *
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

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
$(#sym_dir) ~ "PS"(bold(mu), kappa)$ @PowerSpherical-deCao2020. Angular caps
are enforced *without rejection* by deterministically mapping the raw samples
into the target azimuth/elevation ranges (see
`oracle_rri/oracle_rri/pose_generation/positional_sampling.py`).

The candidate center sampling configuration used in our runs is summarized in
@tab:oracle-cfg-centers.

#figure(
  kind: "table",
  supplement: [Table],
  placement: none,
  caption: [Oracle candidate center sampling configuration.],
  text(size: 8.5pt)[
    #table(
      columns: (auto, auto),
      align: (left, left),
      toprule(),
      table.header([Parameter], [Value]),
      midrule(), [camera_label],
      [rgb], [reference_frame_index],
      [None (final pose)], [num_samples],
      [32], [oversample_factor],
      [1.5], [align_to_gravity],
      [true], [min_radius],
      [0.6 m], [max_radius],
      [3.0 m], [min_elev_deg],
      [-15#sym.degree], [max_elev_deg],
      [25#sym.degree], [delta_azimuth_deg],
      [170#sym.degree], [sampling_strategy],
      [uniform_sphere], [seed],
      [0], bottomrule(),
    )
  ],
) <tab:oracle-cfg-centers>

*Azimuth/elevation caps without rejection.* Let
$tilde((#sym_dir)) = (x, y, z)$ be a raw unit direction. We define

#block[
  #align(center)[
    $
      psi = "atan2"(x, z),
      quad
      theta = "atan2"(y, sqrt(x^2 + z^2))
    $
  ]
]

If the azimuth spread is capped to $Delta_"az" < 2 pi$, we scale
$psi' = psi dot (Delta_"az" / (2 pi))$ and overwrite $(x, z) <- ("sin"(psi'), "cos"(psi'))$.
To cap elevation, we map $y = "sin"(theta)$ from $[-1, 1]$ into the target band:

#block[
  #align(center)[
    $
      y_"min" = "sin"(theta_"min"), \
      y_"max" = "sin"(theta_"max"), \
      y' = y_"min" + (y + 1)(y_"max" - y_"min")/2
    $
  ]
]

and rescale the horizontal components to preserve unit norm:
$(x, z) <- (x, z) dot sqrt(1 - (y')^2) / sqrt(x^2 + z^2)$, followed by
renormalization.

== Candidate orientations

For each candidate center $(#sym_center)_i$, we construct a base camera
orientation according to the configured `ViewDirectionMode`
(`oracle_rri/oracle_rri/pose_generation/orientations.py`). The most common mode
is radial "look-away", which points the camera optical axis away from the
reference position.

The orientation configuration used in our runs is summarized in
@tab:oracle-cfg-orient.

#figure(
  kind: "table",
  supplement: [Table],
  placement: none,
  caption: [Oracle candidate orientation configuration.],
  text(size: 8.5pt)[
    #table(
      columns: (auto, auto),
      align: (left, left),
      toprule(),
      table.header([Parameter], [Value]),
      midrule(), [view_direction_mode],
      [radial_away], [view_max_azimuth_deg],
      [60#sym.degree], [view_max_elevation_deg],
      [30#sym.degree], [view_roll_jitter_deg],
      [0#sym.degree], bottomrule(),
    )
  ],
) <tab:oracle-cfg-orient>

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
the candidate pose is $(#sym_T)_{#fr_world <- (#fr_cam)_i} = (bold(R)_i,
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
Let #sym_mesh be the GT mesh and $cal(B)$ be the snippet occupancy AABB.

The pruning configuration used in our runs is summarized in
@tab:oracle-cfg-prune.

#figure(
  kind: "table",
  supplement: [Table],
  placement: none,
  caption: [Oracle candidate pruning configuration.],
  text(size: 8.5pt)[
    #table(
      columns: (auto, auto),
      align: (left, left),
      toprule(),
      table.header([Parameter], [Value]),
      midrule(), [min_distance_to_mesh],
      [0.4 m], [ensure_collision_free],
      [true], [collision_backend],
      [pytorch3d], [ray_subsample],
      [128], [step_clearance],
      [0.1 m], [ensure_free_space],
      [true], bottomrule(),
    )
  ],
) <tab:oracle-cfg-prune>

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
      quad alpha_k in [0, 1], \
      min_(bold(m) in #sym_mesh) || bold(s)_(i,k) - bold(m) ||_2 > delta
    $
  ]
]

*Free space.* Reject candidates outside the occupancy bounds:
$(#sym_center)_i in cal(B)$.

== Depth rendering with PyTorch3D

Depth rendering is implemented in
`oracle_rri/oracle_rri/rendering/pytorch3d_depth_renderer.py` using PyTorch3D's
`PerspectiveCameras`, `RasterizationSettings`, and `MeshRasterizer`
@PyTorch3D-Cameras-2025. Given a candidate pose $(#sym_T)_{#fr_world <- #fr_cam}$
and camera intrinsics (focal length and principal point), we build a batched
`PerspectiveCameras` instance with `in_ndc=false` and rasterize the GT mesh.

We use hard z-buffering with `faces_per_pixel=1` and `blur_radius=0`, clip
triangles closer than `znear`, and expose performance knobs such as
`bin_size`/`max_faces_per_bin` via the renderer config.

The depth rendering configuration used in our runs is summarized in
@tab:oracle-cfg-render.

#figure(
  kind: "table",
  supplement: [Table],
  placement: none,
  caption: [Oracle depth rendering configuration (effective settings).],
  text(size: 8.5pt)[
    #table(
      columns: (14em, auto),
      align: (left, left),
      toprule(),
      table.header([Parameter], [Value]),
      midrule(), [max_candidates_final],
      [16], [oversample_factor],
      [1.0], [resolution_scale],
      [0.5], [znear],
      [0.001 m], [zfar],
      [20.0 m], [faces_per_pixel],
      [1], [blur_radius],
      [0.0], [bin_size],
      [0], [max_faces_per_bin],
      [None], [cull_backfaces],
      [false (render both sides)], bottomrule(),
    )
  ],
) <tab:oracle-cfg-render>

*Pose conventions.* Our poses are stored as world #sym.arrow.l camera transforms. PyTorch3D
expects a world #sym.arrow.r view mapping using row vectors:
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
      x_"ndc" = - (u + 1/2 - W/2) (2/s), \
      y_"ndc" = - (v + 1/2 - H/2) (2/s)
    $
  ]
]

We then unproject in NDC space:

#block[
  #align(center)[
    $
      bold(p)_("world") =
      "unproject"((x_"ndc", y_"ndc", (#sym_depth)_i(u,v)))
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

The backprojection configuration used in our runs is summarized in
@tab:oracle-cfg-backproj.

#figure(
  kind: "table",
  supplement: [Table],
  placement: none,
  caption: [Oracle backprojection configuration.],
  text(size: 8.5pt)[
    #table(
      columns: (14em, auto),
      align: (left, left),
      toprule(),
      table.header([Parameter], [Value]),
      midrule(), [backprojection_stride],
      [1], bottomrule(),
    )
  ],
) <tab:oracle-cfg-backproj>

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

The oracle scoring uses fixed aggregation choices (no additional tunable
hyperparameters). The effective settings are summarized in
@tab:oracle-cfg-score.

#figure(
  kind: "table",
  supplement: [Table],
  placement: none,
  caption: [Oracle RRI scoring settings (fixed).],
  text(size: 8.5pt)[
    #table(
      columns: (16em, auto),
      align: (left, left),
      toprule(),
      table.header([Setting], [Value]),
      midrule(), [mesh crop],
      [AABB from semidense + candidate bounds], [accuracy (P #sym.arrow.r M)],
      [mean point #sym.arrow.r triangle distance], [completeness (M #sym.arrow.r P)],
      [mean triangle #sym.arrow.r point distance], [bidirectional],
      [accuracy + completeness], [denominator stabilizer],
      [clamp_min $1e-12$], bottomrule(),
    )
  ],
) <tab:oracle-cfg-score>

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

The binning configuration used in our runs is summarized in
@tab:oracle-cfg-binning.

#figure(
  kind: "table",
  supplement: [Table],
  placement: none,
  caption: [Ordinal binning configuration for CORAL training.],
  text(size: 8.5pt)[
    #table(
      columns: (14em, auto),
      align: (left, left),
      toprule(),
      table.header([Parameter], [Value]),
      midrule(), [num_classes $K$],
      [15], [edge fit],
      [quantiles at $k/K$, $k=1..K-1$], [`bucketize`],
      [right=false], [fallback when edges collapse],
      [uniform linspace between min/max], bottomrule(),
    )
  ],
) <tab:oracle-cfg-binning>
