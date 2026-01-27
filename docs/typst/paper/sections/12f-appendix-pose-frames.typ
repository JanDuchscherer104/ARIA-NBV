#pagebreak()

= Appendix: VIN Pose Frames and Consistency Checks

#import "../../shared/macros.typ": *
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

This appendix summarizes the SE(3) frames provided to `VinModelV3.forward`,
using the notation from the coordinate conventions section. We verify the
conventions with offline-cache data loaded via `.configs/offline_only.toml`
(the debug config in `.vscode/launch.json`).

== Inputs to `forward`

Frame glossary (consistent with the main notation):

- #fr_world: global world frame (gravity-aligned).
- #fr_rig_ref: reference rig frame for the snippet (physical headset pose).
- #fr_cam: candidate camera frame in LUF coordinates (left-up-forward).
- #fr_voxel: EVL voxel grid frame (local, gravity-aware).

#figure(
  kind: "table",
  supplement: [Table],
  placement: none,
  caption: [SE(3) inputs passed into `VinModelV3.forward` and their frame meaning.],
  text(size: 7pt)[
    #table(
      columns: (auto, auto, auto, auto),
      align: (left, center, left, left),
      toprule(),
      table.header([Input], [Notation], [SE(3) meaning], [Notes]),
      midrule(),
      [#code-inline("candidate_poses_") #linebreak() #code-inline("world_cam")],
      [$#T(fr_world, fr_cam)$],
      [world #sym.arrow.r camera (PoseTW)],

      [Pose of each candidate camera in world coordinates.],
      [#code-inline("reference_pose_") #linebreak() #code-inline("world_rig")],
      [$#T(fr_world, fr_rig_ref)$],
      [world #sym.arrow.r rig reference (PoseTW)],

      [Physical rig pose; not gravity-aligned in cache.],
      [#code-inline("p3d_cameras")],
      [$#T(fr_cam, fr_world)$],
      [camera #sym.arrow.r world (PerspectiveCameras)],

      [PyTorch3D row-vector convention, with $#T(fr_cam, fr_world) = #T(fr_world, fr_cam)^(-1)$.],
      [#code-inline("backbone_out.") #linebreak() #code-inline("t_world_voxel")],
      [$#T(fr_world, fr_voxel)$],
      [world #sym.arrow.r voxel (PoseTW)],

      [EVL voxel grid pose for sampling voxel-aligned fields.],
      [#code-inline("backbone_out.") #linebreak() #code-inline("pts_world")],
      [$bold(p)^#fr_world$],
      [points in world frame],

      [Voxel-center points used for positional encoding and pooling.], bottomrule(),
    )
  ],
) <tab:vin-input-frames>

*Derived frames inside the model.* The pose encoder builds candidate poses in
the reference rig frame
$#T(fr_rig_ref, fr_cam) = (#T(fr_world, fr_rig_ref))^(-1)#T(fr_world, fr_cam)$.
Semidense and voxel projections are evaluated in camera frames derived from
`p3d_cameras` and must be consistent with the above pose encoding.

== Offline-cache consistency checks

Using cached batches (same config as the debug launcher), we verified:

- `candidate_poses_world_cam` and `p3d_cameras` are consistent:
  $#T(fr_cam, fr_world)$ reconstructed from `candidate_poses_world_cam`
  matches `p3d_cameras` with max abs error $<= 1e-6$.
- Undoing the CW90 display correction on poses *without* rotating
  `p3d_cameras` breaks that alignment (max abs error approx 1.2 in R,
  approx 10 in t), which corrupts semidense projection features.

== Current issues

- #textbf[CW90 correction mismatch]: `apply_cw90_correction=True` in
  `VinModelV3` only adjusts poses unless the caller pre-corrects
  `p3d_cameras`. This can desynchronize pose encoding and projection
  features.
- #textbf[Missing gravity-aligned reference in cache]: offline cache stores only
  `reference_pose_world_rig` (physical rig). The gravity-aligned sampling pose
  is not available, so cached datasets cannot be retrofitted without recompute.
- #textbf[Mixed display vs. physical frames]: display-only rotations (CW90) should
  not leak into model inputs.

== Recommended streamlining

1. #textbf[Single canonical frame per batch]: treat all cached inputs as physical
  rig frames; disable CW90 inside the model for cached datasets.
2. #textbf[Batch-level normalization helper]: if we want to undo CW90, apply it
  consistently to `candidate_poses_world_cam`, `reference_pose_world_rig`,
  and `p3d_cameras` (plus a `cw90_corrected` tag) before entering `forward`.
3. #textbf[Consistency guard]: add a lightweight assertion that checks
  $#T(fr_cam, fr_world)$ from `candidate_poses_world_cam` matches
  `p3d_cameras` (within tolerance) and emits a warning if not.
