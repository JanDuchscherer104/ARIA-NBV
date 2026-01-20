= Dataset and Inputs

We use Aria Synthetic Environments (ASE), a large-scale synthetic dataset of
100,000 procedurally generated indoor scenes with realistic Aria sensor
simulations @ProjectAria-ASE-2025 @SceneScript-avetisyan2024. ASE provides
multi-camera RGB streams, semi-dense SLAM points, accurate trajectories, and
scene meshes, making it suitable for oracle RRI computation and supervised NBV
training.

Each scene is recorded as a single egocentric trajectory with synchronized RGB,
SLAM cameras, and inertial data. We follow the EFM3D data model and represent
snippet windows as fixed-length snippets of 20 frames at 10 Hz (2 s), with a
stride of 10 frames (1 s) between consecutive snippets @EFM3D-straub2024. Our
experiments use the ASE public release v1.0 exported via the ATEK WebDataset
preprocessing pipeline (v0.1.1), which resizes RGB frames to 240 #sym.times 240, depth
frames to 240 #sym.times 240, and SLAM grayscale frames to 240 #sym.times 320 while preserving
per-frame calibration and world-frame rig poses. This structured
representation ensures that all geometry uses consistent coordinate conventions
(LUF for camera frames, gravity-aligned world frame for voxels).
An example semi-dense SLAM point cloud used by the oracle is shown in
Appendix @fig:ase-semidense.

#figure(
  image("/figures/gt_mesh_manhattan_sample.png", width: 100%),
  caption: [Ground-truth mesh sample from ASE used for oracle RRI computation.],
) <fig:ase-mesh>

#figure(
  image("/figures/ase_efm_snippet_hist.png", width: 100%),
  caption: [Distribution of snippet counts per scene in the local ASE-ATEK snapshot.],
) <fig:ase-hist>

== Modalities and supervision

The key modalities for Aria-VIN-NBV are:

- RGB, depth, and SLAM camera streams with per-frame intrinsics and extrinsics.
- Rig trajectory and gravity, providing world-frame poses for all streams.
- Semi-dense SLAM points with per-point inverse-distance uncertainty. Points
  are provided per snippet frame and are padded/clipped to a fixed budget (50k)
  for batching; we collapse them across the snippet into a unique point set for
  oracle scoring and view conditioning, and optionally use per-point observation
  counts as a reliability cue.
- Ground-truth meshes for a subset of scenes, used to compute Chamfer-based
  reconstruction quality.
- Ground-truth 3D boxes and semantic categories (used by the entity-aware
  extension; see @sec:entity-aware).

For efficiency, meshes can be cropped to the occupancy bounds and simplified,
and the processed meshes are cached for reuse across rendering and collision
checks.

== Ground-truth mesh subset

The public ASE release includes GT meshes for a subset of scenes. In the
EFM3D/ASE release, 100 scenes include GT meshes that we use for oracle RRI
labels. Our local ASE-EFM snapshot contains these 100 scenes and 4,608
snippets, with per-scene snippet counts ranging from 8 to 152 (median 40),
summarized in @fig:ase-hist. We refer to this supervised subset as the
*ASE-EFM GT split* throughout the paper.

== Optional visibility metadata

ASE also provides per-point observation metadata (e.g., visibility tables for
semi-dense points). We treat these signals as optional accelerators or analysis
tools and do not rely on them for the oracle label computation described in
this paper.

#figure(
  image("/figures/scene-script/ase_modalities.jpg", width: 100%),
  caption: [ASE modalities and synthetic sensor streams (SceneScript dataset view).],
) <fig:ase-modalities>
