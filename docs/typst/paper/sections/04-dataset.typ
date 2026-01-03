= Dataset and Inputs

We use Aria Synthetic Environments (ASE), a large-scale synthetic dataset of
100,000 procedurally generated indoor scenes with realistic Aria sensor
simulations @ProjectAria-ASE-2025 @SceneScript-avetisyan2024. ASE provides
multi-camera RGB streams, semi-dense SLAM points, accurate trajectories, and
scene meshes, making it suitable for oracle RRI computation and supervised NBV
training.

Each scene is recorded as a single egocentric trajectory with synchronized RGB,
SLAM cameras, and inertial data. We follow the EFM3D data model and represent
snippet windows (typically two seconds at 10 Hz) as typed views with camera
calibration, poses, and semi-dense point clouds @EFM3D-straub2024. This
structured representation ensures that all geometry uses consistent coordinate
conventions (LUF for camera frames, gravity-aligned world frame for voxels).
An example semidense SLAM point cloud used by the oracle is shown in
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

- RGB and SLAM camera streams with per-frame intrinsics and extrinsics.
- Semi-dense SLAM points with per-point inverse-distance uncertainty.
- Ground-truth meshes for a subset of scenes, used to compute Chamfer-based
  reconstruction quality.

== Ground-truth mesh subset

The public ASE release includes GT meshes for a subset of scenes. In the
EFM3D/ASE release, 100 scenes include GT meshes that we use for oracle RRI
labels. The ATEK data store also advertises a larger GT-mesh subset (1,641
scenes), but our current offline snapshot focuses on the 100-scene GT subset
in `.data/ase_efm`. This yields 4,608 snippets (median 40 per scene) with
mesh supervision, as summarized in @fig:ase-hist. We call this subset
the *ASE-EFM GT split* throughout the paper.

We treat EVL outputs as a frozen scene representation. EVL produces voxelized
occupancy probabilities, centerness, free-space cues, and OBB predictions in a
local grid centered near the last rig pose. These features provide strong
inductive bias for RRI prediction, but their local extent motivates additional
view-dependent cues derived from semidense projections.

#figure(
  image("/figures/scene-script/ase_modalities.jpg", width: 100%),
  caption: [ASE modalities and synthetic sensor streams (SceneScript dataset view).],
) <fig:ase-modalities>
