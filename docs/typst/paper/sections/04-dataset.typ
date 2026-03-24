= Dataset and Inputs <sec:dataset>

#import "../../shared/macros.typ": *

We use #ASE_full (#ASE), a large-scale synthetic dataset of
100,000 procedurally generated indoor scenes with realistic Aria sensor
simulations @ProjectAria-ASE-2025 @SceneScript-avetisyan2024.

== ASE: synthetic egocentric trajectories

ASE scenes are recorded as single egocentric walkthrough trajectories with
synchronized RGB + SLAM camera streams, inertial signals, and SLAM products
(trajectory and semi-dense points) in a gravity-aligned world frame
@projectaria-engel2023. The native dataset release also includes dense
per-frame supervision (e.g., depth maps, instance masks, and scene language),
which we primarily use for visualization and qualitative debugging.

== ATEK-EFM variant: WebDataset snippets (EFM3D / EVL format)

We consume ASE through the ATEK Data Store WebDataset export in the EFM3D/EVL
training format @EFM3D-straub2024 @ATEK-DataStore-2025. In this representation,
each sample is a fixed-length snippet of 20 frames at 10 Hz (2 s), with
calibration and rig poses attached per frame, and a stride of 10 frames (1 s)
between consecutive snippets. The standard preprocessing
resizes RGB frames to 240 #sym.times 240 and SLAM grayscale frames to roughly
320 #sym.times 240 while preserving geometry (intrinsics/extrinsics) and
coordinate conventions (LUF camera frames; gravity-aligned world frame for
rig/voxel quantities).

An example snippet with synchronized RGB/SLAM frames is shown in
@fig:ase-snippet-overview.

#figure(
  image("/figures/app-paper/data_frames_81022_11.png", width: 100%),
  caption: [ASE snippet overview: synchronized RGB/SLAM frames.],
) <fig:ase-snippet-overview>

== Modalities and supervision

The key modalities for Aria-VIN-NBV are:

- RGB and SLAM camera streams with per-frame intrinsics and extrinsics.
- Optional per-frame ray distance / depth supervision in the EFM schema.
- Rig trajectory and gravity, providing world-frame poses for all streams
  (trajectory notation: #symb.ase.traj).
- Semi-dense SLAM points (#symb.ase.points_semi) with per-point inverse-distance
  uncertainty (#symb.vin.inv_dist_std) and optional observation counts
  (#symb.vin.n_obs). Points
  are provided per snippet frame and are padded/clipped to a fixed budget (50k)
  for batching; we collapse them across the snippet by concatenating valid
  points, deduplicating when observation counts are requested, and (optionally)
  subsampling to a fixed maximum for projection features. Observation counts
  (track length) are appended when enabled and serve as a reliability cue.
- Ground-truth meshes for a subset of scenes, used to compute Chamfer-based
  reconstruction quality.
- Ground-truth 3D boxes and semantic categories (used by the entity-aware
  extension; see @sec:entity-aware).

Any mesh preprocessing that is specific to oracle labeling (e.g., optional
snippet-bound cropping, simplification, and caching) is described in the oracle
section @sec:oracle-rri to avoid repeating implementation details here.


== Ground-truth mesh subset

The public ASE release includes GT meshes for a subset of scenes. In the
EFM3D/ASE release, 100 scenes include GT meshes that we use for oracle RRI
labels @EFM3D-straub2024. The corresponding ATEK-EFM export provides 4,608
snippet windows. For efficient iteration, we build an offline dataset of oracle
labels for 883 snippets from 80 of the 100 mesh scenes and use an 80/20
train/val split (706/177).
// <rm>
// Hard-coded counts that drift easily. Prefer importing these from the offline-cache stats
// artifact (as done in slides/appendix) and report them once in Evaluation as a table.
// </rm>

== Optional visibility metadata

ASE also provides per-point observation metadata (e.g., visibility tables for
semi-dense points). We treat these signals as optional accelerators or analysis
tools and do not rely on them for the oracle label computation described in
this paper.
