## RRI Metrics Submodule Outline (2025-12-08)

- Goal: Implement oracle RRI based on point↔mesh distances (accuracy/completeness) and Chamfer-style aggregation, built for GPU-first torch/PyTorch3D with optional CPU fallbacks (ATEK/efm3d).
- Proposed package layout under `oracle_rri/rri_metrics/`:
  - `types.py`: Typed dataclasses (`RriInputs`, `RriComponents`, `RriResult`, `DistanceBreakdown`) and enums for metric variants/reductions; shapes + units documented.
  - `sampling.py`: GT mesh sampling utilities (PyTorch3D `sample_points_from_meshes`), deterministic seeds, face-area weighting, optional mask/crop to bbox; helpers to compute crop from union of `P_t`/`P_q`.
  - `pointcloud_ops.py`: Voxel/grid downsampling (torch/open3d), merging `P_t`+`P_q`, filtering invalid/NaN, optional subsample budgets.
  - `distances.py`: Low-level primitives wrapping PyTorch3D `point_mesh_face_distance`, `point_mesh_edge_distance`, and `chamfer_distance`; ability to return directional components, witness face/point indices when needed; fallback adapters to ATEK `compute_pts_to_mesh_dist`.
  - `metrics.py`: User-facing functions to compute accuracy, completeness, symmetric CD, F-score@τ, plus helpers to normalise by baseline CD for RRI numerator/denominator; pure torch API.
  - `oracle_rri.py`: Config-as-factory orchestrator (`OracleRRIConfig`, `OracleRRI`) that takes `EfmSnippetView`, `CandidateDepths` or fused PCs, runs cropping/downsampling → distances → RRI scores; emits `RriResult` with per-candidate scalars and optional per-point diagnostics.
  - `tests/` (new): Unit tests for distance correctness vs trimesh ground-truth, deterministic sampling, RRI numerics on toy scenes; GPU/CPU variants.
- External choices:
  - Prefer PyTorch3D `point_mesh_face_distance` for exact point↔triangle distances on GPU (gives both directions) and `chamfer_distance` for PC↔PC variants; use efm3d/atek CPU functions only as validation/fallback.
  - Reuse existing rendering/backprojection (`rendering.unproject.backproject_batch`) outputs for candidate PCs; rely on `PoseTW`/`CameraTW`.
- Open questions: how to expose witnesses (need face indices -> barycentric post-pass); batch memory strategy for large meshes (chunked point batches vs mesh decimation); threshold set for F-score defaults.

