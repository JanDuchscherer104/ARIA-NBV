---
id: 2026-01-07_rri_problem_formulation_faces_and_candidates_2026-01-07
date: 2026-01-07
title: "Rri Problem Formulation Faces And Candidates 2026 01 07"
status: legacy-imported
topics: [problem, formulation, faces, candidates, 2026]
source_legacy_path: ".codex/rri_problem_formulation_faces_and_candidates_2026-01-07.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# RRI paper notation + implementation notes (faces + candidates)

Date: 2026-01-07

## What the code actually computes

### Chamfer-style point↔mesh distance

The oracle uses PyTorch3D’s *triangle* distance kernels (no explicit sampling of surface points):

- **Accuracy (point→mesh)**: mean **squared** point-to-triangle distance over reconstruction points.
  - Implemented via `pytorch3d.loss.point_mesh_distance.point_face_distance` (returns squared Euclidean distances).
- **Completeness (mesh→point)**: mean **squared** triangle-to-point distance over *mesh faces*.
  - Implemented via `pytorch3d.loss.point_mesh_distance.face_point_distance` (returns squared Euclidean distances).
- **Bidirectional**: `accuracy + completeness`.

Implementation details:

- `oracle_rri/oracle_rri/rri_metrics/metrics.py` builds a packed `(T,3,3)` triangle tensor by repeating vertices per batch and offsetting face indices.
- The reduction uses `scatter_add_` with weights:
  - `1 / (#points)` for accuracy, and `1 / (#faces)` for completeness (i.e., **uniform face weighting**, not area-weighted).

### Oracle RRI score

`oracle_rri/oracle_rri/rri_metrics/oracle_rri.py::OracleRRI.score`:

- Crops the GT mesh to a world-frame AABB (`extend`) via `_crop_mesh_to_aabb`.
- Computes `d_before = CD(P_t, M_GT)` using semi-dense SLAM points `points_t`.
- Merges `P_t` with each candidate point cloud `P_q` to form `P_{t∪q}` and computes `d_after` batched.
- Returns `RRI(q) = (d_before - d_after) / d_before` (with `d_before` clamped to avoid divide-by-zero), plus the full accuracy/completeness breakdowns.

## Candidate set (what `Q` means in the paper)

`oracle_rri/oracle_rri/pose_generation/candidate_generation.py::CandidateViewGenerator`:

- Samples candidate *centers* around a reference pose on a spherical shell (configurable radii + azimuth/elevation caps; uniform or forward-biased sampling).
- Builds (typically roll-free) camera orientations for those centers.
- Prunes candidates with rule objects (free-space, min distance to mesh, path collision).

The resulting finite set of valid camera poses corresponds to the paper’s candidate set `Q` (Typst macro `#sym_candidates`).

## Paper changes made

- `docs/typst/shared/macros.typ`: added `#sym_faces` for the GT face set (`cal(F)_GT`).
- `docs/typst/paper/sections/03-problem-formulation.typ`:
  - Introduced `Q` explicitly (`q in Q ⊂ SE(3)`).
  - Replaced “sample points from the mesh surface” with the actual implementation: point-to-triangle + triangle-to-point squared distances over GT faces.
  - Updated the completeness term to sum over faces (`f in cal(F)_GT`) and renamed the count from `n_M` → `n_F`.

## Open suggestions (if we want to be even more explicit)

- State that the Chamfer components use **squared** Euclidean distances and are **uniformly weighted per face** (not area-weighted).
- Mention the AABB mesh crop (`extend`) used by the oracle to avoid scoring against far-away geometry.
- Clarify that `P_q` is obtained by **depth rendering the GT mesh** from pose `q` and unprojecting valid pixels (occlusions handled by the renderer).
