## GT mesh handling proposal (Nov 28, 2025)

- Problem: `candidate_generation.py` re-crops meshes using locally defined bounds/PathConfig, bypassing the dataset-provided crop spec from `AseEfmDataset._gt_bounds_from_efm` + `load_or_process_mesh`, so downstream components can see divergent meshes and cache keys.
- Principle: compute one `MeshProcessSpec` per snippet inside `AseEfmDataset`, persist it on `EfmSnippetView` (including `spec_hash`), and treat it as the single source of truth for both Trimesh and PyTorch3D consumers.
- Proposed API additions:
  1) Add `MeshArtifact` (namedtuple/dataclass) in `mesh_cache.py` bundling `ProcessedMesh`, optional PyTorch3D `Meshes`, and `spec`.
  2) Extend `load_or_process_mesh` to optionally emit `MeshArtifact` and accept a `cache_registry` so PyTorch3D meshes are keyed by `spec.hash()`.
  3) Add helper `mesh_from_snippet(sample: EfmSnippetView, paths: PathConfig, device=None)` that returns `MeshArtifact`, building PyTorch3D mesh only when requested.
- Usage changes:
  - `AseEfmDataset.__iter__`: attach `mesh_specs=spec`, `mesh_cache_key=spec.hash()`, and optionally stash `ProcessedMesh` / `MeshArtifact` if eager caching is desired.
  - `CandidateViewGenerator.generate_from_typed_sample`: drop ad-hoc spec creation; simply reuse `sample.mesh`, `mesh_verts`, `mesh_faces`, and pass `mesh_cache_key` to `get_pytorch3d_mesh` when a P3D backend is enabled. Bounds for rules come from `sample.get_occupancy_extend()` (already derived from `_gt_bounds_from_efm`).
  - `CandidateDepthRenderer`: no change beyond using the shared helper for PyTorch3D mesh creation when tensors are missing.
- Acceptance criteria:
  - No module constructs mesh bounds manually; all consumers rely on `EfmSnippetView.crop_bounds` and `mesh_cache_key`.
  - Processed mesh files are keyed solely by `MeshProcessSpec.hash()` produced in the dataset.
  - PyTorch3D mesh caching uses the same key, so generation, collision rules, and depth rendering share identical geometry without reprocessing.
  - Existing tests pass; add regression test to ensure `CandidateViewGenerator` does not call `load_or_process_mesh` when `sample.mesh_verts` is set.
