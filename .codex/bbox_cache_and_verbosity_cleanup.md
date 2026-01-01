Task: Cache cropped scene bounds on `EfmSnippetView`, centralise verbosity, and keep debug per module.

Changes:
- Added `crop_bounds` (min/max AABB) to `EfmSnippetView`, moved during `.to()`, and included in repr.
- AseEfmDataset now returns `(mesh, crop_bounds)` from `_prepare_mesh`; stores padded bounds (or mesh bounds fallback) on each snippet; also records bounds when no mesh is loaded but semidense bounds exist. Removed manual verbosity checks—Console now governs output.
- CandidateViewGenerator uses `sample.crop_bounds` first for occupancy extent to avoid recomputation.
- Cleaned verbosity gating patterns per new Verbosity-aware Console; dashboard now has a global verbosity selector while debug toggles are per module (data/candidates/renderer).

Tests:
- `ruff check` on touched files.
- `pytest oracle_rri/tests/pose_generation/test_pose_generation.py -q`
- `pytest oracle_rri/tests/test_console.py -q` (earlier in session).
