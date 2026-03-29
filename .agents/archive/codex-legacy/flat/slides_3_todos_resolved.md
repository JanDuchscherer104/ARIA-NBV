## Task: Resolve remaining TODOs in `docs/typst/slides/slides_3.typ`

### Outcome
- The slide deck now compiles cleanly (`typst compile --root docs docs/typst/slides/slides_3.typ docs/typst/slides/slides_3.pdf`).
- All inline TODO markers and placeholder comments in `slides_3.typ` are removed (the “Next Steps” slide remains as intentional content, but no longer contains “TODO”).

### What was added/changed (high level)
- Added a theory + implementation breakdown of `oracle_rri/oracle_rri/rri_metrics/metrics.py::chamfer_point_mesh_batched`:
  - Inputs/outputs and directionality (accuracy vs. completeness).
  - Formulas for point→mesh and mesh→point components (squared distances).
  - Vectorised batching strategy (packed points, repeated mesh, scatter reductions).
- Added conceptual slides for the remaining stages of the oracle pipeline, following
  `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py::OracleRriLabeler.run`:
  - Pipeline stage graph + key tensors/shapes.
  - Candidate orientation (“look-at” frames + jitter).
  - Candidate pruning rules (shell sampling / min distance / path collision / free-space AABB).
  - Candidate depth rendering (PyTorch3D z-buffer path + validity mask).
  - Backprojection (pinhole unprojection + vectorised `PerspectiveCameras.unproject_points`).
  - Fusion step (`P_t ∪ P_q^i`) + shared AABB propagation.
- Replaced a previously broken/duplicated slide block (unclosed Typst constructs) in the
  rule section; this was the main cause of compilation failures.

### External references (clickable in slides)
- PyTorch3D point↔mesh distance kernels:
  - `pytorch3d/loss/point_mesh_distance.py` (`point_face_distance`, `face_point_distance`, `_DEFAULT_MIN_TRIANGLE_AREA`)
  - `pytorch3d/renderer/mesh/rasterizer.py` (`MeshRasterizer`)
  - `pytorch3d/renderer/cameras.py` (`PerspectiveCameras`, `unproject_points`)
- EFM3D tensor wrappers:
  - `efm3d/aria/pose.py` (`PoseTW`)
  - `efm3d/aria/camera.py` (`CameraTW`)
- `PowerSpherical` (used for forward-biased direction sampling) via its GitHub repo.

### Notes / follow-ups (not implemented here)
- Some GitHub links point to `main` and line anchors; if upstream changes, anchors may drift.
  If this becomes an issue, consider linking to a specific commit SHA instead.
