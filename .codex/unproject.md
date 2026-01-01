## Unprojection helpers (2025-12-01)

- Implemented `oracle_rri/rendering/unproject.py` with `backproject_depth` and `backproject_batch` to convert rendered candidate depths into world-frame point clouds using `CameraTW.unproject` (handles distortion-aware rays) and explicit cam→world poses.
- Added stride, zfar, and max_points controls; masks invalid rays (non-finite depth, outside valid radius, non-positive z).
- Added regression tests `tests/rendering/test_unproject.py` for pinhole geometry correctness and zfar filtering.
- Implemented `_filter_rendered_depths` in `candidate_depth_renderer.py` (sorts candidates by valid hit count and is applied inside `render`).
- Updated `rendering/plotting.py::add_depth_hits` to reuse `backproject_depth` (distortion-aware) while keeping candidate sub-selection.
- `dashboard/panels.py::render_depth_page` continues to pass selected candidate indices; now hits use the shared unprojection logic.
- Simplified `Pytorch3DDepthRenderer`: now uses `fragments.zbuf` directly (no manual barycentric recompute), added CPU unit test `tests/rendering/test_pytorch3d_depth_renderer.py` to ensure hits on a front-facing plane.
- Updated `rendering/unproject.py` to use pinhole intrinsics (fx, fy, cx, cy) for backprojection to match the PyTorch3D render path; distortion is ignored to stay consistent with rendering.
- Notes:
  - Existing plotting backprojection should switch to these helpers to avoid drift.
  - Depth renders still rely on PyTorch3D extrinsics; keep verifying world←cam vs world→cam usage when touching the renderer.
