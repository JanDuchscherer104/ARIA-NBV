# PointNeXt encoder review

## Findings
- Eager imports in `pointnext_encoder.py` triggered `openpoints.models.__init__`, which imports reconstruction modules and requires the optional `chamfer` extension. This caused `nbv-st` startup to fail.
- Helper functions `_resolve_pointnext_root` and `_ensure_pointnext_on_path` were missing, leading to immediate runtime errors when constructing the encoder.

## Changes
- Added `_resolve_pointnext_root`, `_ensure_pointnext_on_path`, and an `openpoints.models` stub to bypass the heavy package `__init__`.
- Moved OpenPoints imports into `PointNeXtSEncoder.__init__` with dependency guard.
- Ensured CUDA extension paths are added to `sys.path` when using the encoder.

## Suggestions
- Consider adding a lightweight unit test that imports `PointNeXtSEncoder` and instantiates it with a minimal config to ensure optional deps don’t break `nbv-st` startup.

## Follow-up
- Removed dependency guards as requested; this reintroduced the hard dependency on `openpoints.cpp.chamfer_dist` via `openpoints.models.__init__`.
- Tests now fail unless chamfer is built. Use `OPENPOINTS_BUILD_CHAMFER_DIST=1 uv sync --all-extras` to compile it, or re-enable the guards if you want PointNeXt to remain optional.
