# OpenPoints uv shim

## Changes
- Added a local `openpoints-shim` package that builds OpenPoints ops during `uv sync`.
- The shim **does not modify** `external/PointNeXt`; it runs build steps directly on the submodule sources.
- `sitecustomize.py` bootstraps `PointNeXt` into `sys.path` so `import openpoints` works automatically.
- Updated `oracle_rri/pyproject.toml` dependencies and uv build deps for minimal OpenPoints runtime needs.
- Updated `docs/contents/setup.qmd` to document the uv-driven OpenPoints build and env toggles.
- Fixed build gating so missing CUDA ops (including chamfer) are rebuilt even if pointnet2 was already built.
- Added chamfer/emd/subsampling directories to `sys.path` so top-level CUDA extensions import correctly.

## Defaults
- `OPENPOINTS_BUILD_CHAMFER_DIST=1` by default (required for OpenPoints imports).

## Potential issues
- The shim expects the `external/PointNeXt` submodule to exist; if missing, it runs `git submodule update --init --recursive`.
- PointOps/other ops require env toggles.
- Build depends on `nvcc` discovery; if `pyvenv.cfg` points to a different base, CUDA detection may fail.

## Suggestions
- Add a small unit test that imports `openpoints` and checks `pointnet2_batch_cuda` is importable.
- Consider a configurable `OPENPOINTS_POINTNEXT_ROOT` override in docs for non-standard repo layouts.
