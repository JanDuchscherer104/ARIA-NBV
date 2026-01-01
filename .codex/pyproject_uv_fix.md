# Pyproject uv fix (2025-11-24)

- Declared `efm3d` as a runtime dependency and mapped it to the local editable source via `[tool.uv.sources] efm3d = { path = "../external/efm3d", editable = true }` so `uv sync` pulls the repo without manual `uv pip install -e`.
- Torch/cu121 index wiring left intact; next step is to run `UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync --extra dev --extra pytorch3d` and a quick smoke test.
- Ran `UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync --extra dev --extra pytorch3d`; `torch==2.4.1+cu121` pulled from `pytorch-cu121` and `efm3d` installed editable. Smoke test `uv run pytest -q tests/test_console.py` passed (2 tests).
- Added missing runtime dep `pytransform3d==3.14.3` (needed by `oracle_rri.utils.frames` and streamlit app); synced successfully.
- Rebuilt `pytorch3d` from source against torch 2.4.1+cu121 using `UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync --extra dev --extra notebook --extra pytorch3d --reinstall-package pytorch3d --no-binary-package pytorch3d`; import of `pytorch3d._C` now succeeds.

# Pyproject uv fix (2025-11-23)

- Updated `oracle_rri/pyproject.toml` to remove the invalid `[tool.uv].torch-backend` key, move `extra-build-dependencies` into dict form, and add a CUDA 12.1 PyTorch index/source map.
- Synced environment with `UV_TORCH_BACKEND=cu121 uv sync --extra dev --extra notebook --extra pytorch3d` from `oracle_rri/`.
- Rebuilt PyTorch3D 0.7.8 against torch 2.4.1+cu121 using the CUDA toolkit at `/home/jandu/miniforge3/envs/aria-nbv` (`FORCE_CUDA=1`, `CUDA_HOME` + PATH set). Verified `pytorch3d.ops.knn_points` on CUDA.
- Test run `uv run pytest -q` currently fails during collection because `tomlkit` and `efm3d` are missing from the environment (pre-existing dependency gaps). No code changes were made to address those deps.
