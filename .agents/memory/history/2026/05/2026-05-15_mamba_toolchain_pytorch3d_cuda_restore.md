---
id: 2026-05-15_mamba_toolchain_pytorch3d_cuda_restore
date: 2026-05-15
title: "Mamba Toolchain And PyTorch3D CUDA Restore"
status: done
topics: [environment, cuda, pytorch3d, streamlit, setup]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/GOTCHAS.md
files_touched:
  - .vscode/settings.json
  - .vscode/launch.json
  - SETUP.md
  - docs/contents/setup.qmd
  - aria_nbv/aria_nbv/app/panels/counterfactual_rollouts.py
  - aria_nbv/tests/app/panels/test_counterfactual_rollouts_panel.py
  - .agents/memory/state/GOTCHAS.md
---

## Task

Made the existing `aria-nbv` mamba environment the default CUDA/toolchain
context while preserving the repo `.venv` as the `uv` runtime. Restored local
PyTorch3D CUDA support and re-enabled the counterfactual rollout page CUDA
option behind a real rasterization smoke check.

## Method

VS Code settings and launch configs now inject the mamba toolchain path,
`CONDA_PREFIX`, `CUDA_HOME`, `FORCE_CUDA=1`, and
`TORCH_CUDA_ARCH_LIST=8.6`. Setup docs now distinguish Torch CUDA visibility
from PyTorch3D CUDA rasterization and include the repair command.

PyTorch3D was rebuilt under:

```sh
mamba activate aria-nbv
export CUDA_HOME="$CONDA_PREFIX" FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6
uv sync --reinstall-package pytorch3d --no-build-isolation-package pytorch3d --no-cache
uv sync --all-extras
```

The public repair docs use the safer single command
`uv sync --all-extras --reinstall-package pytorch3d --no-build-isolation-package pytorch3d --no-cache`;
the separate `uv sync --all-extras` above restored extras after the initial
plain repair command pruned optional/dev packages.

## Verification

- Strict JSON validation passed for `.vscode/settings.json` and
  `.vscode/launch.json`.
- PyTorch3D CUDA rasterization smoke returned
  `pytorch3d_cuda_rasterization_ok torch.Size([1, 8, 8, 1]) cuda:0`.
- The counterfactual rollout page helper returned `["cpu", "cuda"]`.
- A real sample-0 target-RRI rollout completed on CUDA:
  `cuda_live_target_rri_ok 81286 0 target_rri 1 0.07777952402830124`.
- `cd aria_nbv && uv run pytest tests/app/panels/test_counterfactual_rollouts_panel.py -q`
  passed.

## Canonical State Impact

`.agents/memory/state/GOTCHAS.md` now records that the earlier CPU-only
PyTorch3D state was repaired on 2026-05-15 and that CUDA extension rebuilds
must run from the activated mamba toolchain.
