# xFormers install notes (oracle_rri)

## Context

- `external/efm3d` imports `xformers.ops` opportunistically in `efm3d/model/dinov2_utils.py`; if unavailable, EFM3D
  falls back (with warnings) and some features like nested tensors become unavailable.
- `oracle_rri/pyproject.toml` pins `torch==2.4.1` and uses the PyTorch CUDA 12.1 wheel index.

## What works for our current pinned Torch

- Installing `xformers` **unversioned** from `https://download.pytorch.org/whl/cu121` wants to upgrade Torch to 2.5.1
  (because it resolves to `xformers==0.0.29.post1`, which depends on newer Torch).
- Pinning `xformers==0.0.28.post1` resolves cleanly **without changing Torch**.
- `xformers==0.0.28.post1` wheels on the CUDA 12.1 index are **Linux-only** (`manylinux_2_28_x86_64`); there is no
  `win_amd64` wheel for that build. To keep Windows installs viable, this dependency must be gated by
  `sys_platform == "linux"`.

## Implementation in this repo

- Added an optional extra `xformers` (Linux-only, pinned) and mapped `xformers` to the existing `pytorch-cu121` index in
  `oracle_rri/pyproject.toml`.
- Documented install in `docs/contents/setup.qmd`.

## Usage

```bash
cd oracle_rri
uv sync --extra xformers
```

Or when installing into an existing interpreter:

```bash
cd oracle_rri
uv pip install -e ".[xformers]"
```

## Future (Windows / newer CUDA backends)

If we want `xformers` on Windows (or newer CUDA toolkits like cu126/cu128/cu129), we likely need to upgrade our pinned
Torch and switch the PyTorch wheel index accordingly. Follow the official xFormers install docs:
https://github.com/facebookresearch/xformers#installing-xformers

