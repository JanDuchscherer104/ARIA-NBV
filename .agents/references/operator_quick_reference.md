# Operator Quick Reference

Use this file for practical operator aids that do not belong in canonical project state.

## Environment Recovery
- Preferred interpreter: `aria_nbv/.venv/bin/python`
- If the venv is missing or stale, rebuild it with:

```bash
UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync --extra dev --extra notebook --extra pytorch3d
```

- Verify the interpreter before diagnosing dependency issues:

```bash
aria_nbv/.venv/bin/python --version
uv run python --version
```

## Repo Hygiene
Run these before staging or when the worktree looks noisy:

```bash
git status -sb
git diff --stat
git diff --name-only
```

Workflow:
- Classify untracked files as keep, ignore, or delete before staging.
- Add ignores for logs, renders, caches, and other generated artifacts instead of committing them.
- Stage by intent so code, docs, and assets remain reviewable as separate changes.
- Do not revert unrelated worktree changes unless the user explicitly asks.

## Frame and Key Conventions
- Frame hierarchy: world -> rig/device -> camera; use `PoseTW` for poses and `CameraTW` for cameras.
- Transform notation: `T_A_B` means “transform from frame B to frame A.”
- ATEK key prefixes:
  - `mtd`: motion trajectory data
  - `mfcd`: multi-frame camera data
  - `msdpd`: multi-semidense-point data

## EFM Snippet View Quick Reference
- `camera_rgb`, `camera_slam_left`, `camera_slam_right` -> `EfmCameraView`
- `trajectory` -> `EfmTrajectoryView`
- `semidense` -> `EfmPointsView`
- `obbs` -> `EfmObbView` or `None`
- `gt` -> `EfmGTView`
- `mesh` / `has_mesh` -> optional ground-truth mesh
- Use `.to(...)` on the snippet or its sub-views to move tensors without cloning when possible.
