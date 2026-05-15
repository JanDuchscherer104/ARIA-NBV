---
id: 2026-05-13_rerun_sdk_and_dependency_upgrade_probe
date: 2026-05-13
title: "Rerun SDK and Dependency Upgrade Probe"
status: done
topics: [dependencies, rerun, projectaria, rollouts]
confidence: high
canonical_updates_needed: []
---

## Task
Probe the highest compatible Rerun/Project Aria dependency set on 2026-05-13 and keep the rollout/Rerun smoke paths working.

## Method
Used PyPI metadata plus `uv add`/`uv lock --upgrade` to move dependencies incrementally, testing after each accepted change. Direct `rerun-sdk==0.32.0` remained impossible because `projectaria-tools==2.1.2` pins `rerun-sdk==0.26.2`; PyTorch/xFormers latest also remained blocked by the repo's CUDA 12.1/optional-extra constraints.

## Findings
`aria_nbv/pyproject.toml` now uses `projectaria-tools==2.1.2`, `rerun-sdk==0.26.2`, `trimesh==4.12.2`, NumPy 2 via the lock, and removes hard `pyembree` because the only Python 3.11-compatible `pyembree` release requires NumPy < 2. Rerun scalar logging was migrated from `rr.Scalar` to `rr.Scalars`. Candidate geometry now falls back to CPU when a rebuilt PyTorch3D extension lacks CUDA support, and pyembree availability is tested by import rather than `find_spec`.

Rerun 0.26 also exposed a blueprint selection issue: recordings store data under absolute entity paths such as `/world/gt/mesh`, so the inspector blueprint must use `/world`, `/plots/rollout`, and `/metadata` origins/contents. Relative blueprint paths can load the recording but show an empty 3D view.

Blocked upgrade probes:
- `rerun-sdk==0.32.0` conflicts with `projectaria-tools==2.1.2`.
- `torch==2.11.0` / `torchvision==0.26.0` / `torchaudio==2.11.0` with `xformers==0.0.35` conflicts with the repo's `xformers==0.0.28.post1` extra, and latest xFormers did not resolve for the supported Python/platform split.
- `torchmetrics==1.9.0` conflicts with `projectaria-atek==1.0.0`, which is still the latest PyPI release and pins `torchmetrics==0.10.1`; `setuptools<81` keeps `pkg_resources` importable for that old torchmetrics line.

## Verification
Passed:
- `cd aria_nbv && uv run pytest tests/rerun_inspector -q`
- `cd aria_nbv && uv run pytest tests/pose_generation/test_pose_generation.py tests/pose_generation/test_counterfactuals.py -q`
- `cd aria_nbv && uv run pytest tests/rollouts tests/data_handling/test_target_selection.py tests/app/panels -q`
- `cd aria_nbv && uv run nbv-build-rollouts --config-path ../.configs/build_rollouts_v1_smoke.toml --dry-run`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.data/offline_cache/rollouts_v1_smoke.zarr --rollout-index 0 --rollout-context required --save ../.artifacts/rerun/rollout_v1_rerun_0_26_2.rrd`
- `cd aria_nbv && uv run rerun rrd print ../.artifacts/rerun/rollout_v1_rerun_0_26_2.rrd | rg 'world/gt/mesh|Mesh3D|AlbedoFactor|TriangleIndices|Position3D|Scalars'`
- `cd aria_nbv && uv run pytest tests/rerun_inspector/test_loggers.py tests/rerun_inspector/test_rollout_zarr_logger.py -q`
- `cd aria_nbv && uv run nbv-rerun-inspect --config-path ../.configs/rerun_offline.toml --rollout-store ../.data/offline_cache/rollouts_v1_smoke.zarr --rollout-index 0 --rollout-context required --save ../.artifacts/rerun/rollout_v1_rerun_0_26_2_blueprint_fixed.rrd`

## Canonical State Impact
None. This is a dependency compatibility update, not a thesis-direction change.
