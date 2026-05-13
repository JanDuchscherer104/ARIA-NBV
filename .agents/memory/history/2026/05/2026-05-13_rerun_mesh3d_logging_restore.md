---
id: 2026-05-13_rerun_mesh3d_logging_restore
date: 2026-05-13
title: "Rerun Mesh3D Logging Restore"
status: done
topics: [rerun, inspection, mesh, dependencies]
confidence: high
canonical_updates_needed: []
files_touched:
  - .configs/rerun_offline.toml
  - aria_nbv/aria_nbv/rerun_inspector/_config.py
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/tests/rerun_inspector/test_loggers.py
artifacts:
  - .artifacts/rerun/rollout_v1_smoke.rrd
---

## Task

Restore `/world/gt/mesh` to Rerun `Mesh3D` logging after sampled-point and
wireframe diagnostics proved visually unsatisfactory for the inspector.

## Method

The Rerun SDK pin was investigated first. `rerun-sdk==0.31.4` is blocked by
`projectaria-tools==2.0.0`, and `projectaria-tools==2.1.2` pins
`rerun-sdk==0.26.2`, which conflicts with the current `pyembree`/NumPy
constraints. The implementation therefore keeps the current satisfiable
Rerun pin and logs `Mesh3D` using `albedo_factor` alpha.

## Verification

Ran targeted Rerun logger format, lint, and tests. Regenerated the rollout
smoke `.rrd` and verified `/world/gt/mesh` contains `AlbedoFactor`,
`Position3D`, `TriangleIndices`, and `Mesh3DIndicator`.

## Canonical State Impact

No state update required. The dependency upgrade remains blocked until the
Project Aria, Rerun, NumPy, and pyembree constraints are revisited together.
