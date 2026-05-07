---
id: 2026-05-07_rerun_rollout_native_camera_context
date: 2026-05-07
title: "Rerun Rollout Native Camera Context"
status: done
topics: [rerun, rollout-zarr, diagnostics, package]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rerun_inspector/_rollout_zarr.py
  - aria_nbv/aria_nbv/rerun_inspector/_loggers.py
  - aria_nbv/aria_nbv/rerun_inspector/_config.py
  - aria_nbv/aria_nbv/rerun_inspector/_cli.py
  - aria_nbv/tests/rerun_inspector/test_rollout_zarr_logger.py
  - aria_nbv/tests/rerun_inspector/test_rerun_cli.py
  - docs/contents/impl/rerun_offline_inspector.qmd
  - docs/contents/impl/one_scene_smoke.qmd
  - .agents/skills/rerun-nbv-inspector/SKILL.md
  - .agents/skills/rerun-nbv-inspector/references/nbv-inspector-contract.md
  - .agents/skills/rerun-nbv-inspector/references/rerun-python-patterns.md
artifacts:
  - .artifacts/rerun/rollout_multistep_native_cameras_2026-05-07.rrd
  - .artifacts/rerun/rollout_multistep_native_cameras_auto_2026-05-07.rrd
---

Implemented rollout-Zarr inspection as an extension of the normal Rerun offline
sample inspector. Rollout mode now has `auto|required|off` VIN-context policy,
uses normal static sample modalities when lineage resolves, skips synthetic
stores quickly in auto mode, and records fallback context warnings in rollout
metadata.

Rollout candidate frusta moved from aggregate `LineStrips3D` to native
`Transform3D` + `Pinhole` camera entities under
`world/rollout/step/{selected,valid,invalid}/candidate_###/camera`. Verbose
candidate diagnostics are attached via `AnyValues` rather than visible text
labels. A default blueprint separates `world/**`, `plots/rollout/**`, and
`metadata/**` views.

Verification included Rerun inspector tests, ruff checks, Quarto renders, the
Rerun inspector skill validator, `make check-agent-memory`, and saved `.rrd`
smoke artifacts inspected with `rerun rrd print`. No canonical state files need
updates; the durable workflow delta is captured in the Rerun inspector skill
and references.
