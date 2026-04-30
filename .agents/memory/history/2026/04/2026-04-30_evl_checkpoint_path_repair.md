---
id: 2026-04-30_evl_checkpoint_path_repair
date: 2026-04-30
title: "EVL Checkpoint Path Repair"
status: done
topics: [vin, evl, checkpoints, setup]
confidence: high
canonical_updates_needed: []
---

## Task

Repair the failed `nbv-build-offline` startup where EFM Hydra tried to load
`/home/jandu/repos/NBV/.logs/ckpts/dinov2_vitb14_reg4_pretrain.pth`.

## Method

- Verified the DINOv2 checkpoint already existed at
  `.logs/ckpts/dinov2_vitb14_reg4_pretrain.pth`.
- Verified the current public Meta download URL returns `HTTP 200` and the
  local file size matches the advertised `content-length`.
- Rebased active EVL Hydra config paths away from stale `/home/jandu/repos/NBV`
  values.
- Added EVL adapter path normalization so stale absolute `.logs` and
  `external` asset paths are repaired against the current checkout before
  Hydra instantiation.
- Updated `docs/contents/setup.qmd` to use the live DINOv2 URL instead of the
  dead Hugging Face revision.

## Verification

- `cd aria_nbv && uv run ruff check aria_nbv/vin/backbone_evl.py tests/vin/test_backbone_evl.py`
- `cd aria_nbv && uv run pytest tests/vin/test_backbone_evl.py`
- `cd aria_nbv && uv run python - <<'PY' ... EvlBackboneConfig(...).setup_target() ... PY`

The bounded model-load check instantiated EVL successfully on CUDA and resolved
the DINOv2 checkpoint and taxonomy file under `/home/jd/repos/ARIA-NBV`.
