---
id: 2026-05-19_short_ase_atek_identifiers
date: 2026-05-19
title: "Short ASE-ATEK Identifiers"
status: done
topics: [data-handling, offline-store, rerun, rollouts, cli]
confidence: high
canonical_updates_needed: []
---

## Task

Normalize ASE/ATEK sample identifiers so public and future persisted identifiers
use `ASE_<scene>_Atek_<sample>` instead of the long ATEK
`AriaSyntheticEnvironment_<scene>_AtekDataSample_<sample>` form.

## Method

Added shared compact/raw identifier helpers in the data-handling layer and
wired them through raw EFM snippet inference, VIN offline sample preparation,
offline-store lookup compatibility, offline-info CLI payloads, rollout-store
dictionaries/manifests, and Rerun metadata. Existing stores remain readable; old
raw identifiers are compacted at output boundaries and accepted by lookup paths.

## Verification

- `cd aria_nbv && uv run pytest tests/data_handling/test_dataset.py tests/data_handling/test_offline_info_cli.py tests/data_handling/test_vin_offline_store.py tests/rerun_inspector tests/rollouts/test_info_cli.py -q`
- `cd aria_nbv && uv run ruff check aria_nbv/data_handling aria_nbv/rerun_inspector aria_nbv/rollouts tests/data_handling tests/rerun_inspector tests/rollouts`
- `make offline-samples` output was smoke-checked to contain `ASE_` and no
  `AriaSyntheticEnvironment`.

## Canonical State Impact

No canonical state update is required. This is an implementation contract update:
future generated stores should persist compact ASE-ATEK identifiers, while raw
ATEK keys remain private compatibility tokens for shard lookup.
