---
name: dataset-cache-ops
description: Use when operating or documenting ARIA-NBV ASE downloads, ATEK shards, meshes, immutable VIN offline stores, split manifests, storage estimates, and data smoke checks.
---

# Dataset Cache Ops

## When To Use

Use this skill for:

- ASE downloader and data-root setup
- ATEK shards, mesh availability, and snippet coverage checks
- immutable VIN offline stores, manifests, split files, and storage estimates
- `offline_only.toml`, VIN diagnostics, and data smoke commands

Do not use it to restore legacy cache migration or removed training APIs.

## Read First

1. `aria_nbv/AGENTS.md`
2. `aria_nbv/aria_nbv/data_handling/AGENTS.md`
3. `README.md`
4. `.agents/references/operator_quick_reference.md`
5. `.agents/memory/state/GOTCHAS.md`

## Rules

- The canonical training path is `VinOfflineDataset` configured through
  `VinOfflineSourceConfig` with `kind = "offline"`.
- Rebuild available immutable stores with `VinOfflineWriter`; do not revive
  removed legacy cache datasets, providers, or migration tools.
- Keep split manifests and sample indexes file-backed and inspectable.
- Prefer list/smoke commands before expensive downloads or builds.
- Record exact failing commands and traceback summaries when data checks fail.

## Verification

- `cd aria_nbv && uv run nbv-downloader -m list`
- `cd aria_nbv && uv run nbv-summary --config-path ../.configs/offline_only.toml`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py`
- `cd aria_nbv && uv run pytest tests/vin/test_vin_utils.py`
