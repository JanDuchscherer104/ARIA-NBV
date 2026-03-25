---
id: 2025-12-31_offline_cache_cleanup_2025-12-31
date: 2025-12-31
title: "Offline Cache Cleanup 2025 12 31"
status: legacy-imported
topics: [offline, cache, cleanup, 2025, 12]
source_legacy_path: ".codex/offline_cache_cleanup_2025-12-31.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Offline cache cleanup (2025-12-31)

## What changed
- Removed duplicated cache I/O helpers from `oracle_rri/oracle_rri/data/offline_cache.py` so it now delegates naming + metadata helpers to `offline_cache_store.py` (single source of truth).
- Verified cache filenames are now based on `ASE_NBV_SNIPPET_<scene>_<snippet_token>_<config_hash>.pt` (snippet token extracted as numeric suffix).
- Confirmed offline cache uses `PathConfig`-backed paths via `OracleRriCacheConfig.cache_dir` resolution.

## Tests / runs
- `uv run nbv-cache-samples -n 1 --overwrite` (writes cache successfully; note unique suffix added because prior sample with same base key already existed).
- `uv run nbv-summary` (loads cache + runs VIN summary successfully).
- `uv run pytest oracle_rri/data/offline_cache.py -q` → no tests collected (exit code 5).
- `uv run pytest tests/test_efm_dataset.py -q` → 1 failure in `test_batching_supported` due to invalid `verbose` arg (likely test drift; not related to offline cache).

## Findings / suggestions
- Consider purging `cache/samples` when `overwrite=True` to avoid suffixes like `__cc86fb` and to keep filenames clean.
- Update `tests/test_efm_dataset.py` to use `verbosity=Verbosity.QUIET` instead of `verbose=False` if you want the integration test to pass.
- Optional: consider adding a `weights_only` toggle to cached `torch.load` to address the FutureWarning about pickle deserialization.

## Num_workers + offline cache (2025-12-31)

### Changes
- `oracle_rri/oracle_rri/data/offline_cache.py`: added worker-aware map_location resolution (CUDA → CPU in workers) and pickling-safe Console reinit via `__getstate__/__setstate__`.
- `oracle_rri/oracle_rri/lightning/lit_module.py`: move cached PoseTW/reference poses onto the module device before VIN forward.
- `oracle_rri/oracle_rri/lightning/lit_module.py`: confusion matrix / label histogram logging now moves tensors to CPU before matplotlib.

### Validation
- `uv run nbv-summary --datamodule-config.num-workers 2` succeeded using cached samples.
- Generated 2 cached samples: `uv run nbv-cache-samples -n 2 --overwrite`.
- Short training (2 steps) with offline cache + `num_workers=2` succeeded:
  `uv run nbv-train --datamodule-config.num-workers 2 --datamodule-config.train-cache.cache.cache-dir /home/jandu/repos/NBV/.data/oracle_rri_cache --datamodule-config.val-cache.cache.cache-dir /home/jandu/repos/NBV/.data/oracle_rri_cache --datamodule-config.train-cache.map-location cpu --datamodule-config.val-cache.map-location cpu --trainer-config.max-epochs 1 --trainer-config.limit-train-batches 2 --trainer-config.limit-val-batches 0 --no-trainer-config.enable-model-summary --no-trainer-config.use-wandb`.

### Notes
- `nbv-cache-samples -n 2` takes ~2 minutes; earlier run timed out at 120s.
- `torch.load` FutureWarning remains (weights_only=False). Consider adding a config flag if you want to silence it.
- Removed unused `rename_cache_samples` helper from `oracle_rri/oracle_rri/data/offline_cache.py` (no call sites).
