# Datamodule Streamline - Agent B Notes (2026-01-04)

## Findings
- CLI overrides cannot set `datamodule_config.batch_size=None`; for online datasets the config must omit `batch_size` entirely.
- Switching `datamodule_config.source.kind` from offline->online while using `offline_only.toml` triggers extra-field validation errors (offline-only fields like `source.cache`, `train_split`, `val_split`). Use a dedicated online config or strip those keys.
- `vin_snippet_cache` does not fully cover all offline cache entries; missing snippet example: scene=81387 snippet=000019. Current fallback in `OracleRriCacheDataset` uses empty `VinSnippetView` when both cache + EFM load fail. Consider regenerating or verifying cache coverage to avoid silent empties.
- Validation logging is not shown in rich progress output; switch to tqdm if you want explicit val progress output.

## Commands Used (for reference)
- Offline cache train+val (tqdm to show val loop):
  `uv run nbv-train --config-path /home/jandu/repos/NBV/.configs/offline_only.toml --datamodule-config.source.kind offline_cache --datamodule-config.source.cache.limit 1 --datamodule-config.num-workers 0 --datamodule-config.batch-size 1 --no-trainer-config.fast-dev-run --trainer-config.max-epochs 1 --trainer-config.limit-train-batches 1 --trainer-config.limit-val-batches 1 --trainer-config.num-sanity-val-steps 0 --trainer-config.check-val-every-n-epoch 1 --trainer-config.enable-validation --no-trainer-config.use-wandb --no-trainer-config.callbacks.use-rich-progress-bar --trainer-config.callbacks.use-tqdm-progress-bar`

- Online dataset train+val (uses temp config):
  `uv run nbv-train --config-path /tmp/nbv_online_one.toml`

## Temporary Config
- `/tmp/nbv_online_one.toml` created for online 1-step train/val with scene_id=81022, snippet_key_filter=000024.

## Suggestions / Next Steps
- Add a small helper command or config template for online runs with `batch_size` omitted (avoid CLI None parsing).
- Consider a strict mode for `vin_snippet_cache_mode="required"` to fail when an entry is missing (currently it only enforces cache metadata presence).
- Optional: add a cache coverage check between offline cache and vin snippet cache indices to report missing pairs.

## 2026-01-05 Updates
- Added configs:
  - `.configs/online_only.toml` (online-only training config, no batch_size).
  - `.configs/offline_cache_required_one_step.toml` (1-step offline cache run with vin_snippet_cache_mode="required").
- Enforced `vin_snippet_cache_mode="required"`:
  - Raises if cache config missing, if cache entry missing, or if snippet loading fails.
  - Offline cache now re-raises provider errors in required mode instead of falling back to empty snippets.

## 2026-01-05 Offline stats panel
- Added cache type switch in `offline_stats.py` (oracle_rri_cache vs vin_snippet_cache).
- Added VIN snippet cache stats collection in `offline_cache_utils.py` (points count, traj length, inv_dist_std mean).
- Coverage scan can use vin_snippet_cache metadata for dataset config.
