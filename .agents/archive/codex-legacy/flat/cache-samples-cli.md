# Task: Cache samples CLI

## Summary
- Added `nbv-cache-samples` script (kept `nbv-cache-oracle` as an alias) and expanded the cache CLI to accept `-n/--num-samples` as a `max_samples` alias.
- Fixed the Pydantic "not fully defined" error by rebuilding the CLI config with the `OracleRriCacheWriter` type in scope.
- Added optional RRI binner fitting from cached samples via `--fit-binner` and `--binner-*` options.

## Findings / Potential Issues
- `--fit-binner` loads cached samples and requires depth/pointclouds to be present; missing caches will still raise when decoding.
- Running the cache CLI without available ATEK tar shards will still raise the dataset validation error (as before).

## Suggestions
- If cache-only workflows without depth/pointclouds are expected, consider adding a lightweight cache dataset mode for binner fitting.

## Update (2025-12-31)
- Merged offline cache CLI into `oracle_rri.lightning.cli` as `cache_main`; removed duplicated `offline_cache_cli.py`.
- Switched CLI parsing to Pydantic Settings (no manual arg normalization); added `AliasChoices` for `config_path` and `max_samples` to accept `--config-path` and `--num-samples/-n`.
- Entry points `nbv-cache-samples` / `nbv-cache-oracle` now point to `oracle_rri.lightning.cli:cache_main`.
- Replaced custom deep-merge helper with Pydantic's internal `deep_update` utility for TOML + CLI override merging.

## Validation
- `ruff format` / `ruff check` on `oracle_rri/oracle_rri/lightning/cli.py` pass.
- `uv run pytest oracle_rri/oracle_rri/lightning/cli.py -q` fails during import due to missing `power_spherical` in the current environment (unrelated to CLI changes).
