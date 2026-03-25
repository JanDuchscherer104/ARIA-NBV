---
id: 2026-01-05_vin_snippet_cache_multiprocessing_2026-01-05
date: 2026-01-05
title: "Vin Snippet Cache Multiprocessing 2026 01 05"
status: legacy-imported
topics: [snippet, cache, multiprocessing, 2026, 01]
source_legacy_path: ".codex/vin_snippet_cache_multiprocessing_2026-01-05.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# VIN snippet cache multiprocessing (2026-01-05)

## Summary
- Added DataLoader-backed VIN snippet cache builder so cache creation can use multiple workers.
- CLI now accepts `--num-workers`, `--persistent-workers`, `--prefetch-factor`, `--use-dataloader`, and `--skip-missing-snippets`.
- Missing snippets can be skipped without failing the build (configurable).
- EFM snippet loader now rebuilds the per-scene dataset once if a snippet is not found in the current stream.

## Key changes
- `oracle_rri/oracle_rri/data/vin_snippet_cache.py`: new `VinSnippetCacheBuildDataset`, `_build_vin_payload`, DataLoader path, skip-missing logic.
- `oracle_rri/oracle_rri/lightning/cli.py`: new CLI flags and wiring from datamodule settings.
- `oracle_rri/oracle_rri/data/efm_snippet_loader.py`: rebuild dataset on first miss to avoid false negatives from exhausted streams.
- `tests/data/test_vin_snippet_cache.py`: added tests for DataLoader path and skip-missing behavior.
- `tests/data/test_efm_snippet_loader.py`: added test for dataset rebuild on miss.

## Why you saw `Failed to locate snippet`
The WDS iterator is stateful; once it streams past a snippet, a later lookup returns nothing. With multiprocessing, snippets are processed out-of-order, so misses can happen even when the data exists. The loader now retries once with a fresh dataset instance.

## How to resume cache build with multiprocessing
```
uv run nbv-cache-vin-snippets \
  --config-path ./.configs/offline_cache_required_one_step_vin_cache.toml \
  --split all \
  --resume \
  --num-workers 8 \
  --skip-missing-snippets
```

Notes:
- If a build is already running with `--overwrite`, stop it before restarting with `--resume` to avoid concurrent writes.
- `--use-dataloader` forces the DataLoader path even when `--num-workers 0` (useful for debugging).
