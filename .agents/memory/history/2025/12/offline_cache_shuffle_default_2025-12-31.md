---
id: 2025-12-31_offline_cache_shuffle_default_2025-12-31
date: 2025-12-31
title: "Offline Cache Shuffle Default 2025 12 31"
status: legacy-imported
topics: [offline, cache, shuffle, default, 2025]
source_legacy_path: ".codex/offline_cache_shuffle_default_2025-12-31.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Offline cache default shuffle

## Summary
- Updated `OracleRriCacheWriterConfig` to default to `AseEfmDatasetConfig(wds_shuffle=True)` so offline cache creation shuffles shards by default.
- Added README note to document the new default and how to override for deterministic order.

## Files touched
- `oracle_rri/oracle_rri/data/offline_cache.py`
- `oracle_rri/oracle_rri/data/README.md`

## Notes
- Change affects only the cache writer default; other dataset usage remains unchanged unless explicitly configured.
