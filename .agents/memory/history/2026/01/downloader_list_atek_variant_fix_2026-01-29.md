---
id: 2026-01-29_downloader_list_atek_variant_fix_2026-01-29
date: 2026-01-29
title: "Downloader List Atek Variant Fix 2026 01 29"
status: legacy-imported
topics: [downloader, list, atek, variant, 2026]
source_legacy_path: ".codex/downloader_list_atek_variant_fix_2026-01-29.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Downloader list-mode variant fix (2026-01-29)

## Task
- Fix `nbv-downloader -m list` to respect `-c` (ATEK config) instead of reporting the max-shard config.
- Report:
  - Whether `efm_eval` snippets differ from `efm`.
  - Download coverage (% shards) for `efm` and `efm_eval`.
  - Downloaded snippet counts for `efm` and `efm_eval` in list output (note: each shard contains multiple snippets).

## Findings
- Root cause: `cli_list()` used `ASEMetadata.get_scenes_with_meshes()` without a config, and `ASEMetadata._maybe_store()` keeps the per-scene entry with the **largest** `shard_count` across configs.
  - That made list-mode show the **max-shards variant** (typically `cubercnn_eval`) even when `-c efm` was passed.
- Current manifest totals for **GT-mesh scenes**:
  - `efm`: 576 shards across 100 scenes
  - `efm_eval`: 576 shards across 100 scenes
  - `cubercnn_eval`: 1641 shards across 100 scenes
- On disk:
  - `.data/ase_efm`: 576/576 shard tars downloaded
  - `.data/ase_efm_eval`: 576/576 shard tars downloaded
- Snippet multiplicity:
  - Each shard tar contains multiple snippets; for the current local downloads, it is **8 snippets per shard** → 4608 snippets total per variant (576 * 8).
- `efm_eval` vs `efm`:
  - The ATEK manifest lists different CDN filenames + SHA1 sums for `efm` vs `efm_eval`.
  - However, spot-checking matching local shards shows identical member file lists and identical hashes for representative member payload files; snippet payloads appear equivalent (difference seems to be archive-level packaging).

## Changes Made
- `oracle_rri/oracle_rri/data/metadata.py`
  - Extended `ASEMetadata.get_scenes_with_meshes(config: str | None = None)` so list mode can request config-specific scenes.
- `oracle_rri/oracle_rri/data/download_stats.py` (new)
  - Added helpers to compute downloaded shard coverage and snippet counts from local shard tar files.
  - Snippet counting counts `*.sequence_name.txt` members; list mode uses a small sample to infer a constant snippets-per-shard count (fast), with fallback to a full scan if inconsistent.
- `oracle_rri/oracle_rri/data/downloader.py`
  - `cli_list()` now calls `get_scenes_with_meshes(config=config.atek_config_name)` so `-c` is respected.
  - Added a “Downloaded overview (GT-mesh scenes)” section for `efm` + `efm_eval`, including shards %, scenes, and snippet totals.
- `tests/data/test_ase_downloader_list_stats.py` (new)
  - Covers config-specific mesh scene selection, snippet counting, and downloaded stat aggregation.

## Validation
- `ruff format` + `ruff check` on touched files.
- `oracle_rri/.venv/bin/python -m pytest -q tests/data/test_ase_downloader_list_stats.py`

## Follow-ups / Notes
- If you want exact snippet totals deterministically in list mode, we can add a CLI toggle (e.g. `--snippet-count-mode exact|estimate`) or cache shard snippet counts under `.data/` (ignored by git).
