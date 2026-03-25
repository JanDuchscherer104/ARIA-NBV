---
id: 2025-12-31_offline_cache_metadata_identification_2025-12-31
date: 2025-12-31
title: "Offline Cache Metadata Identification 2025 12 31"
status: legacy-imported
topics: [offline, cache, metadata, identification, 2025]
source_legacy_path: ".codex/offline_cache_metadata_identification_2025-12-31.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Offline cache metadata identification

## Summary
- Added a TODO to document that offline cache metadata tracks only a single config snapshot, even when samples from multiple configs exist in the same cache directory.
- Captured the need for lightweight per-sample or per-hash bookkeeping so we can identify and drop subsets without invalidating anything (treat as augmentation).

## Key findings
- Compatibility is enforced only at append time; load-time does not validate/filter by config.
- Sample filenames already encode `config_hash`, but metadata/index do not provide an easy view of config groupings.

## Suggestions
- Extend `index.jsonl` to include `config_hash` per entry, or add a sidecar summary map (`config_hash -> count + signatures`).
- Provide a small CLI/summary helper to report counts per config hash.
