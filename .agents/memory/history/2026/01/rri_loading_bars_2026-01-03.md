---
id: 2026-01-03_rri_loading_bars_2026-01-03
date: 2026-01-03
title: "Rri Loading Bars 2026 01 03"
status: legacy-imported
topics: [loading, bars, 2026, 01, 03]
source_legacy_path: ".codex/rri_loading_bars_2026-01-03.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

## Task
Suggest additional places to use the new Streamlit loading bars.

## Suggestions
- Offline cache scan loops (train/val label hist, cache writer runs).
- Binner fitting iterations (train dataloader loop).
- Candidate generation / depth/pointcloud rendering pipelines (per-snippet loops).
- Dataset downloads or cache rebuilds (long-running file I/O).
- Heavy plotting/analysis panels that aggregate large caches.

## Notes
- Use progress bars for per-split scans and long-running per-snippet loops; add a cancel/skip button when possible.
