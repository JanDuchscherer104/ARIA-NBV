---
id: 2025-12-31_ase_efm_snippet_stats_2025-12-31
date: 2025-12-31
title: "Ase Efm Snippet Stats 2025 12 31"
status: legacy-imported
topics: [ase, efm, snippet, stats, 2025]
source_legacy_path: ".codex/ase_efm_snippet_stats_2025-12-31.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# ASE EFM snippet stats (local .data/ase_efm)

## Task summary
- Scanned local ASE-ATEK shard snapshot in `.data/ase_efm` and summarized per-scene snippet counts.
- Generated histogram figure and documented stats in `docs/contents/ase_dataset.qmd`.

## Key findings
- Snapshot contains **100 scenes** with **4,608 snippets** total.
- Per-scene snippet counts: **min 8**, **p10 8**, **median 40**, **mean 46.1**, **p90 88.8**, **max 152** (scan date: 2025-12-31).
- Top counts observed: scene `83550` (152), `81283` (136), `81286` (136), `83515` (128), `82004` (120), `82647` (120), `82889` (120).

## Changes made
- Added histogram figure: `docs/figures/ase_efm_snippet_hist.png`.
- Documented snapshot stats + histogram in `docs/contents/ase_dataset.qmd`.
- Added Wikipedia histogram reference in `docs/references.bib`.

## Assumptions / caveats
- Snippet counts are computed by counting `sequence_name.txt` entries inside each shard (assumed to be one per snippet) and summing across shards per scene.
- Counts reflect **local** snapshot only; not necessarily representative of full ASE.

## Suggestions
- If `.data/ase_efm` changes, re-run the scan and regenerate the histogram.
- Consider adding a small CLI/script under `scripts/` to reproduce these stats programmatically.
