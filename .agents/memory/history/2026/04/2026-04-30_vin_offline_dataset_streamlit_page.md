---
id: 2026-04-30_vin_offline_dataset_streamlit_page
date: 2026-04-30
title: "VIN Offline Dataset Streamlit Page"
status: done
topics: [streamlit, vin, offline-store, diagnostics]
confidence: high
canonical_updates_needed: []
---

Task: recover the pruned offline dataset Streamlit workflow without restoring
legacy oracle-cache or VIN-snippet-cache APIs.

Method: added a standalone immutable VIN offline dataset page, routed it through
the Streamlit dispatcher and app navigation, and expanded
`collect_vin_offline_dataset_stats` with render-ready block diagnostics,
sampled row sanity summaries, and sampled histogram vectors.

Outputs: the new page can inspect either a direct store directory or an
experiment TOML whose datamodule source is `VinOfflineSourceConfig`. It reports
sample/split/scene coverage, numeric bytes, materialized block flags, manifest
block details, sampled row summaries, and candidate/RRI/VIN-length histograms.

Verification: `uv run ruff format` and `uv run ruff check` passed on touched
Python files. Targeted tests passed: `uv run pytest tests/test_panels_dispatcher.py`,
`uv run pytest tests/data_handling/test_vin_offline_store.py`, and
`uv run pytest tests/data_handling/test_public_api_contract.py`.

Canonical state impact: none. The implementation keeps immutable
`VinOfflineDataset` as the only supported offline training path and does not
restore legacy cache migration or runtime training APIs.
