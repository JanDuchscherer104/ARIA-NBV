---
id: 2026-04-30_offline_dataset_diagnostics_restored
date: 2026-04-30
title: "Offline Dataset Diagnostics Restored"
status: done
topics: [vin-offline-store, streamlit, diagnostics, data-handling]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/data_handling/_offline_diagnostics.py
  - aria_nbv/aria_nbv/data_handling/__init__.py
  - aria_nbv/aria_nbv/app/panels/offline_dataset.py
  - aria_nbv/tests/data_handling/test_vin_offline_store.py
artifacts:
  - /tmp/aria-nbv-page-inspect/offline_dataset_restored_diagnostics.png
  - /tmp/aria-nbv-page-inspect/offline_dataset_coverage_tab.png
---

## Task

Restored thesis-grade diagnostics on the immutable VIN Offline Dataset Streamlit
page while keeping the implementation on `VinOfflineStoreReader` and
`VinOfflineDatasetConfig`.

## Method

Added immutable-store diagnostics for RRI components, candidate geometry in the
reference-rig frame, runtime memory estimates, backbone numeric stats, lean VIN
batch shape preview, and raw ASE tar coverage. Extended the Streamlit panel with
tabs for RRI Components, Candidate Geometry, Batch & Memory, Backbone, Binner,
and Coverage. Excluded migration and cache/store discrepancy functionality.

## Verification

- `uv run ruff format aria_nbv/data_handling/_offline_diagnostics.py aria_nbv/app/panels/offline_dataset.py tests/data_handling/test_vin_offline_store.py`
- `uv run ruff check aria_nbv/data_handling/_offline_diagnostics.py aria_nbv/app/panels/offline_dataset.py tests/data_handling/test_vin_offline_store.py`
- `uv run pytest tests/data_handling/test_vin_offline_store.py -q`
- `uv run pytest tests/data_handling/test_public_api_contract.py tests/test_panels_dispatcher.py -q`
- Real-store smoke against `.data/offline_cache/vin_offline`: 43 samples, RRI
  components present, candidate pose summaries present, memory/backbone stats
  present, VIN batch shape preview present.
- Coverage smoke against one raw tar shard: 8 dataset snippets, 43 store
  snippets, 8 covered.
- Browser smoke at `http://localhost:8501/_page_offline_dataset`: restored tabs
  render after inspection; Coverage tab reports 136 dataset snippets, 43 store
  snippets, 43 covered, 31.62% coverage, and 17 tar shards scanned.

## Canonical State Impact

No canonical state updates required. The active immutable offline diagnostics
surface now supports M1/M2 inspection and later target-aware extensions without
reintroducing removed data-cache migration paths.
