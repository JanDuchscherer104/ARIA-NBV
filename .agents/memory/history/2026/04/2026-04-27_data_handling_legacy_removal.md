---
id: 2026-04-27_data_handling_legacy_removal
date: 2026-04-27
title: "Data Handling Legacy Removal"
status: done
topics: [data-handling, vin-offline-store, cleanup]
confidence: high
canonical_updates_needed: []
---

## Task

Remove the completed data-handling migration compatibility layer after the VIN
offline-store cutover.

## Method

Inspected the current data-handling, Lightning, app, config, and test surfaces
with grep plus code-index context. Deleted the legacy oracle-cache,
VIN-snippet-cache, compatibility wrapper, and migration modules. Simplified the
Streamlit diagnostics and attribution panels to use the canonical datamodule and
VIN offline-store path.

## Findings

The only active non-test users of the removed modules were app diagnostics,
candidate cache UI, RRI split histograms, and Lightning source-selection
branches. These were converted to root data-handling exports and
`VinOfflineSourceConfig` / `VinOfflineDatasetConfig`.

The immutable offline store no longer writes migration provenance fields on
sample-index rows. The store format version was bumped to require rebuilt
stores.

## Verification

- `uv run ruff format` on touched Python files.
- `uv run ruff check` on touched Python files.
- `uv run pytest tests/data_handling/test_public_api_contract.py tests/data_handling/test_vin_offline_store.py tests/vin/test_vin_utils.py`
- Parsed the updated TOML configs with `tomllib`.

## Canonical State Impact

No canonical state doc was changed. The cleanup updates module-local docs and
the data-handling guidance instead.
