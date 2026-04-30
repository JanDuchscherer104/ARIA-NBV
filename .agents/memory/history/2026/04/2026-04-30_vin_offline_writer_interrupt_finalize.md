---
id: 2026-04-30_vin_offline_writer_interrupt_finalize
date: 2026-04-30
title: "VIN Offline Writer Interrupt Finalization"
status: done
topics: [data-handling, vin-offline-store, oracle, cli]
confidence: high
canonical_updates_needed: []
---

## Task

Investigate whether `nbv-build-offline` persisted samples after a single
Ctrl-C and make the behavior graceful for future partial runs.

## Findings

The previous run did not persist samples. The store directory
`.data/offline_cache/vin_offline` was absent and the temp directory only
contained an empty `shards/` directory. The writer buffered rows in memory until
`samples_per_shard` and did not catch `KeyboardInterrupt`, so a Ctrl-C before
the first shard flush discarded prepared rows.

## Changes

`VinOfflineWriter.run()` now catches `KeyboardInterrupt`, flushes already
prepared rows, writes manifest/index/splits, renames the temp store into the
configured store directory, and records interrupt metadata in manifest
`stats.interrupted` and `provenance.finalized_after_interrupt`.

## Verification

- `cd aria_nbv && uv run ruff check aria_nbv/data_handling/_offline_writer.py tests/data_handling/test_vin_offline_store.py`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py -k 'keyboard_interrupt or offline_writer_finalizes or prepare_vin_offline_sample_filters'`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/data_handling/test_public_api_contract.py`

The regression test confirms an interrupt during the third sample finalizes the
first two prepared samples into a valid partial store.
