---
id: 2026-05-14_rollout_store_manifest_upgrade
date: 2026-05-14
title: "Rollout Store Manifest Upgrade"
status: done
topics: [rollouts, zarr, metadata, provenance]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/rollouts/manifest.py
  - aria_nbv/aria_nbv/rollouts/zarr_store.py
  - aria_nbv/aria_nbv/rollouts/dataset_writer.py
  - aria_nbv/aria_nbv/rollouts/cli.py
  - aria_nbv/aria_nbv/rollouts/info_cli.py
  - aria_nbv/aria_nbv/data_handling/README.md
---

## Task

Implemented a manifest-backed rollout store contract so generated
`rollouts.zarr` shards expose generation metadata and source coverage without
loading replay payload arrays.

## Method

Added a top-level `manifest.json` sidecar, bumped the rollout schema to
`0.4-manifested-shards`, and wired manifest creation through the rollout writer
and `nbv-build-rollouts` CLI. Added `RolloutZarrStoreReader.manifest()` and
`nbv-rollouts-info` for cheap metadata inspection. The existing factual Zarr
tables remain the replay source of truth.

## Outputs

The regenerated smoke store at `.data/offline_cache/rollouts_v1_smoke.zarr`
now contains `manifest.json`; root attrs keep only compact metadata and a
manifest SHA-256. The manifest includes resolved config, raw TOML hash/text for
CLI runs, git/env summary, source coverage, config hashes, and counts.

## Verification

Ran targeted rollout, Rerun, and Streamlit panel tests; dry-ran and executed
`nbv-build-rollouts` for the smoke config; inspected the generated store through
`nbv-rollouts-info --json` and `--validate`.

## Canonical State Impact

No canonical memory update is needed. The durable datastore contract change is
documented in the package README and enforced by the schema/validation tests.
