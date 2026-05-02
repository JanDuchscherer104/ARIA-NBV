---
id: 2026-05-02_litkg_backlog_offline_store_smoke
date: 2026-05-02
title: "Litkg Backlog And Offline Store Smoke"
status: done
topics: [agents-db, litkg, offline-store, vin, diagnostics]
confidence: high
canonical_updates_needed: []
---

## Task

Resolve completed litkg agent-memory TODOs and unblock the local
`offline_only.toml` VIN offline-store smoke path without adding reader
backward compatibility.

## Method

Resolved `todo-040`, `todo-041`, and `todo-042` because commit `3a3a4ff`
already added the `aria-litkg-memory` skill, root litkg retrieval targets, and
expanded `.configs/litkg.toml` source classes. Left the remaining litkg work
active: authority/freshness ranking, consolidation, stale/contradiction checks,
Semantic Scholar refresh, and agents-db KG edges.

For the local data artifact, the selected policy was no backward compatibility:
the reader remains strict, the outdated local v5 store was migrated as an
artifact, and new samples were generated through the current writer.

The local `.data/offline_cache/vin_offline` diagnostic store now has a v6
manifest, 48 sample-index rows, four shards, and a 38/10 train/val split. The
old 43 rows were not recomputed; five current-writer rows were appended for
`AriaSyntheticEnvironment_81286_AtekDataSample_000043` through `000047`.
Temporary `/tmp` writer config and sidecar store artifacts were removed after
the merge.

## Verification

Checks completed:

- `make agents-db AGENTS_ARGS='validate'`
- `make agents-db`
- `make check-agent-memory`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py tests/vin/test_vin_utils.py`
- `cd aria_nbv && uv run nbv-summary --config-path offline_only.toml`

## Canonical State Impact

No durable canonical-memory updates are needed beyond the resolved backlog
records and this debrief.
