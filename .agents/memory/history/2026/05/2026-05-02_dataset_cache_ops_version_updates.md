---
id: 2026-05-02_dataset_cache_ops_version_updates
date: 2026-05-02
title: "Dataset Cache Ops Version Update Guidance"
status: done
topics: [skills, offline-store, dataset-cache, migration]
confidence: high
canonical_updates_needed: []
---

## Task

Improve the `dataset-cache-ops` skill so agents know how to handle immutable
VIN offline-store dataset-version updates.

## Method

Updated the skill to document the expected policy: keep readers strict, prefer
current-writer rebuilds for canonical stores, preserve expensive payloads only
when intended and semantically safe, use temporary migration/merge scripts
outside the repo, delete those helpers after verification, and avoid committing
permanent migration compatibility.

## Verification

Planned checks: skill quick validation and `make check-agent-memory`.

## Canonical State Impact

No canonical memory updates are needed.
