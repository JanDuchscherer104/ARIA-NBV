---
id: 2026-05-10_candidate_rollout_dataset_generation
date: 2026-05-10
title: "Candidate Rollout Dataset Generation"
status: done
topics: [rollouts, candidate-generation, target-rri, zarr, q-h]
confidence: medium
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/pose_generation/
  - aria_nbv/aria_nbv/data_handling/
  - aria_nbv/tests/pose_generation/
  - aria_nbv/tests/data_handling/
  - .configs/build_rollouts_v1_smoke.toml
---

## Task

Implemented the first concrete V1 target-RRI rollout data-generation path as a standalone `rollouts.zarr` builder, keeping the immutable VIN offline strict-v7 store unchanged.

## Method

Added a fixed-count mixed candidate sampler with full-shell strategy, mixture, and sampler-probability provenance. Wired runtime actor-visible target context into counterfactual rollouts so `TARGET_POINT` candidates require a selected target center. Added target-cropped oracle RRI scoring from matched GT target AABBs and a rollout dataset writer/CLI that reads `VinOfflineDataset` samples, selects V1 targets, runs rollout recipes, and writes validated Zarr replay tables.

## Outputs

The new path writes candidate provenance into the existing rollout Zarr schema, tightens non-synthetic validation so actor-action rows cannot use placeholder provenance, and blocks `q_train_mask` unless the target record is actor-valid, GT-label-valid, finite-labeled, and non-padded. The smoke config is `.configs/build_rollouts_v1_smoke.toml`; the console entry point is `nbv-build-rollouts`.

## Verification

Ran focused pose-generation, target-selection, rollout-Zarr, RRI metric, lint, and CLI dry-run checks on 2026-05-10. Full-scale data generation remains unverified until a local/LRZ VIN offline store with live snippets and GT meshes is available.

## Canonical State Impact

No canonical state update is required yet. The implementation is a first local/small-shard data path; LRZ Slurm/DSS resume-safe generation and learned one-step policy selection remain follow-up work.
