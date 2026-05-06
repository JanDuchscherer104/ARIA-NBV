---
id: 2026-05-06_lrz_dry_run_templates
date: 2026-05-06
title: "LRZ Dry-Run Templates"
status: done
topics: [lrz, dss, slurm, pyxis, docs]
confidence: high
canonical_updates_needed: []
files_touched:
  - .configs/lrz/README.md
  - .configs/lrz/dry_run_matrix.toml
  - scripts/templates/lrz/README.md
  - scripts/templates/lrz/oracle_generation_dry_run.sbatch
  - scripts/templates/lrz/rollout_generation_dry_run.sbatch
  - scripts/templates/lrz/vin_training_dry_run.sbatch
  - scripts/templates/lrz/diagnostics_dry_run.sbatch
  - docs/contents/impl/lrz_dry_runs.qmd
  - docs/contents/impl/overview.qmd
  - docs/_quarto.yml
---

## Task

Added documentary LRZ dry-run templates and DSS layout documentation for oracle
generation, rollout generation, VIN training, and diagnostics. The templates
remain dry-run placeholders and do not implement generation code.

## Method

Used the LRZ AI Systems, docs-curator, and agents-db workflows. Ran a litkg
context pack for the cross-surface task, then added `.configs/lrz` planning
docs, `scripts/templates/lrz` Slurm/Pyxis dry-run templates, and a public
implementation page linked from the implementation overview and sidebar.

## Findings

The new docs preserve the current decision that LRZ deterministic sharding,
DSS staging, and resume-safe writes are prerequisites before scale generation.
They avoid hard-coding allocation-specific quota or inode numbers and keep
rollout staging out of a finalized package storage schema.

## Verification

- `bash -n scripts/templates/lrz/*.sbatch`
- `aria_nbv/.venv/bin/python` TOML parse for `.configs/lrz/dry_run_matrix.toml`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/impl/lrz_dry_runs.qmd`
- `make check-agent-memory`
- `make agents-db AGENTS_ARGS='validate'`
- secret/path scan over the LRZ additions; only generic credential wording was
  matched, with no committed usernames, tokens, fixed DSS paths, or host-local
  paths.

## Canonical State Impact

No canonical state update is required. Existing project state already names
LRZ deterministic sharding, Slurm/DSS staging, and resume-safe writes as hard
gates before full-scale generation.
