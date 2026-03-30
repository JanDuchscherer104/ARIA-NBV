---
id: 2026-03-25_deleted_docs_asset_reference_audit
date: 2026-03-25
title: "Deleted Docs Asset Reference Audit"
status: done
topics: [docs, cleanup, audit]
confidence: high
canonical_updates_needed: []
---

## Task

Investigated whether any files currently deleted in `git status` are still referenced by source docs, slides, or the Typst paper.

## Method

- Ran `make context` and reviewed the repo guidance surfaces.
- Enumerated deleted paths from `git status --porcelain=v1 -z`.
- Searched current source docs only: `docs/**/*.qmd` and `docs/**/*.typ`, excluding generated/site-lib trees.
- Verified exact path-style matches manually to separate real breakages from basename-only false positives.

## Findings

Confirmed stale references to deleted assets:

- `docs/contents/impl/vin_nbv.qmd` still references deleted figures under `docs/figures/impl/vin/`:
  - `vin_sh_components.png`
  - `vin_radius_fourier_features.png`
  - `vin_rri_binning.png`
  - `vin_rri_thresholds.png`
- `docs/typst/slides/slides_3.typ` still references deleted figures under `docs/figures/impl/vin/`:
  - `vin_shell_descriptor_concept.png`
  - `vin_pose_descriptor.png`
  - `vin_sh_components.png`
  - `vin_radius_fourier_features.png`
  - `vin_evl_features.png`
  - `vin_rich_summary.png`
  - `vin_rri_binning.png`
- `docs/contents/ext-impl/prj_aria_tools_impl.qmd` still references deleted `docs/figures/impl/prj_aria/projectaria_core_classes.svg`.
- `docs/contents/literature/vin_nbv.qmd` still references deleted `docs/figures/VIN_arch.png`; a likely moved replacement exists at `docs/figures/arXiv-VIN-NBV/VIN_arch.png`.

Also found stale code-path mentions inside docs:

- `docs/contents/todos.qmd`, `docs/contents/impl/vin_nbv.qmd`, and `docs/typst/slides/slides_3.typ` still mention deleted module paths:
  - `oracle_rri/oracle_rri/vin/rri_binning.py`
  - `oracle_rri/oracle_rri/vin/coral.py`

Confirmed non-issues:

- Deleted `docs/_shared/references.bib` is not used by current source docs; sources point to `docs/references.bib` or `/references.bib`.
- Deleted old Mermaid assets under `docs/figures/diagrams/mermaid/` are not used by current sources; current paper/slides/docs point to `docs/figures/diagrams/vin_nbv/mermaid/`.
- Deleted `.codex/_render/*` artifacts are not referenced by current source docs.

## Canonical State

This task only identified stale references during repo cleanup.

## Prompt Follow-Through

This note predates the privileged owner-directive memory contract. No additional durable owner prompt items were backfilled here beyond any canonical state updates already recorded in this debrief.
