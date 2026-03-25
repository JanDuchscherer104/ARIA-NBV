---
id: 2026-01-30_oracle_rri_section_consolidation_2026-01-30
date: 2026-01-30
title: "Oracle Rri Section Consolidation 2026 01 30"
status: legacy-imported
topics: [section, consolidation, 2026, 01, 30]
source_legacy_path: ".codex/oracle_rri_section_consolidation_2026-01-30.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Oracle RRI section consolidation (2026-01-30)

## Goal

Consolidate the full Oracle RRI labeling pipeline description (candidate generation → depth rendering → backprojection → point↔mesh scoring → optional binning) into a single main paper section: `docs/typst/paper/sections/05-oracle-rri.typ`.

## What changed

### 1) Main section becomes the single source of truth

- Expanded `docs/typst/paper/sections/05-oracle-rri.typ` to include:
  - inputs/outputs + example snippet figure
  - candidate center sampling + orientation + pruning rules (with key equations + figures)
  - PyTorch3D depth rendering details + parameter table + diagnostics figure
  - NDC-aligned backprojection details + parameter table + diagnostics figure
  - Chamfer accuracy/completeness + RRI equations via `#eqs`
  - scoring settings table + result figures (acc/comp + RRI bars)
  - optional ordinal binning (equations + config table + binning figures)
- Added an explicit section label: `<sec:oracle-rri>` so other sections can reference it.

### 2) Appendix de-duplicated

- Replaced `docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ` with a minimal pointer to Section 5 to avoid duplicated tables/figures/labels.
- Updated `docs/typst/paper/sections/03-problem-formulation.typ` to reference `@sec:oracle-rri` instead of the appendix for binning context.

### 3) Parameter tables are now consistent with slides

- In `docs/typst/paper/sections/05-oracle-rri.typ`, the numeric values shown in parameter tables are loaded from:
  - `/typst/slides/data/paper_figures_oracle_labeler.toml`
- This avoids stale hard-coded numbers (the previous appendix tables had drifted from the current TOML).

## Notes / gotchas

- Typst scripts (`_t`, `_q`, …) must live inside math mode; fixed several instances by wrapping with `$...$` (e.g., `$#(symb.oracle.points)_t$`).

## Verification

- `cd docs && typst compile --root . typst/paper/main.typ /tmp/nbv_paper_compile_test.pdf`
- `cd docs && typst compile --root . typst/slides/slides_4.typ /tmp/nbv_slides_4_compile_test.pdf`

## Follow-ups (optional)

- If you want *zero* remaining redundancy, we can further trim the Problem Formulation / Training sections where they restate RRI/binning equations now present in Section 5, and instead reference `@sec:oracle-rri` + `#eqs`.
