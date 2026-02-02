# Paper “AI slop” TODO pass (2026-01-30)

Goal: add explicit TODO comments throughout the Typst paper where content is
irrelevant/redundant/inconsistent, and where notation diverges from `slides_4.typ`
or from `docs/typst/shared/macros.typ` (especially `#symb`, `#eqs`, and bold(...) usage).

## Scope (files touched)

Main paper sections:

- `docs/typst/paper/sections/01-introduction.typ`
- `docs/typst/paper/sections/02-related-work.typ`
- `docs/typst/paper/sections/03-problem-formulation.typ`
- `docs/typst/paper/sections/04-dataset.typ`
- `docs/typst/paper/sections/05-coordinate-conventions.typ`
- `docs/typst/paper/sections/05-oracle-rri.typ`
- `docs/typst/paper/sections/06-architecture.typ`
- `docs/typst/paper/sections/07-training-objective.typ`
- `docs/typst/paper/sections/07a-binning.typ`
- `docs/typst/paper/sections/07b-training-config.typ`
- `docs/typst/paper/sections/08a-frustum-pooling.typ`
- `docs/typst/paper/sections/09a-evaluation.typ`
- `docs/typst/paper/sections/09b-ablation.typ`
- `docs/typst/paper/sections/09c-wandb.typ`
- `docs/typst/paper/sections/10-discussion.typ`
- `docs/typst/paper/sections/10a-entity-aware.typ`
- `docs/typst/paper/sections/11-conclusion.typ`

Appendices:

- `docs/typst/paper/sections/12b-appendix-extra.typ`
- `docs/typst/paper/sections/12f-appendix-pose-frames.typ`
- `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ`
- `docs/typst/paper/sections/12h-appendix-offline-cache.typ`
- `docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ` (already minimized in prior step)

## Main themes flagged via TODOs

### 1) Notation drift vs. `macros.typ` / `slides_4.typ`

- Raw frame-transform expressions like `T(symb.frame...)` appear in the paper (e.g.,
  coordinate conventions / architecture) and should be replaced with the canonical macro
  form (`#T(...)` + `#fr_*`) used elsewhere.
- Matrices/vectors are sometimes written as scalars (`K_i`, quoted `"RRI"_...`); TODOs call
  out replacing with `bold(...)` or `#symb` entries.
- Several metric equations are manually retyped in the paper; TODOs recommend reusing
  `#eqs.metrics.*` and `#eqs.features.*` to prevent divergence.

### 2) Hard-coded numbers that should be imported from artifacts

- Multiple sections hard-code dataset split counts (80 scenes / 883 snippets / 706–177 split)
  and performance metrics (Spearman/top-3 acc). TODOs request importing from the same JSON/TOML
  artifacts that slides use (e.g., offline-cache stats + W&B summary exports).

### 3) Redundancy across sections

- Pipeline details are now consolidated in `05-oracle-rri.typ`; TODOs in Problem Formulation,
  Coordinate Conventions, Evaluation, and optional modules note where to shorten and cross-reference.
- Some figures (snippet overview) appear in multiple places; TODOs mark where to pick one location.

### 4) “Strong claim” / speculative language

- TODOs mark broad claims that need either supporting evidence/citations or softened phrasing
  (e.g., sim-to-real “directly testable”, causal interpretations in W&B analysis, “practical path”).

## Verification

- `cd docs && typst compile --root . typst/paper/main.typ /tmp/nbv_paper_compile_test.pdf`
- `cd docs && typst compile --root . typst/slides/slides_4.typ /tmp/nbv_slides_4_compile_test.pdf`

## Suggested next step

Pick one cleanup track and execute end-to-end:

1) **Notation unification pass**: replace `T(symb.frame...)`, quoted `"RRI"`, and non-bold matrices
   with canonical `#symb/#eqs` usage (keeping math consistent with slides).
2) **Artifact-driven numbers**: create a tiny `paper/data/*.json` or reuse slide JSON directly for
   counts/metrics; remove all hard-coded metrics from prose/tables.
3) **De-duplication**: decide where each equation/figure “lives” (Problem vs Oracle vs Eval) and
   remove duplicates, leaving cross-refs.

