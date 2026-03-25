# Aria NBV Context Map

Use this map to pick the smallest set of files to open.

## High-level overview
- Project entry: `docs/index.qmd`
- Current TODOs: `docs/contents/todos.qmd`
- Codebase snapshot: `make context` + `make context-dir-tree`

## RRI definition / metrics
- Theory: `docs/contents/theory/rri_theory.qmd`, `docs/contents/theory/surface_metrics.qmd`
- Paper: `docs/typst/paper/sections/03-problem-formulation.typ`, `docs/typst/paper/sections/05-oracle-rri.typ`

## Dataset / ASE
- Dataset doc: `docs/contents/ase_dataset.qmd`
- Paper: `docs/typst/paper/sections/04-dataset.typ`

## Oracle pipeline
- Impl overview: `docs/contents/impl/oracle_rri_impl.qmd`, `docs/contents/impl/rri_computation.qmd`
- Paper: `docs/typst/paper/sections/05-oracle-rri.typ`, `docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ`
- Code: `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py`

## Candidate generation
- Docs: `docs/contents/impl/data_pipeline_overview.qmd`
- Paper: `docs/typst/paper/sections/08-system-pipeline.typ`
- Code: `oracle_rri/oracle_rri/pose_generation/*`

## VIN model / learning
- Docs: `docs/contents/impl/vin_nbv.qmd`, `docs/contents/impl/vin_v2_feature_proposals.qmd`
- Paper: `docs/typst/paper/sections/06-architecture.typ`, `docs/typst/paper/sections/07-training-objective.typ`
- Code: `oracle_rri/oracle_rri/vin/*`

## Evaluation protocol
- Docs: `docs/contents/theory/surface_metrics.qmd`
- Paper: `docs/typst/paper/sections/09a-evaluation.typ`

## Typst slides
- Sources: `docs/typst/slides/*.typ`
- Use `scripts/nbv_typst_includes.py` to expand slide includes if present.

## Literature (LaTeX)
- Sources: `literature/**/*.tex`, `literature/**/*.bib`
- Use `scripts/nbv_literature_search.sh "<query>"` for focused grep.

## AST context summary (code)
- Use `scripts/nbv_get_context.sh packages` or `classes` with `oracle_rri/scripts/get_context.py`.
