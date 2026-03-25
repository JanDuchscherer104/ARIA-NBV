# Aria NBV Context Map

Use this map to pick the smallest relevant set of files before broad search.

## Fixed entrypoints
- Highest-level ground truth: `docs/typst/paper/main.typ`
- Canonical current truth: `.agents/memory/state/PROJECT_STATE.md`, `.agents/memory/state/DECISIONS.md`, `.agents/memory/state/OPEN_QUESTIONS.md`, `.agents/memory/state/GOTCHAS.md`
- Hot-path reference: `.agents/references/python_conventions.md`
- Broad source family index: `docs/_generated/context/source_index.md`
- Secondary references: `.agents/references/agent_memory_templates.md`, `.agents/references/context7_library_ids.md`

## Concept-to-source matrix

Only the non-obvious cross-surface routes live here. Obvious file-name or heading matches should be handled by `source_index.md`, outlines, or direct `rg`.

| Topic | Canonical state | References | Paper | Quarto docs | Literature | Code | First reveal command |
|---|---|---|---|---|---|---|---|
| Coordinate frames and conventions | `.agents/memory/state/DECISIONS.md`, `.agents/memory/state/GOTCHAS.md` | `.agents/references/python_conventions.md` | `docs/typst/paper/sections/05-coordinate-conventions.typ`, `docs/typst/paper/sections/12f-appendix-pose-frames.typ` | `docs/contents/ext-impl/efm3d_symbol_index.qmd`, `docs/contents/glossary.qmd` | `literature/tex-src/arXiv-project-aria/definitions.tex` | `oracle_rri/oracle_rri/pose_generation`, `oracle_rri/oracle_rri/rendering` | `scripts/nbv_typst_includes.py --paper --mode outline` |
| Oracle RRI computation | `.agents/memory/state/PROJECT_STATE.md`, `.agents/memory/state/DECISIONS.md`, `.agents/memory/state/GOTCHAS.md` | `.agents/references/python_conventions.md` | `docs/typst/paper/sections/05-oracle-rri.typ`, `docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ` | `docs/contents/impl/oracle_rri_impl.qmd`, `docs/contents/impl/rri_computation.qmd`, `docs/contents/theory/rri_theory.qmd` | `literature/tex-src/arXiv-VIN-NBV/sec/3_methods.tex` | `oracle_rri/oracle_rri/pipelines/oracle_rri_labeler.py`, `oracle_rri/oracle_rri/rendering/candidate_depth_renderer.py` | `scripts/nbv_get_context.sh match oracle_rri_labeler` |
| Candidate generation and pose sampling | `.agents/memory/state/PROJECT_STATE.md`, `.agents/memory/state/GOTCHAS.md` | `.agents/references/python_conventions.md` | `docs/typst/paper/sections/08-system-pipeline.typ` | `docs/contents/impl/data_pipeline_overview.qmd`, `docs/contents/impl/aria_nbv_package.qmd` | `literature/tex-src/arXiv-GenNBV/3-Method.tex` | `oracle_rri/oracle_rri/pose_generation/` | `scripts/nbv_get_context.sh match candidate` |
| Data contracts and typed containers | `.agents/memory/state/GOTCHAS.md` | `.agents/references/python_conventions.md` | `docs/typst/paper/sections/06-architecture.typ` | `docs/contents/impl/aria_nbv_package.qmd` | — | `oracle_rri/oracle_rri/data/efm_views.py`, `oracle_rri/oracle_rri/vin/types.py`, `oracle_rri/oracle_rri/utils/base_config.py` | `scripts/nbv_get_context.sh contracts` |
| VIN architecture and predictors | `.agents/memory/state/PROJECT_STATE.md`, `.agents/memory/state/DECISIONS.md`, `.agents/memory/state/GOTCHAS.md` | `.agents/references/python_conventions.md` | `docs/typst/paper/sections/06-architecture.typ`, `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ` | `docs/contents/impl/vin_nbv.qmd`, `docs/contents/impl/vin_v2_feature_proposals.qmd`, `docs/contents/impl/vin_coverage_aware_training.qmd` | `literature/tex-src/arXiv-VIN-NBV/sec/3_methods.tex`, `literature/tex-src/arXiv-EFM3D/method.tex` | `oracle_rri/oracle_rri/vin/` | `scripts/nbv_get_context.sh match VinModel` |
| Training objective and configs | `.agents/memory/state/DECISIONS.md`, `.agents/memory/state/OPEN_QUESTIONS.md`, `.agents/memory/state/GOTCHAS.md` | `.agents/references/python_conventions.md` | `docs/typst/paper/sections/07-training-objective.typ`, `docs/typst/paper/sections/07a-binning.typ`, `docs/typst/paper/sections/07b-training-config.typ` | `docs/contents/impl/vin_coverage_aware_training.qmd`, `docs/contents/impl/optuna_vin_v2_searchspace_2026-01-07.qmd` | `literature/tex-src/arXiv-VIN-NBV/sec/4_experiments.tex` | `oracle_rri/oracle_rri/vin/`, `oracle_rri/oracle_rri/configs/optuna_config.py` | `scripts/nbv_get_context.sh match train` |
| Offline cache and dataset splits | `.agents/memory/state/DECISIONS.md`, `.agents/memory/state/GOTCHAS.md` | `.agents/references/python_conventions.md` | `docs/typst/paper/sections/12h-appendix-offline-cache.typ` | `docs/contents/impl/data_pipeline_overview.qmd`, `docs/contents/setup.qmd` | `literature/tex-src/arXiv-EFM3D/dataset.tex` | `oracle_rri/oracle_rri/data/offline_cache.py`, `oracle_rri/oracle_rri/data/offline_cache_store.py`, `oracle_rri/oracle_rri/data/offline_cache_serialization.py` | `scripts/nbv_get_context.sh match offline_cache` |

## Reveal order by source family
- Conceptual or architectural question: `main.typ` -> `.agents/memory/state/` -> `.agents/references/` -> this map -> source-specific outline/index
- Paper or doc structure question: `scripts/nbv_typst_includes.py --paper --mode outline` or `scripts/nbv_qmd_outline.sh --compact`
- Literature-backed question: `scripts/nbv_literature_index.sh` -> `scripts/nbv_literature_search.sh "<term>"`
- Code-backed question: `oracle_rri/AGENTS.md` -> `.agents/references/python_conventions.md` -> `.agents/memory/state/GOTCHAS.md` -> `scripts/nbv_get_context.sh contracts` -> `modules` or `match <term>` -> `functions` or `classes` -> raw file reads
- Historical question: `.agents/memory/history/` only after the topic is localized and the current state docs are insufficient
