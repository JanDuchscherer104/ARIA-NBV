# Aria-NBV Context Map

Use this map only for non-obvious cross-surface routing. Obvious filename or
heading matches should use `docs/_generated/context/source_index.md`, outline
tools, or targeted `rg`.

## Fixed Entrypoints
- Highest-level ground truth: `docs/typst/paper/main.typ`
- Current truth: `.agents/memory/state/PROJECT_STATE.md`
- Decisions and gotchas: `.agents/memory/state/DECISIONS.md`,
  `.agents/memory/state/GOTCHAS.md`
- Source-family index: `docs/_generated/context/source_index.md`
- Package rules: `aria_nbv/AGENTS.md`
- Docs rules: `docs/AGENTS.md`

## Concept Routes
| Topic | First sources | First reveal |
|---|---|---|
| Coordinate frames and conventions | `docs/typst/paper/sections/05-coordinate-conventions.typ`, `docs/typst/paper/sections/12f-appendix-pose-frames.typ`, `.agents/references/python_conventions.md` | `scripts/nbv_typst_includes.py --paper --mode outline` |
| Oracle RRI computation | `aria_nbv/aria_nbv/rri_metrics/AGENTS.md`, `aria_nbv/aria_nbv/pipelines/oracle_rri_labeler.py`, `docs/typst/paper/sections/05-oracle-rri.typ` | `scripts/nbv_get_context.sh match OracleRriLabeler` |
| Candidate generation and pose sampling | `aria_nbv/aria_nbv/pose_generation/AGENTS.md`, `aria_nbv/aria_nbv/pose_generation/`, `docs/typst/paper/sections/08-system-pipeline.typ` | `scripts/nbv_get_context.sh match candidate` |
| Data contracts and typed containers | `aria_nbv/aria_nbv/data_handling/AGENTS.md`, `aria_nbv/aria_nbv/data_handling/efm_views.py`, `aria_nbv/aria_nbv/vin/types.py` | `make context-contracts` |
| VIN architecture and predictors | `aria_nbv/aria_nbv/vin/AGENTS.md`, `docs/typst/paper/sections/06-architecture.typ`, `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ` | `scripts/nbv_get_context.sh match VinModel` |
| Training objective and configs | `aria_nbv/aria_nbv/lightning/AGENTS.md`, `aria_nbv/aria_nbv/configs/AGENTS.md`, `docs/typst/paper/sections/07-training-objective.typ` | `scripts/nbv_get_context.sh match train` |
| Offline cache and dataset splits | `aria_nbv/aria_nbv/data_handling/AGENTS.md`, `aria_nbv/aria_nbv/data_handling/offline_cache_store.py`, `docs/typst/paper/sections/12h-appendix-offline-cache.typ` | `scripts/nbv_get_context.sh match offline` |
| Agent scaffold maintenance | `.agents/skills/aria-nbv-scaffold-maintenance/SKILL.md`, `scripts/validate_agent_scaffold.py`, `scripts/quarto_generate_agent_docs.py` | `make check-agent-scaffold` |

## Source-Family Handoffs
- Docs and literature: `aria-nbv-docs-context`
- Package contracts and symbols: `aria-nbv-code-context`
- Agent scaffold: `aria-nbv-scaffold-maintenance`
