---
id: 2026-03-30_thesis_outlook_counterfactual_rl_slides
date: 2026-03-30
title: "Thesis Outlook Deck: Counterfactual RL Slides"
status: done
topics: [slides, typst, counterfactual-rollouts, rl, figures]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/slides/slides_thesis_outlook.typ
  - docs/typst/shared/macros.typ
  - aria_nbv/scripts/export_thesis_rl_figures.py
artifacts:
  - docs/figures/thesis_outlook/counterfactual_paths.png
  - docs/figures/thesis_outlook/counterfactual_step.png
  - docs/figures/thesis_outlook/policy_comparison.png
  - docs/typst/slides/slides_thesis_outlook.pdf
assumptions:
  - Synthetic diagnostic figures are acceptable in the thesis outlook deck as long as they are explicitly labeled as diagnostics rather than ASE benchmarks.
---

Task
- Update the thesis outlook presentation with the new counterfactual rollout / RL theory-to-implementation bridge and include figures from the newly added plotting surfaces.

Method
- Added a deterministic figure export script over a synthetic unit-box scene using the implemented multi-step rollout and Gymnasium RL env surfaces.
- Exported slide-ready PNG assets into `docs/figures/thesis_outlook/`.
- Inserted two new slides into `slides_thesis_outlook.typ` for the rollout scaffold and RL environment boundary.
- Fixed a pre-existing Typst compatibility issue in `docs/typst/shared/macros.typ` by replacing `dot.o` with `dot.op` for `typst 0.13.1`.

Findings / outputs
- The deck now shows concrete progress instead of only future directions:
  - multi-step counterfactual rollout boundary
  - RL env / diagnostics boundary
  - rollout path, shell diagnostic, and policy sanity-check figures
- The offline-only RL slide was simplified so the deck stays at 10 pages and no longer splits across two pages.

Verification
- `cd /home/jandu/repos/NBV/aria_nbv && ruff check scripts/export_thesis_rl_figures.py`
- `cd /home/jandu/repos/NBV/aria_nbv && uv run --with 'kaleido==0.2.1' python scripts/export_thesis_rl_figures.py`
- `cd /home/jandu/repos/NBV/docs && typst compile typst/slides/slides_thesis_outlook.typ --root .`
- `cd /home/jandu/repos/NBV/aria_nbv && uv run pytest -s tests/pose_generation/test_counterfactuals.py tests/rl/test_counterfactual_env.py`

Canonical state impact
- None. This was a presentation/documentation update over already-implemented rollout and RL surfaces.
