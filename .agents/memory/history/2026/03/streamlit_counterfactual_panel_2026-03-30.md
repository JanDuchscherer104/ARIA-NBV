---
id: 2026-03-30_streamlit_counterfactual_panel
date: 2026-03-30
title: "Streamlit counterfactual rollout panel"
status: done
topics: [streamlit, pose-generation, counterfactuals, plotting]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/app/panels/candidates.py
  - aria_nbv/tests/app/panels/test_candidates_panel.py
artifacts:
  - candidate-page counterfactual rollout subsection with cached execution, plots, summaries, and logs
assumptions:
  - counterfactual rollouts should be an on-demand diagnostic subsection of the existing candidate page instead of a standalone Streamlit page
---

Task

Integrate the new multi-step counterfactual rollout utilities into the Streamlit pose-generation section.

Method

Extended the existing candidate page with an experimental "Counterfactual Rollouts" expander. The panel builds a rollout config from the active candidate generator config, runs the rollout generator on demand, caches results in Streamlit session state, captures Console output into a log tab, and renders both full rollout paths and a per-step candidate shell view using the new counterfactual plotting helpers.

Findings

The main integration concern was recomputation: running rollouts directly from widget values would rerun on every UI interaction, so the panel now uses an explicit run button and a stable cache key derived from the sample and rollout config. The per-step shell view also needs to guard against trajectories that terminate before selecting any step.

Verification

- `cd aria_nbv && ruff format aria_nbv/app/panels/candidates.py tests/app/panels/test_candidates_panel.py`
- `cd aria_nbv && ruff check aria_nbv/app/panels/candidates.py tests/app/panels/test_candidates_panel.py`
- `cd aria_nbv && uv run pytest -s tests/app/panels/test_candidates_panel.py tests/pose_generation/test_counterfactuals.py`

Canonical state impact

No canonical state doc changed. This extends the diagnostic UI for pose-generation work without changing the repo's architectural claims.
