---
id: 2026-05-15_counterfactual_rollout_plotly_target_overlays
date: 2026-05-15
title: "Counterfactual Rollout Plotly Target Overlays"
status: done
topics: [streamlit, counterfactual-rollouts, plotting, target-rri]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/app/scene_view.py
  - aria_nbv/aria_nbv/app/panels/data.py
  - aria_nbv/aria_nbv/app/panels/counterfactual_rollouts.py
  - aria_nbv/aria_nbv/utils/data_plotting.py
  - aria_nbv/aria_nbv/pose_generation/plotting.py
---

## Task

Aligned the Counterfactual Rollouts Plotly 3D viewer with the Data page scene-view pattern and made the active target/evaluation target visible in the rollout view.

## Method

Added shared app-level scene controls for mesh opacity, semidense points, source trajectory, source frusta, bounds, and all-GT OBB overlays. Extended the Plotly builder layer with reusable oriented-box helpers and rollout-specific actor-visible and matched-GT target OBB methods. The Counterfactual Rollouts page now uses those controls in the path and step-shell tabs with minimal evidence-view defaults.

## Verification

Ran focused syntax/import, app-panel, pose-generation, and lint checks. Full `git diff --check` is still blocked by unrelated dirty generated PDFs, while a scoped diff check over touched files passed.

## Canonical State Impact

No canonical thesis or data-store state update is needed. This is a Streamlit/Plotly inspection improvement only; rollout generation, scoring, Zarr schema, and Rerun logging were not changed.
