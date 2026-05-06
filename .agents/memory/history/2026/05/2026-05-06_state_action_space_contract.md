---
id: 2026-05-06_state_action_space_contract
date: 2026-05-06
title: "State And Action Space Contract"
status: done
topics: [docs, glossary, notation, rollout, qh]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/theory/rl_planning.qmd
  - docs/typst/shared/glossary.typ
  - docs/typst/shared/symbols/rl.typ
  - docs/typst/shared/equations/rl.typ
  - docs/notation.yml
  - docs/typst/thesis_slides/slides_thesis_outlook.typ
---

Task: document ARIA-NBV state and action spaces for historic, offline,
counterfactual, geometry-rich, and oracle rollout states.

Method: added explicit rendered state variants `s_t^{hist}`, `s_t^{off}`,
`s_t^{cf0}`, `s_t^{cf+}`, and `s_t^{oracle}` to shared notation and glossary
sources; rewrote the RL planning page around actor-visible versus oracle-only
modalities; changed finite actions to candidate-table indices with
`q_t=q_{t,a_t}`; regenerated glossary artifacts.

Findings: the previous shared `hist_ego`/`hist_cf` and `state_ego`/`state_cf`
notation was too coarse for the thesis-core rollout contract. The docs now
state that all-candidate GT renders and labels are oracle-only before
selection, while the main `Q_H` actor input uses the minimal counterfactual
state and the geometry-rich state is an ablation.

Verification: `make glossary` passed. Full Quarto/Typst render and stale
notation searches were run after this debrief during the same implementation
turn.

Canonical state impact: no separate state-memory update is required; the public
theory page and generated glossary are now the durable contract surfaces.
