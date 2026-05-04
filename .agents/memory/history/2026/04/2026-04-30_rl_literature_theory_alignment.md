---
id: 2026-04-30_rl_literature_theory_alignment
date: 2026-04-30
title: "RL Literature and Theory Alignment"
status: done
topics: [docs, literature, rl, rollout, thesis]
confidence: high
canonical_updates_needed: []
---

## Task

Add literature review entries and thesis-facing RL theory text conditioned on the current roadmap, research questions, and archived ideas, while keeping the thesis scope aligned with bounded oracle-RRI rollout before learned value or continuous-RL claims.

## Method

Used the scientific-writing and plan-grill workflows. Grounded the edit in `docs/contents/thesis/roadmap.qmd`, `docs/contents/thesis/questions.qmd`, `.agents/archive/docs/ideas.qmd`, the local litkg registry, local paper TeX/PDF mirrors, and temporary arXiv PDF text extracts for the missing RL papers.

## Outputs

Added bibliography and manifest entries for Trajectory Transformer, Double DQN, IQL, deep energy-based policies, Gumbel-Top-k, Next Best Sense, object-centric 3DGS NBV, and 3DGS viewpoint selection for human pose estimation. Added public literature pages for RL/rollout planning and active 3DGS/targeted NBV, plus a theory page formalizing the `ArgTopK -> ArgTop1_h` rollout ladder, reward convention, offline-RL gate, and continuous-policy boundary.

## Verification

Verification was run after the docs edits in the same working session. No canonical memory update was needed because existing project state and decisions already encode the bounded rollout first, geometry-first actor state, multi-step reward, target-RRI, and VIN evidence-gate decisions.
