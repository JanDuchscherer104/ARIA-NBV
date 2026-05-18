---
id: 2026-05-18_full_rri_qh_contract_repair
date: 2026-05-18
title: "Full RRI QH Contract Repair"
status: done
topics: [rri, rollouts, q-learning, zarr, agents-db]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/PROJECT_STATE.md
  - .agents/memory/state/DECISIONS.md
  - .agents/memory/state/OPEN_QUESTIONS.md
files_touched:
  - aria_nbv/aria_nbv/rri_metrics/eval_pointclouds.py
  - aria_nbv/aria_nbv/rri_metrics/oracle_rri.py
  - aria_nbv/aria_nbv/pose_generation/counterfactuals.py
  - aria_nbv/aria_nbv/pose_generation/target_counterfactuals.py
  - aria_nbv/aria_nbv/rollouts/zarr_store.py
  - scripts/agents_db.py
  - .agents/issues.toml
  - .agents/todos.toml
---

## Task

Implemented the full RRI/Q_H contract repair for the ASE-depth-root oracle stream, root-normalized rollout rewards, target-local crop scoring, target eval crop persistence, and agents DB collision guard.

## Method

Root eval point clouds now carry explicit root time, trajectory index, and frame index, and observed-prefix ASE depth selection is filtered by timestamp rather than nearest XYZ pose. Target RRI scoring crops current eval points and candidate points to the matched GT OBB before target-local fusion/capping, while scene diagnostics remain separate. Oracle unions use source-balanced capping so saturated roots cannot erase all candidate evidence.

The rollout Zarr schema is now `0.7-root-gain-target-crops`. Candidate tables persist `target_root_gain` and `scene_root_gain` as reward fields, state-relative `target_rri` and `scene_rri` as diagnostics, and point-mesh/log-gain diagnostics. The `q_h/td_reward` field is the training reward with `reward_metric="target_root_gain"` and `return_semantics="cumulative_target_root_gain"`, while `td_reward_target_rri` remains diagnostic. The new `target_eval_crops/` group stores fixed-length oracle/eval-only current and candidate crop payloads with masks, lengths, source roles, and crop policy metadata.

## Backlog And Memory

The active RRI issue now uses `issue-031`, preserving the resolved LitKG `issue-029`. Q_H evidence and model todos now describe cumulative root-normalized target gain as the objective and target RRI as a diagnostic. The agents DB validator rejects reuse of active IDs that already exist in resolved records.

## Verification

Passed `cd aria_nbv && uv run pytest tests/rri_metrics tests/pose_generation/test_counterfactuals.py tests/rollouts/test_zarr_store.py -q` with 83 tests. Passed `cd aria_nbv && uv run pytest tests/agent_memory/test_agents_db.py -q`. Passed `cd aria_nbv && uv run ruff check aria_nbv/rri_metrics aria_nbv/pose_generation aria_nbv/rollouts tests/rri_metrics tests/rollouts tests/pose_generation/test_counterfactuals.py` and `cd aria_nbv && uv run ruff check tests/agent_memory/test_agents_db.py`. Passed `make agents-db AGENTS_ARGS='validate' && make agents-db` and `make check-agent-memory`. Passed `make qmd-frontmatter-check`, one-by-one Quarto renders for `docs/contents/thesis/questions.qmd`, `docs/contents/thesis/roadmap.qmd`, and `docs/contents/theory/rl_planning.qmd`, `cd docs && typst compile typst/thesis/proposal.typ --root .`, and `cd docs && quarto check`.

## Canonical State Impact

Canonical state now treats ASE GT RGB depth as the default oracle root-eval stream, target-root-gain as the rollout/Q_H reward, target RRI as diagnostic compatibility, and target eval crops as oracle/eval-only audit payloads.
