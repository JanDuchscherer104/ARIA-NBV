---
id: 2026-05-08_target_architecture_mermaid_refresh
date: 2026-05-08
title: "Target Architecture Mermaid Refresh"
status: done
topics: [mermaid, thesis, qh, architecture, proposal]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/figures/proposal_system_flow.mmd
  - docs/typst/thesis/figures/qh_actor_oracle_contract.mmd
  - docs/typst/thesis/figures/qh_vin_gnn_architecture.mmd
  - docs/typst/thesis/figures/qh_rollout_replay_doubleq.mmd
  - docs/typst/thesis/figures/qh_teacher_student_render_path.mmd
  - docs/typst/thesis/figures/qh_directional_memory.mmd
assumptions:
  - "The first pass should add thesis figure sources and rendered review artifacts, not a new Quarto page."
  - "The proposal continues to include proposal_system_flow.png at the existing path."
---

## Task

Implemented the 2026-05-08 target architecture Mermaid refresh for the thesis
proposal surface. The work distilled `docs/typst/thesis/proposal.typ`,
`.agents/work/architecture/A01-architecture-inspiration.md`, and
`.agents/work/architecture/A02-architecture-inspiration-diagrams.md` into a
canonical diagram pack for the finite-candidate target-conditioned `Q_H`
architecture.

## Method

Used the ARIA Mermaid style guide and symbol map. Replaced the existing proposal
system-flow source with a frontmatter-based KaTeX Mermaid flowchart, refreshed
the proposal PNG, and added focused `Q_H` architecture, actor/oracle contract,
rollout/replay Double-Q, teacher/student, and directional-memory diagrams under
`docs/typst/thesis/figures/`.

The worktree already contained unrelated guidance, tool, and advisor handout
changes before this task. This debrief covers only the thesis figure refresh.

## Outputs

The refreshed proposal figure now shows actor-visible state, finite candidates,
hard masks/reasons, target RRI oracle supervision, one-step scorer, oracle
lookahead headroom, masked `Q_H`, oracle re-evaluation, and M6+ bridge work.

The new architecture diagrams visualize the leakage boundary, QH-VIN-GNN /
spatial candidate-set network, replay and masked Double-Q training flow,
privileged dense-render teacher path, and low-order `S^2` directional memory.

## Verification

- `aria_nbv/.venv/bin/python tools/mermaid/scripts/aria_mermaid_lint.py docs/typst/thesis/figures/*.mmd`
- `PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome npx -y @mermaid-js/mermaid-cli ...`
- `typst compile typst/thesis/proposal.typ --root .`
- `./.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/proposal.typ -o /tmp/aria-proposal-pages-final --root docs --ppi 300 --pages 8-9`

On 2026-05-08, a follow-up visual QA pass fixed Mermaid/KaTeX row-break
escaping so multiline labels render as equations instead of collapsed or raw
KaTeX text. The proposal system-flow figure was also compacted so it fits on
one proposal page after the rendered-page inspection exposed a tall split
layout.

Global `mmdc` was not installed, so rendering used the repo-documented `npx`
Mermaid CLI path with the system Chrome executable.

## Canonical State Impact

No canonical state update is needed. The diagrams are presentation artifacts
for the current thesis direction already owned by roadmap, questions, proposal,
and project state.
