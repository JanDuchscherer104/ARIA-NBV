---
id: 2026-04-29_thesis_roadmap_research_questions
date: 2026-04-29
title: "Thesis Roadmap And Research Questions Docs"
status: done
topics: [docs, thesis, roadmap, research-questions, kg]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/roadmap.qmd
  - docs/contents/questions.qmd
---

## Task

Replaced the stale roadmap and research-question pages with a full-time
master's thesis plan through 2026-09-30 and a target-conditioned NBV research
question structure.

## Method

Used the current paper framing, canonical state, and transcript-derived plan to
anchor the public docs around target-conditioned RRI, entity-aware supervision,
and multi-step rollouts. Added KG-friendly writing rules with stable anchors,
internal links, and bibliography-backed citations.

## Findings

The updated roadmap makes full continuous RL, VLM planning, simulator-backed
online training, and real-device deployment explicit stretch work unless the
August evidence gate justifies them. The research-question page now organizes
the thesis around objective design, target encoding, candidates, model scoring,
planning, feasibility, scaling, and extension boundaries.

## Verification

- `cd docs && quarto render contents/roadmap.qmd`
- `cd docs && quarto render contents/questions.qmd`
- `scripts/nbv_qmd_outline.sh --compact`
- `rg` scan of touched pages for stale terms: `Seminar`, `hesita`,
  `coral_intergarion`, `JanDuchscherer104/NBV`, and related old typo markers

The Quarto renders succeeded. Quarto still emits the existing site-wide warning
for `contents/resources/agent_scaffold/index.qmd`, which was not introduced by
these pages.

## Canonical State Impact

No canonical state doc was changed. The new public docs reflect the already
recommended active direction in `.agents/memory/state/PROJECT_STATE.md`.
