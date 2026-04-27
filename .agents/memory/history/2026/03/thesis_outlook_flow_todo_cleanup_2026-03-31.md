---
id: 2026-03-31_thesis_outlook_flow_todo_cleanup
date: 2026-03-31
title: "Thesis Outlook Flow And TODO Cleanup"
status: done
topics: [slides, typst, thesis, advisor]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - docs/typst/slides/slides_thesis_outlook.pdf
---

## Task
Improve the advisor-deck narrative and slide ordering, keep the content concise without losing information, and cleanly resolve the three old inline TODOs.

## Method
Reordered the deck around advisor relevance: decisions, recommendations, blockers, theory, Hestia transfer, VINv4 bridge, implemented evidence, roadmap, open questions, and simulator backup. Split the evidence section into two slides so the `app/multi-step` figures occupy at least half a slide each. Replaced the old policy-comparison figure with `app/multi-step` figures only, removed the stale TODO comment, and turned the simulator backup into an explicit candidate ranking plus recommendation slide.

## Findings
The deck reads best as a short decision memo rather than as an implementation changelog. The most important cleanup outcome is that the evidence slides are now legible and comply with the requested `app/multi-step` figure constraint, while the simulator material is present but clearly demoted to lower-priority backup context.

## Verification
- `typst compile docs/typst/slides/slides_thesis_outlook.typ docs/typst/slides/slides_thesis_outlook.pdf --root docs`
- visual inspection of the full rendered deck, including the evidence and simulator slides
- source grep confirming no remaining `TODO`, `policy_comparison`, or `thesis_outlook/` figure references in the deck body

## Canonical State Impact
No canonical state docs changed. This was a presentation-structure cleanup aligned with the existing canonical priorities.
