---
id: 2026-03-31_thesis_outlook_theory_boundary_slide
date: 2026-03-31
title: "Thesis Outlook Theory Boundary Slide"
status: done
topics: [slides, typst, rl, theory, advisor-meeting]
confidence: high
canonical_updates_needed: []
files_touched:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf
  - /tmp/thesis_theory_outlook_2/page-06.png
---

Task:
- Add a theory-outlook slide that captures the most important RL / planning keywords and equations, and explicitly separates what is already implemented from what still needs to change.

Method:
- Inserted a new slide after the geometry-first MDP contract slide.
- Reused shared equations from `docs/typst/shared/macros.typ` rather than introducing slide-local math.
- Summarized the current repo boundary around oracle reward, low-discount return, hard validity masks, and critic/Q-function outlook.
- Recompiled and visually inspected the new slide in context.

Findings:
- The new slide states the current theory boundary clearly:
  - reward is oracle RRI now and potentially VIN surrogate later
  - hard validity masks currently handle collision / clearance / bounds
  - the first RL variant is low-discount / close-greedy
  - Q / critic should estimate cumulative quality rather than coverage
- It also makes the implementation gap explicit:
  - current repo has hard masking plus a flat invalid-action penalty
  - current repo does not yet have a VIN-backed counterfactual evaluator / critic
  - privileged-critic use of GT cues is still unresolved

Verification:
- `cd /home/jandu/repos/NBV/docs && typst compile typst/slides/slides_thesis_outlook.typ typst/slides/slides_thesis_outlook.pdf --root .`
- visual inspection of `/tmp/thesis_theory_outlook_2/page-06.png`

Canonical state impact:
- None. This is deck clarification, not a change to canonical project truth.
