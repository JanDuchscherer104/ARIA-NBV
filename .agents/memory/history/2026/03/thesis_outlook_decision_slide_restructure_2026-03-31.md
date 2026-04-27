---
id: 2026-03-31_thesis_outlook_decision_slide_restructure
date: 2026-03-31
title: "Thesis Outlook Decision Slide Restructure"
status: done
topics: [slides, typst, advisor-meeting, scope, rl, vin]
confidence: high
canonical_updates_needed: []
files_touched:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf
  - /tmp/slides_decisions_1774959761.pdf
  - /tmp/slides_decisions_1774959761/page-02.png
  - /tmp/slides_decisions_1774959761/page-03.png
  - /tmp/slides_decisions_1774959761/page-04.png
  - /tmp/slides_decisions_1774959761/page-05.png
assumptions:
  - The advisor-facing opening should prioritize explicit decisions over blocker/background slides.
  - It is acceptable to keep the existing theory and implementation slides after the opening, as long as slides 2-4 clearly frame the present decisions.
---

Task:
- Rework slides 2 onward so they clearly present current design decisions, their options, and a concrete recommendation, based on `ideas.qmd`.

Method:
- Replaced the previous generic `Scope Anchors` / blocker framing with three explicit decision slides:
  - VIN / ablation budget
  - counterfactual state contract
  - RL regime + data collection
- Kept the opening `Top 5 Decisions` slide, but made it feed directly into the new slide sequence.
- Verified output from a fresh temporary PDF and copied that verified build to the canonical deck path because direct recompilation to the canonical PDF path lagged behind the fresh output in this environment.

Findings:
- The deck now surfaces the user-requested decisions much more directly:
  - how much single-step VIN / ablation work to do before moving to multi-step
  - whether historical and counterfactual states should stay geometry-first or require RGB synthesis / 3DGS
  - whether offline-only RL should come first, with online RL deferred until simulator access and fast surrogate reward exist
- The counterfactual-state slide now explicitly asks whether counterfactual SLAM PC emulation is needed, or whether semi-dense historical points plus dense candidate points are sufficient.
- The new structure is materially more concise and better aligned with the high-priority questions in `docs/contents/ideas.qmd`.

Verification:
- `typst compile /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ /tmp/slides_decisions_1774959761.pdf --root /home/jandu/repos/NBV/docs`
- `pdftoppm -f 2 -l 5 -png /tmp/slides_decisions_1774959761.pdf /tmp/slides_decisions_1774959761/page`
- visual inspection of `/tmp/slides_decisions_1774959761/page-02.png` through `/tmp/slides_decisions_1774959761/page-05.png`
- `cp /tmp/slides_decisions_1774959761.pdf /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf`
- `pdftotext -f 2 -l 5 /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf -`

Canonical state impact:
- None. This work clarifies deck structure and presentation, but does not change canonical project truth.
