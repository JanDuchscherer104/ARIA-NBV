---
id: 2026-03-31_thesis_outlook_inline_todos_resolved
date: 2026-03-31
title: "Thesis Outlook Inline TODOs Resolved"
status: done
topics: [slides, typst, advisor-meeting, prioritization]
confidence: high
canonical_updates_needed: []
files_touched:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf
  - /tmp/thesis_todo_ppm/page-02.png
  - /tmp/thesis_todo_ppm/page-03.png
  - /tmp/thesis_todo_ppm/contact.png
---

Task:
- Resolve the new inline TODOs in the thesis-outlook deck and verify that the rendered slides reflect the updated opening priorities and scope framing.

Method:
- Rewrote the first two advisor-facing slides to make the decision order explicit and concise.
- Added a scope-anchor comparison slide that separates the recommended thesis core from deferred or optional branches.
- Recompiled the deck and regenerated page images from scratch to avoid stale preview artifacts.

Findings:
- The opening now starts with the highest-priority meeting decisions: cluster/workstation, Aria Gen2 plus ASE simulator applications, thesis core, and the phase-1 state/simulator contract.
- The second slide now clearly separates the recommended thesis anchor from alternative scope anchors that should be deferred or explicitly de-scoped.
- `slides_thesis_outlook.typ` no longer contains inline `TODO` markers.

Verification:
- `typst compile /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf --root /home/jandu/repos/NBV/docs`
- `pdftoppm -f 1 -l 12 -png /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf /tmp/thesis_todo_ppm/page`
- visual inspection of `/tmp/thesis_todo_ppm/page-02.png`, `/tmp/thesis_todo_ppm/page-03.png`, and `/tmp/thesis_todo_ppm/contact.png`

Canonical state impact:
- None. This work only improves deck phrasing and ordering.
