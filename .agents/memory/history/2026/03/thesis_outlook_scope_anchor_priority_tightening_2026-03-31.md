---
id: 2026-03-31_thesis_outlook_scope_anchor_priority_tightening
date: 2026-03-31
title: "Thesis Outlook Scope Anchor Priority Tightening"
status: done
topics: [slides, typst, advisor-meeting, scope, priorities]
confidence: high
canonical_updates_needed: []
files_touched:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf
  - /tmp/slides_thesis_outlook_fresh_1774958951.pdf
  - /tmp/thesis_scope_fresh_1774958951/page-02.png
  - /tmp/thesis_scope_fresh_1774958951/page-03.png
---

Task:
- Tighten the advisor-facing opening of the thesis-outlook deck so it better reflects `ideas.qmd`, especially the decision between more single-step ablations and moving to multi-step planning.

Method:
- Reworked slide 1 into five explicit meeting decisions.
- Rewrote the scope-anchor slide so the recommended thesis core is separated from branches to defer.
- Shortened the wording on the opening slides to reduce text density while preserving the main decisions.
- Recompiled to both the canonical PDF path and a fresh temporary PDF to avoid stale-render confusion.

Findings:
- The opening now makes the priority order explicit: compute, access, thesis core, ablation budget, and state contract.
- The scope slide now matches `ideas.qmd` more closely by making the current ASE/mesh-backed ecosystem and a non-myopic baseline the core recommendation, while treating heavy VIN-v4 search and RGB synthesis as deferred.
- The new slide pair is materially more concise and should be easier to discuss live with an advisor.

Verification:
- `typst compile /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf --root /home/jandu/repos/NBV/docs`
- `typst compile /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ /tmp/slides_thesis_outlook_fresh_1774958951.pdf --root /home/jandu/repos/NBV/docs`
- `pdftotext -f 2 -l 3 /tmp/slides_thesis_outlook_fresh_1774958951.pdf -`
- visual inspection of `/tmp/thesis_scope_fresh_1774958951/page-02.png` and `/tmp/thesis_scope_fresh_1774958951/page-03.png`

Canonical state impact:
- None. This is deck phrasing and prioritization only.
