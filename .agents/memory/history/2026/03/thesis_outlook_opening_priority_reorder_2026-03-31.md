---
id: 2026-03-31_thesis_outlook_opening_priority_reorder
date: 2026-03-31
title: "Thesis Outlook Opening Priority Reorder"
status: done
topics: [slides, typst, priorities, advisor-meeting]
confidence: high
canonical_updates_needed: []
files_touched:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf
  - /tmp/thesis_priority_opening/page-02.png
  - /tmp/thesis_priority_opening/page-03.png
assumptions:
  - The advisor deck should foreground the operational asks and scope-lock questions from `ideas.qmd` before any theory or implementation detail.
---

Task:
- Reorder the first two slides so they reflect the explicit priority order requested by the user: cluster/workstation, Aria Gen2 + simulator application, thesis scope, then the simulator/state-contract decision.

Method:
- Replaced the generic decision agenda slide with a four-item opening slide matching the requested order.
- Rewrote the recommendation slide so the left block covers access/infrastructure and the right block covers scope/modeling contract.
- Preserved the existing downstream deck structure.

Findings:
- The new opening now foregrounds the operational asks before the research framing.
- The simulator question is now presented as two separate ideas:
  - ask for access now
  - do not make phase 1 depend on simulator-driven counterfactual modalities

Verification:
- `cd /home/jandu/repos/NBV/docs && typst compile typst/slides/slides_thesis_outlook.typ typst/slides/slides_thesis_outlook.pdf --root .`
- visual inspection of `/tmp/thesis_priority_opening/page-02.png`
- visual inspection of `/tmp/thesis_priority_opening/page-03.png`

Canonical state impact:
- None. This is advisor-deck prioritization only.
