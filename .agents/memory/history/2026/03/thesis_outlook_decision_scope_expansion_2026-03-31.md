---
id: 2026-03-31_thesis_outlook_decision_scope_expansion
date: 2026-03-31
title: "Thesis Outlook Decision Scope Expansion"
status: done
topics: [slides, typst, thesis, planning, rl]
confidence: high
canonical_updates_needed: []
files_touched:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf
artifacts:
  - /tmp/slides_scope_expand_fix4_1774961719.pdf
  - /tmp/slides_scope_expand_fix4_1774961719/contact.png
---

Task
- Integrated additional advisor-facing scope questions into the opening decision slides of the thesis outlook deck.

Method
- Expanded the opening slide sequence to include planning-target, hierarchy-scope, EFM/CORAL, candidate-rule, and discrete-vs-continuous action decisions.
- Reworked the decision-slide layout into two non-breakable columns so each decision stays on a single slide without shrinking text.
- Compiled to temporary PDFs and visually inspected the first six slides after each iteration until the pagination was stable.

Findings
- The initial content expansion caused Typst to split Decision 1 and Decision 3 across multiple pages.
- A two-column pattern with non-breakable grouped blocks fixed the pagination while keeping the added questions visible.
- The opening now explicitly surfaces:
  - EFM/EVL backbone and CORAL ablation scope
  - optimal multi-step return proxy and RL method family
  - candidate rules as masks/reward-only vs predicted feasibility
  - offline-only vs offline+online RL
  - continuous vs discrete phase-1 actions
  - VLA / semantic-global planning as a scoped hierarchical extension rather than thesis core

Verification
- `typst compile /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ /tmp/slides_scope_expand_fix4_1774961719.pdf --root /home/jandu/repos/NBV/docs`
- visual inspection of `/tmp/slides_scope_expand_fix4_1774961719/contact.png`
- copied the verified PDF to `/home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf`

Canonical State Impact
- None. This was a presentation-structure update only.
