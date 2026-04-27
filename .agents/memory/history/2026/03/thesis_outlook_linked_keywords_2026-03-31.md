---
id: 2026-03-31_thesis_outlook_linked_keywords
date: 2026-03-31
title: "Thesis Outlook Linked Keywords"
status: done
topics: [slides, typst, links, documentation]
confidence: high
canonical_updates_needed: []
files_touched:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf
  - /tmp/thesis_link_check_cairo/page-04.png
  - /tmp/thesis_link_check_cairo/page-06.png
  - /tmp/thesis_link_check_cairo/page_more-08.png
  - /tmp/thesis_link_check_cairo/page_more-09.png
  - /tmp/thesis_link_check_cairo/page_more-12.png
---

Task:
- Add helpful external and internal links throughout the thesis-outlook slide deck using Typst `#link(...)` and the shared `#gh(...)` helper.

Method:
- Reused the existing `#gh(path)` macro from `docs/typst/shared/macros.typ`.
- Added a small set of URL constants to the slide file for Hestia, Gymnasium, Stable-Baselines3, and simulator candidates.
- Inserted internal GitHub links on the blocker and implemented-evidence slides, plus external links for Hestia and simulator/tooling names.
- Recompiled the deck and visually inspected the affected pages with `pdftocairo`.

Findings:
- The deck now links key implemented RL and rollout surfaces directly to their source files.
- External links highlight Hestia and the simulator options without overwhelming the deck.
- Inline file links on the blocker slide wrap onto a second line, but remain legible and do not break the layout.

Verification:
- `cd /home/jandu/repos/NBV/docs && typst compile typst/slides/slides_thesis_outlook.typ typst/slides/slides_thesis_outlook.pdf --root .`
- visual inspection of pages 3, 5, 7, 8, and 11 via `pdftocairo`

Canonical state impact:
- None. This is a presentation usability improvement only.
