---
id: 2026-03-31_thesis_outlook_decision_quote_blocks
date: 2026-03-31
title: "Thesis Outlook Decision Quote Blocks"
status: done
topics: [slides, typst, advisor-meeting, formatting]
confidence: high
canonical_updates_needed: []
files_touched:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.pdf
  - /tmp/slides_quotes_1774960683.pdf
  - /tmp/slides_quotes_fresh_1774960683/page-03.png
  - /tmp/slides_quotes_fresh_1774960683/page-04.png
  - /tmp/slides_quotes_fresh_1774960683/page-05.png
---

Task:
- Make the decision slides present explicit `Questions` and `Recommendation` callouts using `#quote-block[...]`.

Method:
- Reworked all three decision slides to keep the `Topic` and `Options` blocks, then added consistent `Questions` and `Recommendation` quote blocks beneath them.
- Rendered the updated decision slides from a fresh PDF to avoid stale-output confusion.

Findings:
- The decision slides now read more clearly as advisor prompts:
  - Decision 1 asks about VIN effort and whether the goal is trust-building or a stronger future reward/critic model.
  - Decision 2 asks about modality contract and counterfactual SLAM-PC emulation.
  - Decision 3 asks about offline-only versus offline+online RL and whether modality/return-contract work should come first.

Verification:
- `typst compile /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ /tmp/slides_quotes_1774960683.pdf --root /home/jandu/repos/NBV/docs`
- `pdftoppm -f 2 -l 5 -png /tmp/slides_quotes_1774960683.pdf /tmp/slides_quotes_fresh_1774960683/page`
- visual inspection of `/tmp/slides_quotes_fresh_1774960683/page-03.png`, `/tmp/slides_quotes_fresh_1774960683/page-04.png`, and `/tmp/slides_quotes_fresh_1774960683/page-05.png`

Canonical state impact:
- None. This is presentation-structure work only.
