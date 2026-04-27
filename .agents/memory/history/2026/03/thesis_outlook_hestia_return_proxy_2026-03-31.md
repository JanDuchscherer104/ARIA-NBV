---
id: 2026-03-31_thesis_outlook_hestia_return_proxy
date: 2026-03-31
title: "Thesis Outlook Deck: Hestia Review Ideas and RRI Return Proxy"
status: done
topics: [slides, typst, hestia, rl, mdp]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - docs/typst/slides/slides_thesis_outlook.pdf
---

Task: incorporate the most relevant next-step ideas from `.agents/tmp/ChatGPT-reports/hesita-literature-review.md` into the thesis outlook deck, including explicit formulas for a multi-step RRI-based return proxy and a clearer statement of historical versus counterfactual MDP state modalities.

Method: extracted the report's highest-signal transferable ideas: directional observability over plain occupancy, hierarchical target-then-motion control, close-greedy reward design, and feasibility projection. Reworked the Hestia slide, replaced the geometry-first MDP slide with an explicit state-plus-return-proxy slide, and extended the roadmap slide with target-head supervision and feasibility projection. Recompiled and visually inspected the updated MDP slide and deck layout.

Findings / outputs:
- the Hestia slide now argues for borrowing the *planning decomposition* and *directional observation idea* while keeping ARIA-NBV's *RRI objective*
- the MDP slide now explicitly distinguishes logged historical modalities from geometry-only counterfactual modalities
- the deck now includes a concrete horizon-`H` return proxy based on either oracle RRI or a future VIN-predicted RRI proxy
- the rollout slide layout was compacted again to avoid an accidental page split after adding more theory content elsewhere

Verification:
- `cd docs && typst compile typst/slides/slides_thesis_outlook.typ typst/slides/slides_thesis_outlook.pdf --root .`
- `pdfinfo docs/typst/slides/slides_thesis_outlook.pdf`
- `pdftoppm -f 8 -l 8 -png docs/typst/slides/slides_thesis_outlook.pdf /tmp/thesis_hestia_page8/page`
- visual inspection of `/tmp/thesis_hestia_page8/page-08.png`

Canonical state impact: none. This was a presentation refinement pass aligned with existing project direction.
