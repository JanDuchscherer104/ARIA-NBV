---
id: 2026-03-31_thesis_outlook_rl_implementation_slides
date: 2026-03-31
title: "Thesis Outlook RL Implementation Slides"
status: done
topics: [slides, rl, streamlit, typst]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/slides/slides_thesis_outlook.typ
artifacts:
  - docs/typst/slides/slides_thesis_outlook.pdf
---

Task
- Update the thesis outlook deck to reflect the actual RL implementation in `aria_nbv`, with emphasis on the Streamlit RL inspector and the current short-horizon shell-based env.
- Resolve the remaining slide TODOs around supervisor-priority decisions and simulator strategy.

Method
- Read the RL page, env/config, PPO wrapper, counterfactual rollout utilities, oracle scorer, and targeted tests.
- Revised the RL slides to cover the implemented rollout scaffold, env/action/reward contract, Streamlit RL inspector surface, and the current theory boundary.
- Added the requested `docs/figures/app/multi-step/T3-greedy-rl.png` figure to the deck.
- Pulled `ideas.qmd`, the project state docs, and `.agents/tmp/ChatGPT-reports/simulators.md` to replace the remaining TODO placeholders with repo-grounded slide content.

Findings
- The current RL implementation is intentionally narrow: discrete shell actions, minimal geometry-first observations, oracle-RRI rewards on mesh-backed snippets, SB3 PPO wiring with `gamma = 0.1`, and evaluation-first diagnostics in Streamlit.
- The RL inspector exposes shell preview, episode replay, and seeded policy comparison, while PPO training remains outside the app.
- The simulator decision is best framed as a data-contract problem. The highest-signal thesis path remains ASE-native geometry-first RL first, with public simulator work deferred unless the existing contract saturates.

Verification
- Ran `cd docs && typst compile typst/slides/slides_thesis_outlook.typ --root .`
- Rendered the updated PDF pages to PNG with `pdftoppm` for visual sanity checks and tightened two overflowing slides until the deck paginated cleanly.
- Verified that `slides_thesis_outlook.typ` no longer contains `TODO` markers.

Canonical state impact
- None. This was a presentation-layer update only.
