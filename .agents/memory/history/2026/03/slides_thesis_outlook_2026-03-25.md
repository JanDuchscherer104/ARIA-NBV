---
id: 2026-03-25_slides_thesis_outlook
date: 2026-03-25
title: "Thesis Outlook Slides Debrief"
status: done
topics: [slides, thesis, docs]
confidence: high
canonical_updates_needed: []
---

# Thesis Outlook Slides Debrief

- Date: 2026-03-25
- Scope: Created and revised the master's-thesis outlook slide deck in `docs/typst/slides/slides_thesis_outlook.typ`.

## What changed

- Reused the `slides_4.typ` slide-entry pattern and theme configuration without changing shared templates.
- Rebuilt the deck into a tighter RL-focused structure:
  - highest-priority decisions
  - ranked RL directions
  - simulator / state / reward
  - offline-only RL methods
  - privileged critic + continuous policy
  - scaling inside the mesh subset
  - counterfactual RGB as a later phase
- Shifted the content emphasis from generic future directions to ranked RL options grounded in:
  - `ideas.md`
  - `.agents/tmp/gpt-report.md`
  - `.agents/tmp/deep-research-report.md`
- Kept the slide text terse and keyword-driven so the equations carry more of the technical detail.
- Updated the first content slide to foreground the explicitly requested operational decisions:
  - workstation / cluster access
  - simulator access
  - Aria Gen2 devkit
  - ASE mesh subset vs. alternative dataset
  - stable VIN vs. RL environment runner
- Extended `docs/typst/shared/macros.typ` with additional RL notation and equation helpers:
  - richer `symb.rl` entries for actions, value functions, policy symbols, advantages, and the two history forms
  - explicit `s_t^ego` and `s_t^cf` state definitions
  - `o_(t+1) = cal(G)(M_GT, x_(t+1))` and persistent-memory update notation
  - additive geometric reward with collision / motion penalties
  - planning objective for MPC / beam / CEM
  - critic targets for IQL / CQL / LEQ
  - GAE and PPO clip objectives
  - hierarchical policy notation for high-level target then local action
- The current deck now compiles to 9 pages total including the title slide.

## Verification

- `typst` was not available on `PATH`.
- Verified compilation with the documented fallback:
  - `/snap/typst/current/bin/typst compile /home/jandu/repos/NBV/docs/typst/slides/slides_thesis_outlook.typ /tmp/slides_thesis_outlook.pdf --root /home/jandu/repos/NBV/docs`

## Notes

- Kept the deck text-first and self-contained: no new data imports, no reference slide.
- This change adds a presentation artifact rather than changing project truth.
