---
id: 2026-05-05_math_heavy_glossary_lookup
date: 2026-05-05
title: "Math-Heavy Tiered Glossary Lookup"
status: done
topics: [glossary, thesis, mdp, notation, docs]
confidence: high
canonical_updates_needed:
  - .agents/memory/state/DECISIONS.md
files_touched:
  - scripts/glossary_build.py
  - docs/typst/shared/glossary.typ
  - docs/notation.yml
  - docs/styles.css
  - docs/contents/glossary.qmd
  - docs/contents/theory/rl_planning.qmd
  - docs/contents/thesis/roadmap.qmd
  - .agents/memory/state/DECISIONS.md
---

## Task

Convert the public glossary into a tiered thesis math lookup surface, with core
ARIA-NBV MDP/RRI symbols and formulas visible before supporting and background
terms.

## Method

Extended the glossary source metadata with `tier`, `lookup_rank`,
`symbol_refs`, and `equation_refs`; updated `scripts/glossary_build.py` to
validate those fields, cross-check notation references, emit the new metadata,
and render a "Core Math Lookup" table before category cards. Added the
target-conditioned NBV MDP concept family, including state, finite candidate
action sets, counterfactual transition, target-RRI reward, finite-horizon
return, `Q_H`, and hard validity masks.

Updated the shared notation registry and Typst symbol/equation modules for the
new MDP symbols and equations. Existing glossary terms were tiered so core
NBV/RRI/target/protocol terms lead, support terms remain normal cards, and
background dataset/tooling terms remain linkable but visually demoted.

## Findings

The generated glossary now reports 49 validated terms, 39 symbols, and 28
equations. The core lookup table exposes the advisor-facing formulas for
`M_NBV`, `A(s_t)`, `P_{t+1}`, `r_t^e`, `G_t^(H)`, and `Q_H(s_t, q)` without
requiring a reader to open the full RL planning page.

The reward convention was also aligned in `rl_planning.qmd`, `roadmap.qmd`, and
`DECISIONS.md`: cumulative target RRI under an equal acquisition budget is the
main thesis comparison; log-improvement, episode-normalized rewards, and scalar
motion/rule penalties are follow-up ablations.

## Verification

- `make glossary`
- `aria_nbv/.venv/bin/python -m py_compile scripts/glossary_build.py`
- `cd docs && quarto render contents/glossary.qmd`
- `cd docs && quarto render contents/theory/rl_planning.qmd`
- `cd docs && quarto render contents/thesis/roadmap.qmd`
- `git diff --check -- docs/typst/shared/glossary.typ scripts/glossary_build.py docs/notation.yml docs/contents/theory/rl_planning.qmd docs/contents/thesis/roadmap.qmd .agents/memory/state/DECISIONS.md docs/styles.css`

The combined multi-file Quarto command form treated extra paths as Pandoc
arguments in this environment, so the affected pages were rendered separately.

## Canonical State Impact

`DECISIONS.md` now records the tiered glossary policy and the thesis-core reward
baseline convention. No proposal source or GitHub issues were changed.
