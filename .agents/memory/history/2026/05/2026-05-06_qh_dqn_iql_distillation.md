---
id: 2026-05-06_qh_dqn_iql_distillation
date: 2026-05-06
title: "Q_H DQN And IQL Distillation"
status: done
topics: [docs, glossary, literature, q-learning, rl]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/contents/literature/rl_planning.qmd
  - docs/typst/shared/glossary.typ
  - docs/contents/glossary.qmd
  - docs/glossary/terms.yml
  - docs/_generated/context/glossary.jsonl
  - docs/literature/sources.jsonl
  - docs/literature/README.md
  - docs/references.bib
  - docs/literature/tex-src/arXiv-DQN/
  - docs/literature/pdf/DQN.pdf
---

## Task

Added DQN 2013 to the local literature corpus and enriched the `Q_H` glossary
entry plus the RL planning literature page with source-grounded DQN, Double DQN,
and IQL distillations.

## Method

The DQN arXiv source and PDF were fetched through the local downloader after
adding the manifest row. The RL planning page now grounds ARIA-NBV adoption in
the DQN source files (`intro.tex`, `background.tex`, `method.tex`,
`experiments.tex`), local Double DQN source, and local IQL source. The glossary
source now describes `Q_H` as a masked finite-candidate candidate-query
Transformer value function with DQN replay, Double-DQN selector/evaluator
backups, and IQL offline-support constraints.

## Verification

- `make glossary`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/literature/rl_planning.qmd`
- `cd docs && quarto render contents/glossary.qmd`
- `make kg-claim-check KG_CLAIM="ARIA-NBV Q_H is grounded in DQN replayed Q-learning, Double DQN masked selector/evaluator backups, and IQL offline support constraints for finite-candidate target-conditioned NBV"`

## Canonical State Impact

No `.agents/memory/state/` file needed a new durable thesis decision. The
durable public terminology source is `docs/typst/shared/glossary.typ`; generated
glossary outputs were refreshed from it.
