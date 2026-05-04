---
id: 2026-04-30_litkg_semantic_index_argtopk_plan
date: 2026-04-30
title: "litkg Semantic Index and ArgTopK Rollout Alignment"
status: done
topics: [litkg, semantic-scholar, thesis, planning, docs]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/skills/aria-nbv-context/scripts/nbv_context_index.sh
  - .agents/skills/aria-nbv-context/scripts/nbv_literature_index.sh
  - scripts/kg/ingest_papers.sh
  - docs/contents/thesis/questions.qmd
artifacts:
  - docs/_generated/context/source_index.md
  - docs/_generated/context/literature_index.md
  - .agents/kg/generated/literature/registry.jsonl
---

# Debrief

## Task

Refresh ARIA-NBV's internal literature routing from the litkg-rs Semantic
Scholar registry and add minimal thesis prose aligning the next multi-step NBV
step with the current implementation and paper-backed plan.

## Method

The litkg-rs wrapper was updated to match the current nested CLI commands
(`ingest`, `lit`, `kg`, and `s2`). `make kg-semantic-enrich` was run through the
wrapper, producing a local registry with 56 records and 24 Semantic Scholar
enriched records. The lightweight context generators were corrected from the
stale `literature/` path to `docs/literature/`, then `make context` regenerated
the source and literature indexes.

## Outputs

The internal literature index now includes the litkg registry summary, all
Semantic Scholar-enriched records with links and citation counts, remaining
registry records without Semantic Scholar metadata, and the local TeX paper
families. The source index now reports 9 literature families, 101 TeX files,
and 8 bibliography files.

The thesis questions page now defines the bounded rollout sequence
`ArgTopK -> ArgTop1_1 -> ... -> ArgTop1_H` as a finite-horizon selector over the
existing counterfactual rollout implementation. The text keeps the first
claim at bounded oracle-RRI lookahead versus greedy and cites VIN-NBV, GenNBV,
and Hestia as the relevant paper anchors.

## Verification

- `make kg-semantic-enrich`
- `make context`
- `make qmd-frontmatter-check`
- `cd docs && quarto render contents/thesis/questions.qmd`
- `python3 /home/jd/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/aria-nbv-context`
- `make check-agent-memory`
- `git diff --check -- .agents/skills/aria-nbv-context/scripts/nbv_context_index.sh .agents/skills/aria-nbv-context/scripts/nbv_literature_index.sh scripts/kg/ingest_papers.sh docs/contents/thesis/questions.qmd`

## Canonical State Impact

No canonical state update was needed. Existing project state already locks the
first non-myopic milestone as bounded oracle-RRI lookahead versus one-step
greedy under equal budget, with continuous control and value/Q heads deferred
behind evidence gates.
