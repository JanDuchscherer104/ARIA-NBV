---
id: 2026-05-08_advisor_distillation_handout
date: 2026-05-08
title: "Advisor Distillation Handout"
status: done
topics: [thesis, typst, advisor-facing, proposal]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
---

## Task

Created a separate Typst advisor handout for the current ARIA-NBV thesis
contract. The handout is a single Typst source that imports the existing thesis
proposal template, proposal style, metadata, shared notation, and equations, but
does not include the existing proposal section files or any Mermaid/PNG assets.

## Method

The handout distills the current state into current substrate versus planned
thesis core, a formal model, three testable research questions, a literature
adoption ledger, leakage-aware policy comparison, symbolic final scale
thresholds, risk/deferred-decision wording, and a Typst-native Gantt from
2026-04-29 to 2026-09-30.

It includes the requested clarifications for target-match acceptance thresholds,
V1 OBS-SEL / PRED-Q / GT-EVAL visibility, Double-Q terminal flag semantics, and
candidate-table regeneration after a selected action.

The follow-up expansion digested
`.agents/work/architecture/A01-architecture-inspiration.md` and shifted the
handout from compactness toward elegance: equations are numbered, formulas are
split to avoid number/content collisions, and the architecture section now
records the QH-VIN-GNN direction: residual finite-horizon Q on top of the
target-conditioned one-step scorer, EVL scene memory, target ROI reads,
directional `S^2` memory, actor-visible belief renders, permutation-equivariant
set/GNN candidate reasoning, dueling/distributional heads, and privileged
teacher/student render use only as training/evaluation scaffolding.

## Verification

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation.pdf --root .`
- `.agents/skills/typst-authoring/scripts/render_png.sh -i docs/typst/thesis/advisor_distillation.typ -o /tmp/advisor-distillation-pages --root docs --pages 4-9 --ppi 220`
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ`
- `make kg-claim-check` supported the implemented-substrate, V1 actor-visible protocol, GT-label/evaluation boundary, and headroom-gated `Q_H` claims.
- The broad literature-adoption-role claim returned `unverifiable`; the returned evidence still pointed at the roadmap/advisor-contract sources, so this was treated as a KG broad-claim limitation rather than a handout content blocker.
- A follow-up claim check for privileged teacher/dense-render usage returned
  `unverifiable` because canonical sources support the GT-not-actor-visible
  boundary but do not yet establish privileged teacher wording as canonical
  thesis scope. The handout therefore frames this as a proposed architecture
  constraint, not an implemented fact.

## Output

The compiled PDF has 12 total pages: title, AI transparency, contents, six body
pages, and bibliography pages. The body intentionally exceeds the earlier compact
cap after the 2026-05-08 follow-up preference for elegance over compactness.

## Canonical State Impact

No canonical state updates are needed. This is a derived advisor-facing handout
aligned to the existing roadmap/questions/current-memory thesis direction.
