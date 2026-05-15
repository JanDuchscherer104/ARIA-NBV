---
id: 2026-05-13_advisor_distillation_geometric_ml_remediation
date: 2026-05-13
title: "Advisor Distillation Geometric ML Remediation"
status: done
topics: [advisor-handout, typst, bibliography, literature, geometric-ml]
confidence: high
canonical_updates_needed: []
files_touched:
  - docs/typst/thesis/advisor_distillation.typ
  - docs/references.bib
  - docs/literature/sources.jsonl
---

## Task

Implemented the advisor handout remediation requested on 2026-05-13: make the planned `Q_H` architecture read as a principled geometric-ML design, add the requested review references to the bibliography/source registry, and keep simulator/backbone work as bridge context rather than thesis core.

The triggering review argued that the handout was scientifically strong but still risked reading like "architecture soup". It endorsed the conditional success rule, leakage-aware policy comparison, RQ dependency chain, and finite-candidate `Q_H` boundary, then recommended making the architecture derive from task symmetries rather than from a list of possible modules.

## Review Pointers Preserved

- Keep the central conditional claim: first measure oracle-lookahead headroom. If `Delta_"look"` is positive, learned actor-visible `Q_H` must recover headroom over the learned myopic target scorer under oracle re-evaluation. If headroom is near zero, report a negative planning result for the evaluated candidate distribution, horizon, branch factor, target set, and split instead of rescuing the story with more architecture.
- Preserve the RQ chain: objective/metrics, target matching, candidate and rollout support, lookahead/`Q_H`, scaling, and online/continuous escalation. Later questions must not rescue failed earlier contracts.
- Make architecture a geometric-ML contract: finite candidate tables require permutation-equivariant per-candidate outputs; camera, target, and history poses require relative/local-frame geometry; directional visibility lives on `S^2`; the selected target is a query/gauge, not metadata.
- Treat `Q_H` as a small relational decision model on top of the expensive ASE/EVL/oracle substrate, not as a dense 3D backbone rewrite. The intended flow is calibrated myopic target scorer, candidate tokenization, permutation-equivariant candidate interaction, and residual dueling value head.
- Use equivariance surgically. EGNN-style candidate graphs are useful ablations because they encode translation/rotation/reflection/permutation structure without higher-order tensor features. SE(3)-Transformer/e3nn-style tensor machinery is conceptually relevant but too heavy for thesis core.
- Make the directional-memory question explicit: does target-local directional observability improve non-myopic target-RRI recovery beyond target-relative pose and current frustum evidence? Second-moment `S^2` memory is the cheap default; spherical histograms or low-order spherical harmonics are principled ablations.
- Evaluate rollout generation by support coverage, not only by label count. Important data products include all-candidate one-step labels, paired greedy/lookahead roots, stochastic support traces, invalid near-misses, target strata, candidate-strategy histograms, and successor-table availability.
- Keep candidate generation experimental, not incidental. Strategy provenance, target-bearing angle, path increment, predicted support, invalid reason, and per-strategy RRI histograms should explain whether no-headroom results come from the objective or from a too-myopic candidate distribution.
- Use literature roles narrowly. VIN-NBV remains the objective/ranking precedent; GenNBV is a diversity/state-embedding and continuous-control contrast; SCONE and MACARONS are coverage/online contrasts; Hestia informs post-`Q_H` hierarchy and directional visibility; ProcTHOR, Habitat, Isaac, MinkowskiEngine, PTv3, and KPConv are scale or substrate/backbone bridge references, not thesis-core commitments.
- Keep target-local ROI/crop features leakage-safe. Actor-visible predicted-OBB crops are strong target grounding candidates, but GT crops remain labels/evaluation only.
- Keep sparse/point backbones and full simulator paths as scaling ablations. They answer whether a stronger accumulated-geometry encoder or external substrate preserves target-RRI supervision after simpler finite-candidate baselines are controlled.
- Keep privileged-teacher distillation promising but non-core. Oracle-only fields may supervise a student in later ablations, but must not enter V1 actor inputs.
- Reduce thesis contributions to four durable claims: leakage-safe target-RRI protocol, headroom-measured finite-candidate planning, geometry-aware residual `Q_H`, and support-aware rollout data generation.

## Method

Used the repo-local docs and Typst authoring guidance. The handout gained a geometric inductive-bias table, an architecture role ladder, and a support-coverage contract. Bibliography/source registry additions were limited to the review's source families and kept existing citation keys stable.

The implementation chose the "full theory table" option but kept math depth controlled: no new graph, simulator, or distillation equations were added. The handout now explains the planned value model through a theory table and role ladder while reusing the existing RPE and `S^2` equations. Simulator/backbone references were added to the bibliography and source registry, but cited only in one compact bridge/scale row.

## Implementation Boundaries

- Edited only `docs/typst/thesis/advisor_distillation.typ`, `docs/references.bib`, `docs/literature/sources.jsonl`, and this debrief.
- Did not change proposal, roadmap, questions, shared equation definitions, generated PDFs, or Python code.
- Did not promote external simulators, full point/sparse backbones, SE(3)-equivariant tensor networks, distributional heads, privileged distillation, or continuous actor-critic into thesis-core requirements.
- Preserved existing citation keys where already used: `zhou2023query`, `DeepSets-zaheer2017`, `SetTransformer-lee2019`, `SCONE-guedon2022`, `MACARONS-guedon2023`, and `Hestia-lu2026`.

## Verification

Verification performed:

- `cd docs && typst compile typst/thesis/advisor_distillation.typ /tmp/advisor-distillation-gdl.pdf --root .` passed.
- Rendered pages 5-18 with `.agents/skills/typst-authoring/scripts/render_png.sh`; visually inspected the RQ/value-model pages, geometric-bias table, architecture ladder, support-coverage table, and literature ledger.
- `.agents/skills/typst-authoring/scripts/hygiene_checks.sh --strict docs/typst/thesis/advisor_distillation.typ docs/typst/shared` passed blocking checks. Remaining matches were existing advisory glossary/raw-TeX and shared-symbol review prompts.
- `docs/literature/sources.jsonl` validated as 36 JSONL records.
- KG checks: the simulator/backbone bridge-scope claim was supported by canonical roadmap/state. The GDL and QCNet-specific checks remained unverifiable because newly added paper records have not yet been ingested and paper-node source-path coverage is a known KG limitation; no contradiction was returned.
- `make check-agent-memory` passed.
- Scoped `git diff --check` passed for the touched source files.

## Canonical State Impact

No canonical state update is needed. The thesis spine remains target-specific RRI, actor-visible target conditioning, myopic scorer control, oracle-lookahead headroom, and residual finite-candidate `Q_H` recovery. External simulators, sparse/point backbones, and full equivariant models remain bridge or ablation work.

## Follow-Up Pointers

- If these new bibliography/source records should become KG-supported claims, refresh or ingest the literature registry so GDL, EGNN, SE(3)-Transformer, simulator, and backbone sources are available to claim checks.
- If the advisor asks for implementation detail, keep the first model small: myopic scorer plus residual dueling `Q_H` with MLP/DeepSets/Set-Transformer controls before QCNet RPE, EGNN, SH memory, or distillation.
- If no-headroom evidence appears, diagnose target matching, candidate support, and supervision scale before adding architecture.
