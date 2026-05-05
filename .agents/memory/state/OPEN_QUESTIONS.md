---
id: open_questions
updated: 2026-05-05
scope: repo
owner: jan
status: active
tags: [research, nbv, vin, training]
---

# Open Questions

## Advisor-Facing Deferred Decisions
- What exact held-out test split should be used for the final full-scale ASE GT-mesh experiment: train/val/test proportions, scene identities, and whether any scenes are reserved for qualitative-only failure cases?
- What is the exact pass/fail threshold for the M5 Q_H success bar: absolute cumulative target-RRI gain, statistical test, effect size, number of scenes/snippets, and tolerated scene-level regression?
- What is the minimum advisor-acceptable fallback if full 100-scene / 4,608-snippet coverage is blocked by data, storage, or LRZ failures after the small trusted subset passes?

## Target and Matching Details
- What additional target matching criterion `X` is needed beyond compatible class, OBB IoU, visibility/support, projected area, and semidense/EVL point support?
- How should ambiguous multi-object matches be resolved when several predicted OBBs overlap one GT target or one predicted OBB overlaps several GT targets?
- Which observed target support signals should enter V1 first: projected area, semidense point count, EVL voxel evidence, OBB confidence, class prior, or a compact crop descriptor?

## Q_H and Offline RL Details
- What exact horizon values, discount/normalization convention, target-RRI clipping, and return target definition should define $Q_H$?
- Should Q_H use only cumulative target RRI in the main run, or should path length, motion rules, validity, and diversity penalties receive a small first ablation?
- What is the exact IQL scope if Q_H is stable: report-only ablation, full comparison, or defer to future work?
- When, if ever, should an online Gymnasium/SB3 baseline be created after the fitted Double-Q path exists?

## Storage, Scale, and Reporting
- What exact Zarr group layout should be used for rollouts, Q_H training fields, target crops, candidate masks, lineage, and optional heavy diagnostics?
- How much detail must be embedded in target mesh crops to preserve fine supervision without causing avoidable storage blow-up?
- Which LRZ storage target and retention policy should own active shards, final Zarr stores, generated reports, and non-committed Rerun recordings?
- Which CI/pre-commit checks are mandatory before full-scale generation without blocking proposal and M1 groundwork?

## Representation and Ablation Questions
- How should stage dependence and label-distribution drift be handled: stage-aware features, dynamic binning, calibration analysis, or some combination?
- Which candidate-specific signals should be prioritized for target-conditioned scoring and Q_H: directional observability, target-conditioned local reads, stronger projection encoders, or transformer-style query-centric fusion?
- Which current VIN components actually help enough to keep: surface reconstruction input, modified CORAL, auxiliary Huber loss, pretrained projection encoders?
- If semantic-global planning is pursued later, what grounded world-memory schema and verifier / replanner loop would be required?

## Recently Locked Decisions
- The thesis/system name is ARIA-NBV.
- The thesis core stays within ASE/EFM and the ASE mesh/oracle counterfactual rollout loop; Habitat, Isaac, online simulators, SceneScript, and real-device guidance are stretch or bridge work.
- V0 uses GT OBB input as sanity/upper-bound; V1 OBS-SEL / PRED-Q / GT-EVAL is mandatory for the main target-conditioned scorer and Q_H result.
- Invalidity is represented by hard masks and explicit reason codes, not by low RRI labels.
- Fitted Double-Q / Q_H over finite candidates is a hard M5 deliverable, and IQL is only a second offline-RL ablation after Q_H is stable.
- Full 100 GT-mesh ASE scenes / 4,608 snippet windows are the final scale bar after small-subset correctness and LRZ/Zarr gates pass.
