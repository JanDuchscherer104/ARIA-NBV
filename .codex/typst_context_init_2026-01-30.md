# Typst context init (2026-01-30)

This note records the Typst paper/slides entry points and their current local dependencies, collected as part of the standard “make context + read included sections” initialization workflow.

## Commands run

- `scripts/nbv_context_index.sh` → updates `.codex/context_sources_index.md`
- `make context` → updates `.codex/codex_make_context.md`

## Paper (Typst)

**Entry point:** `docs/typst/paper/main.typ`

**Local imports in entry:**

- `docs/typst/paper/charged_ieee_local.typ` (imported as `ieee`)
- `docs/typst/shared/macros.typ`

**Included section files (22)** (from `#include` in `docs/typst/paper/main.typ`):

- `docs/typst/paper/sections/01-introduction.typ` — *Introduction*
- `docs/typst/paper/sections/02-related-work.typ` — *Related Work*
- `docs/typst/paper/sections/03-problem-formulation.typ` — *Problem Formulation* (`<sec:problem>`)
- `docs/typst/paper/sections/04-dataset.typ` — *Dataset and Inputs*
- `docs/typst/paper/sections/05-coordinate-conventions.typ` — *Coordinate Conventions and Geometry*
- `docs/typst/paper/sections/05-oracle-rri.typ` — *Oracle RRI Computation*
- `docs/typst/paper/sections/06-architecture.typ` — *Aria-VIN-NBV Architecture*
- `docs/typst/paper/sections/08a-frustum-pooling.typ` — *Semi-dense Frustum Pooling and View-Conditioned Tokens*
- `docs/typst/paper/sections/07-training-objective.typ` — *Training Objective*
- `docs/typst/paper/sections/07a-binning.typ` — *Stage-Aware Binning and Priors*
- `docs/typst/paper/sections/07b-training-config.typ` — *Current Training Configuration (VINv3 Baseline)*
- `docs/typst/paper/sections/09a-evaluation.typ` — *Evaluation Protocol*
- `docs/typst/paper/sections/09c-wandb.typ` — *WandB Run Analysis (Jan 3, 2026)* (`<sec:wandb-analysis>`)
- `docs/typst/paper/sections/09b-ablation.typ` — *Ablation Plan and Open Experiments*
- `docs/typst/paper/sections/10-discussion.typ` — *Discussion and Limitations*
- `docs/typst/paper/sections/10a-entity-aware.typ` — *Toward Entity-Aware NBV* (`<sec:entity-aware>`)
- `docs/typst/paper/sections/11-conclusion.typ` — *Conclusion*
- `docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ` — *Appendix: OracleRriLabeler Pipeline* (`<sec:appendix-oracle-rri-labeler>`)
- `docs/typst/paper/sections/12f-appendix-pose-frames.typ` — *Appendix: VIN Pose Frames and Consistency Checks* (`<sec:appendix-pose-frames>`)
- `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ` — *Appendix: VINv3 Streamlining Rationale*
- `docs/typst/paper/sections/12h-appendix-offline-cache.typ` — *Offline cache and batching*
- `docs/typst/paper/sections/12b-appendix-extra.typ` — *Appendix: Additional Diagnostics* (`<sec:appendix-extra>`)

## Slides (Typst)

**Entry point:** `docs/typst/slides/slides_4.typ`

**Note:** `slides_4.typ` uses `#import` (no `#include` blocks), so its “section graph” is import-based.

**Local imports in entry:**

- `docs/typst/slides/template.typ`
- `docs/typst/slides/notes.typ`
- `docs/typst/shared/macros.typ`

**Slide titles parsed (52)** (from `#slide(title: [...])` / `#section-slide(title: [...])`):

- ASE ATEK Dataset
- Oracle RRI Pipeline
- Candidate Generation
- Candidate Generation: Position sampling
- Candidate Generation: View directions
- Candidate Generation: Jitter + rules
- Candidate Depth Rendering
- Backprojection
- Oracle RRI: Accuracy + Completeness
- Oracle RRI: Relative improvement
- Offline cache: Motivation
- Offline cache: coverage + footprint
- Offline cache: what is stored?
- VinOracleBatch + VinSnippetView
- Data Flow: VinDataModule + VinSnippetCache
- Ordinal Binning
- Quantile binning (equal-mass ordinal classes)
- Ordinal labels + per-bin statistics (fit data)
- Bin calibration: midpoints, means, and variance
- CORAL Implementation Deltas
- VIN Pipeline
- Candidate Pose Encoding
- Scene branch: FieldBundle (EVL voxel field)
- Scene branch: voxel_valid_frac (coverage proxy)
- Scene branch: Global context + FiLM
- Semidense branch: semidense_proj (scalar stats)
- Semidense projection: transforms + bins
- Semidense projection maps
- Semidense branch: grid CNN
- Branch 8: trajectory context (traj_feat + traj_ctx)
- Branch 9: head input concat + VinPrediction outputs
- VIN-NBV (Frahm 2025) vs our VINv3
- Signal strength (sweep evidence)
- Oracle target + CORAL objective
- Logged diagnostics (selected)
- Metric definitions (logged keys)
- Loss + weighting equations (VIN Lightning)
- Metric equations (VIN Lightning)
- Training dynamics: CORAL loss (train)
- Validation: ordinal performance
- Auxiliary regression: loss + schedule
- Diagnostics gallery
- Common failure modes (and the checks we use)
- Failure case: cent_pr_nms artifact
- Optuna sweep: what helped vs not
- Mode collapse case study: vin-v3-01 vs T41
- Best W&B run (v03-best)
- Run dynamics: v03-best
- Training regime: what to try next
- Known limitations and open TODOs
- Key takeaways
- References

## Other files explicitly opened during init

- `docs/index.qmd`
- `docs/contents/todos.qmd`

## Typst compile reminders (repo convention)

- Paper: `cd docs && typst compile typst/paper/main.typ --root .`
- Slides: `cd docs && typst compile typst/slides/slides_4.typ --root .`

