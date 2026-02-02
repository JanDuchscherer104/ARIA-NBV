# Peer Review: Aria-VIN-NBV Typst paper (2026-01-20)

Scope: review of `docs/typst/paper/main.typ` + included sections as of 2026-01-20.

## Summary (as a reviewer)

The paper presents *Aria-VIN-NBV*, a quality-driven next-best-view (NBV) planner for egocentric indoor scenes, built on (i) an **oracle** Relative Reconstruction Improvement (RRI) computation pipeline in Aria Synthetic Environments (ASE) using GT meshes + semidense SLAM points, and (ii) a **learned** View Introspection Network (VIN v2) that predicts ordinal RRI scores from a frozen EVL backbone plus candidate-conditioned semidense projection cues. The paper is strongest as a *system + pipeline report* that emphasizes correctness, diagnostics, and reproducibility.

## Strengths

- Clear motivation for optimizing *reconstruction quality* (RRI) vs. coverage proxies; places work well relative to VIN-NBV and GenNBV.
- Oracle pipeline is conceptually clean and well modularized (candidate gen → render → backproject/fuse → point↔mesh eval → RRI labels).
- Architecture section explains the actual implementation (EVL field channels + pose encoding + global pooling + semidense view conditioning + CORAL head) and separates core vs ablations.
- Diagnostics emphasis is appropriate for a complex geometry/renderer/system; appendices provide code-faithful details.
- Metrics definitions (RRI, Chamfer-style error split, Spearman/top-k bin accuracy, CORAL monotonicity) are stated explicitly, which improves scientific readability.

## Major concerns (blocking for a research-paper submission)

1) Lack of quantitative evaluation / baselines
- The current text reads closer to a “snapshot report” than a paper with validated claims.
- There is no systematic evaluation table on a clearly defined split (scenes held out).
- Ablations are presented as a plan, not results. A paper needs at least a minimal ablation matrix with measured deltas.

2) Dataset split + leakage control is underspecified
- The paper says “100 scenes with GT meshes” and a local snapshot of 4,608 snippets, but does not define:
  - train/val/test splits (by scene, not snippet),
  - how snippets are sampled per scene,
  - whether multiple snippets from the same scene appear across train/val/test.
- Without per-scene holdout, generalization claims (even implicit) are hard to support.

3) “Top-k” metric is bin-level, not NBV selection
- The logged top-3 metric is **top-3 bin accuracy** (label ∈ top-3 predicted ordinal bins), which measures ordinal classification, not “did we select a good view”.
- For NBV, the load-bearing metric should include at least one *selection-aware* score:
  - Recall@k of oracle-best candidate (or top-m oracle candidates) vs. predicted ranking,
  - NDCG / Kendall / per-snippet Spearman (macro-averaged).

4) Aggregation choices for Spearman can bias the result
- Current definition is “micro-averaged over all valid candidates in an epoch” (flattened over snippets and candidates).
- This may overweight scenes/snippets with more valid candidates or larger candidate sets.
- Recommend additionally reporting **macro-averaged Spearman per snippet** (mean over snippets, optionally with CI).

5) Oracle cost / throughput and failure rates are not quantified
- Since oracle RRI generation is expensive and candidate pruning can produce “0 candidates”, the paper should report:
  - average candidates per snippet (before/after pruning),
  - fraction of snippets dropped,
  - oracle throughput (snippets/s) on a specified GPU/CPU,
  - typical memory footprint (mesh + batch sizes).

## Minor concerns / polish

- Terminology: ensure consistent spelling (“semi-dense” vs “semidense”; “Weights & Biases” vs “WandB”).
- Training config table likely needs: batch size, number of epochs, effective dataset size, hardware (GPU model), and rationale (why OneCycle values).
- Architecture section contains “coverage ratio and empty fraction” but without a formula; consider adding a 1–2 line definition to avoid ambiguity.
- Entity-aware objective is promising but currently too abstract; clarify how `RRI_e(q)` is computed (OBB proxy? mesh subset?) or frame it as future work more explicitly.
- Consider adding one short “failure case” paragraph (e.g., out-of-bounds candidates, missing walls, or NaN projection features) with a diagnostic you implemented to catch it.

## Concrete, minimal improvements to make this paper submission-ready

1) Add a compact results table (even preliminary)
- On a scene-level split of the 100 GT-mesh scenes (e.g. 80/10/10), report:
  - Spearman (macro per snippet + micro),
  - Recall@1/3/5 for oracle-best candidate,
  - top-3 bin accuracy (keep, but do not treat as selection metric),
  - monotonicity violation rate,
  - collapse rate (e.g., std of predicted scores across candidates).

2) Add 2–3 baselines
- Random ranking.
- A simple heuristic baseline using semidense projection stats (e.g., empty fraction / candidate visibility).
- EVL-only baseline without semidense conditioning.

3) Convert ablation plan into measured ablations
- At minimum: `+ semidense proj stats`, `+ frustum MHCA`, `+ trajectory ctx`, `+ voxel gating`.

4) Add a reproducibility paragraph
- Exact config path/commit, EVL checkpoint id, ATEK preprocessing version, and command lines for oracle generation + training.

## Questions to clarify (as reviewer)

- Which EVL checkpoint/config is used and what inputs are provided (RGB only vs multi-stream; semidense/free-space)?
- How is candidate pose distribution parameterized (N, radius/elevation ranges, roll sampling)?
- How often do you get zero candidates after pruning and what do you do with those snippets during training?
- Do you evaluate on scenes not seen during training (scene-level split)?

