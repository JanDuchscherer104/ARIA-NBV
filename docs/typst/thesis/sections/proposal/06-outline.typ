#import "../../../shared/macros.typ": *

= Preliminary Thesis Outline

The thesis will be written as an empirical methods thesis. It starts from the quality-driven #NBV problem, introduces the Aria data and geometry contracts, then develops the oracle, model, and rollout components in the same order in which they are validated. The preliminary bibliography is rendered at the end of this proposal from the shared repository bibliography.

#figure(
  table(
    columns: (0.8fr, 1.55fr, 1.45fr),
    inset: 5pt,
    table.header([*Chapter*], [*Purpose*], [*Expected evidence*]),
    [Introduction],
    [Motivate target-conditioned, quality-driven #NBV for egocentric indoor reconstruction and state the thesis questions.],
    [Problem definition, scope boundary, and contribution summary.],
    [Background and related work],
    [Review active perception, classical view planning, #NBV, #RRI, egocentric foundation models, target-aware reconstruction, bounded rollout, and offline value learning.],
    [Literature synthesis centered on VIN-NBV, EFM3D, GenNBV, Hestia, radiance-field #NBV, Trajectory Transformer, Gumbel-Top-k, Double Q-learning, and IQL.],
    [Data and geometry contracts],
    [Describe #ASE snippets, camera and pose frames, immutable offline stores, candidate generation, and Rerun inspection.],
    [Offline-store inventory, frame-safety checks, and visual diagnostics.],
    [Oracle #RRI and target-specific #RRI],
    [Define scene-level and target-level reconstruction-quality labels, invalid cases, and accuracy/completeness components.],
    [Trusted oracle examples, label distributions, target crops, and throughput measurements.],
    [Target-conditioned candidate scoring],
    [Present the VIN-style scorer, target encoding, ordinal objective, and calibration analysis.],
    [Held-out ranking, top-k oracle hit rate, calibration, ablations, and failure cases.],
    [Bounded rollout and $Q_H$ evaluation],
    [Compare one-step greedy, random, oracle rollout, temperature-softmax rollout, model-scored rollout, and candidate-query Transformer $Q_H$ under equal budget.],
    [Cumulative target #RRI, scene #RRI, $Q_H$ success bar, acquisition cost, invalid-action rate, runtime, and trajectory visualizations.],
    [Discussion and conclusion],
    [Interpret limits, failure modes, simulator and deployment gaps, and future continuous-control or entity-aware extensions.],
    [Evidence-gated conclusion and reproducibility notes.],
  ),
  caption: [Preliminary thesis chapter outline.],
) <tab:proposal-outline>

The expected final contribution is a reproducible target-aware finite-candidate view-selection study, not a broad claim that continuous reinforcement learning has been solved for egocentric reconstruction. The thesis will therefore separate implemented results from extensions. The required result includes candidate-query Transformer $Q_H$ over finite candidate sets if the prerequisite gates pass; SceneScript-style semantic memory, 3D Gaussian Splatting simulators, open-vocabulary targets, actor-critic control, and real-device guidance are future directions unless they become necessary to explain the core target-aware #RRI and $Q_H$ results.
