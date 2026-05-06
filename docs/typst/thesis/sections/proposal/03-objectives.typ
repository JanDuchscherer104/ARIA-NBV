#import "../../../shared/macros.typ": *

= Objectives

The thesis objective is to produce a reproducible target-aware #NBV stack for egocentric indoor reconstruction. The expected outcome is not a claim of finished continuous-control reinforcement learning. It is a controlled study of whether target-specific #RRI supervision, learned candidate scoring, bounded multi-step rollout, and finite-candidate fitted Double-Q / $Q_H$ selection can improve reconstruction quality under the same view budget as one-step greedy selection.

The primary research question is whether a target-conditioned #NBV model can produce acquisition-cost-efficient trajectories that maximize target-specific #RRI for a selected target of interest. This question decomposes into five measurable objectives. First, the data and oracle contracts must be trustworthy enough that candidate geometry and #RRI labels can be inspected sample by sample. Second, target-specific #RRI must be defined using ground-truth oriented bounding box crops before predicted targets or open-vocabulary targets are introduced. Third, a VIN-style scorer must rank candidates by scene-level and target-level #RRI on held-out snippets. Fourth, bounded multi-step rollout must be compared against one-step greedy selection under equal candidate and acquisition budgets. Fifth, a target-conditioned fitted $Q_H$ model must learn bounded cumulative target #RRI from the trusted rollout traces and beat one-step greedy or model scoring under the same acquisition budget, or produce a documented blocker.

#figure(
  table(
    columns: (1.1fr, 1.45fr, 1.35fr),
    inset: 5pt,
    table.header([*Objective*], [*Evidence*], [*Exit condition*]),
    [Offline trust contract],
    [Immutable VIN offline samples expose required pose, camera, point, candidate, and #RRI fields.],
    [One train and one validation sample load without online fallback and can be inspected geometrically.],
    [Target-specific #RRI],
    [Ground-truth OBB crops define target mesh and target point subsets for oracle evaluation.],
    [Target #RRI and scene #RRI are reported side by side on a trusted subset.],
    [Target-conditioned scorer],
    [A VIN-style model receives a target encoding and ranks candidates by target-specific #RRI.],
    [Held-out rank correlation, top-k oracle hit rate, and calibration are compared against a scene-level scorer.],
    [Bounded rollout],
    [Greedy, random, bounded oracle rollout, stochastic or beam rollout, model-scored rollout, and fitted $Q_H$ are evaluated with the same budget.],
    [$Q_H$ improves cumulative target #RRI over one-step greedy/model scoring or the limitation is documented with failure cases.],
  ),
  caption: [Planned thesis objectives and evidence gates.],
) <tab:proposal-objectives>

The required contributions are the oracle and data contract, the target-specific #RRI formulation, the target-conditioned candidate scorer, bounded rollout evaluation, and masked fitted Double-Q / $Q_H$ over finite candidate sets. Optional extensions include richer entity tokens, open-vocabulary target selection, IQL-style offline value learning after $Q_H$ is stable, and continuous target-then-pose control. These extensions become thesis-core only if the earlier evidence gates show that the finite-candidate target-aware setting is reliable and time remains.

The thesis will evaluate success using reconstruction-quality and planning metrics rather than only training loss. The core metrics are target #RRI, scene #RRI, cumulative target #RRI, acquisition cost, invalid-action rate, runtime, rank correlation, top-k oracle hit rate, ordinal calibration, and representative failure visualizations. This metric set keeps model quality, planning quality, and data validity visible as separate axes.
