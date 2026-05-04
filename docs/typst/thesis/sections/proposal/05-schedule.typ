#import "../../../shared/macros.typ": *

= Schedule

The planned thesis window runs from 29 April 2026 to 30 September 2026. The schedule mirrors the current roadmap and keeps the highest-risk dependencies early: offline-store trust, geometry inspection, and oracle correctness precede target-conditioned modeling and rollout claims. Writing and figure preparation run throughout the project, but the final thesis narrative is frozen only after the August evidence gate.

#figure(
  table(
    columns: (1.15fr, 1.2fr, 1.8fr),
    inset: 5pt,
    table.header([*Dates*], [*Milestone*], [*Exit condition*]),
    [29 Apr--10 May],
    [Scope and diagnostics foundation],
    [The worktree is classified, the partial offline-store smoke path is understood, and the Rerun inspector workflow is documented.],
    [11 May--31 May],
    [Data, cache, and oracle correctness],
    [Offline-store contracts, split handling, frame conventions, candidate validity, and oracle throughput are stable enough for model claims.],
    [1 Jun--21 Jun],
    [VIN baseline and calibration],
    [A reproducible one-step VIN baseline is trained and evaluated with rank, calibration, ordinal-bin, and top-k diagnostics.],
    [22 Jun--12 Jul],
    [Target-specific #RRI],
    [Ground-truth OBB-cropped target #RRI is implemented, inspected, and compared with scene-level #RRI on a trusted subset.],
  ),
  caption: [Planned thesis schedule through the target-specific oracle milestone.],
) <tab:proposal-schedule-a>

#figure(
  table(
    columns: (1.15fr, 1.2fr, 1.8fr),
    inset: 5pt,
    table.header([*Dates*], [*Milestone*], [*Exit condition*]),
    [13 Jul--9 Aug],
    [Target-conditioned VIN],
    [The model accepts a target encoding and is evaluated against scene-level, target-level, oracle, and geometric baselines.],
    [10 Aug--30 Aug],
    [Multi-step rollouts],
    [Greedy, random, bounded oracle, stochastic or beam, and model-scored rollouts are compared by cumulative target #RRI and acquisition cost.],
    [31 Aug--13 Sep],
    [Value/RL gate],
    [A masked discrete Q or close-greedy RL baseline is attempted only if rollout evidence and scorer calibration justify it.],
    [14 Sep--30 Sep],
    [Thesis freeze and release],
    [Final figures, ablations, failure cases, reproducible configs, and demonstration smokes are complete.],
  ),
  caption: [Planned thesis schedule from target-conditioned modeling to release freeze.],
) <tab:proposal-schedule-b>

The main risk is that oracle or offline-store trust takes longer than expected. In that case, the thesis remains valid as a stronger geometry-first study of scene-level #RRI, target-specific #RRI, and one-step candidate scoring, while learned multi-step control is reduced to oracle rollout analysis. A second risk is that target-specific labels are too sparse or unstable for reliable model training. The fallback is to report target #RRI as an oracle evaluation and diagnostic tool while keeping the learned model scene-level. A third risk is that bounded lookahead does not improve over one-step greedy under the available candidate budget. That outcome is still scientifically useful if it is paired with failure analysis, because it identifies whether candidate generation, target observability, or the reward definition is the limiting factor.

The stretch path is value learning or policy learning. It will be pursued only after held-out VIN ranking, oracle-evaluated VIN-selected rollouts, calibration and stage-shift analysis, and Rerun failure visualization are available. If those gates are not met by the end of August, the thesis will prioritize robust target-aware ablations and a defensible future-work discussion over a weak reinforcement-learning claim.
