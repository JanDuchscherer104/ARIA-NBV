#import "../../../shared/macros.typ": *

= Method and Timeline

The method proceeds through gates rather than a single end-to-end rewrite. First, ARIA-NBV validates the immutable offline-store and oracle contracts. Second, it implements target-specific #RRI by matching actor-visible observed/predicted targets to GT target crops for labels and evaluation. Third, it trains a target-conditioned one-step scorer as the comparator for planning. Fourth, it generates replayable rollout traces from random-valid, oracle-greedy/lookahead, and oracle-scored temperature-softmax policies. Temperature-softmax is a data-diversity policy over scored candidates, not the final objective.

The fifth gate trains $Q_H$ as a candidate-query Transformer. Scene, target, history, and candidate fields are encoded as tokens; hard masks suppress invalid candidates; the output is one finite-horizon Q value per candidate. The target is bounded cumulative target #RRI from ASE oracle rollouts. Double-Q-style selection/evaluation separation is the first target-construction strategy because max-based finite-candidate learning is vulnerable to overestimation @DoubleDQN-vanHasselt2015. CQL and BCQ remain useful offline-RL references for distribution-shift risk, while Decision Transformer is a useful sequence-modeling reference if later trajectory decoding becomes justified @CQL-kumar2020 @BCQ-fujimoto2019 @DecisionTransformer-chen2021.

#figure(
  table(
    columns: (0.92fr, 1.18fr, 1.9fr),
    inset: 4.5pt,
    table.header([*Dates*], [*Milestone*], [*Decision gate*]),
    [29 Apr--10 May],
    [M0 scope/proposal],
    [Compact proposal, bibliography/source policy, and mandatory $Q_H$ scope frozen.],
    [11 May--31 May],
    [M1 data/oracle],
    [Offline-store, geometry, candidate-label, Rerun, and throughput evidence pass or block scale-up.],
    [1 Jun--21 Jun],
    [M2 scorer baseline],
    [Scene-level VIN-style scorer is reproducible and calibration/ranking evidence is available.],
    [22 Jun--12 Jul],
    [M3 target #RRI],
    [V0 sanity and V1 actor-visible target protocol are trusted on a small subset.],
    [13 Jul--9 Aug],
    [M4 target scorer],
    [Observed/predicted target encoding improves target ranking or exposes a documented limitation.],
    [10 Aug--30 Aug],
    [M5 rollouts/$Q_H$],
    [$Q_H$ is trained and oracle-evaluated against one-step baselines under equal budget.],
    [31 Aug--30 Sep],
    [M6--M8 evidence/freeze],
    [Optional bridge work only after $Q_H$ evidence; final coverage, figures, configs, and thesis text freeze.],
  ),
  caption: [Compact thesis timeline.],
)

The main risk is substrate fragility: frame mistakes, invalid candidates, or sparse target labels could make learned planning claims meaningless. The fallback is not to switch to continuous actor-critic control, but to report the exact failing gate and preserve a defensible target-aware oracle/scorer/rollout study. Continuous control, online Gymnasium/SB3, Habitat/Isaac, SceneScript-style semantic memory, and real-device guidance remain post-$Q_H$ bridge designs unless the M5 evidence and time budget make them credible.
