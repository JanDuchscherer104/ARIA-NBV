#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Schedule and Risk Control

The planned thesis window runs from 29 April 2026 to 30 September 2026. The
schedule is organized around evidence gates rather than feature optimism:
offline-store trust and oracle correctness precede target-conditioned modeling;
target #RRI and observed target selection precede $Q_H$ data generation; bridge
work begins only after the finite-candidate planning evidence is interpretable.

#figure(
  image("../../figures/proposal_gantt.png", width: 100%),
  caption: [Mermaid Gantt chart for the proposal timeline and evidence gates.],
) <fig:proposal-gantt>

#figure(
  table(
    columns: (0.9fr, 1.18fr, 1.78fr),
    table.header([*Dates*], [*Milestone*], [*Exit condition*]),
    [2026-04-29 to 2026-05-10],
    [M0 scope/proposal],
    [Proposal, roadmap, questions, source policy, and hard $Q_H$ boundary are aligned and renderable.],
    [2026-05-11 to 2026-05-31],
    [M1 data/oracle],
    [Offline-store, split, frame/CW90, candidate-label, depth/backprojection, Rerun, and throughput evidence pass or block scale-up.],
    [2026-06-01 to 2026-06-21],
    [M2 one-step baseline],
    [Scene-level VIN baseline, calibration/ranking plots, LRZ sharding plan, and Zarr rollout/Q schema are ready.],
    [2026-06-22 to 2026-07-12],
    [M3 target #RRI],
    [V0 GT-OBB target labels and V1 OBS-SEL / PRED-Q / GT-EVAL contract are trusted on a small subset.],
    [2026-07-13 to 2026-08-09],
    [M4 target scorer],
    [Target-conditioned one-step scorer is evaluated against scene-level scoring and oracle target #RRI.],
    [2026-08-10 to 2026-08-30],
    [M5 rollouts/$Q_H$],
    [Random-valid, oracle-greedy/lookahead, and temperature-softmax traces are stable; candidate-query $Q_H$ is trained and oracle-evaluated.],
    [2026-08-31 to 2026-09-13],
    [M6 bridge design],
    [Online discrete $Q_H$, IQL, actor-critic, hierarchy, and simulator bridge are written as designs or gated ablations.],
    [2026-09-14 to 2026-09-27],
    [M7 experiments/writing],
    [Final tables, figures, failure cases, coverage report, and thesis narrative are frozen.],
    [2026-09-28 to 2026-09-30],
    [M8 release],
    [Configs, smoke checks, demo path, and final PDF artifacts are reproducible.],
  ),
  caption: [Milestone exit criteria.],
) <tab:proposal-schedule>

The most likely failure mode is not that continuous RL is needed too early; it
is that target labels, validity masks, or scale generation are not trusted
enough for learned planning. The fallback policy is therefore conservative. If
M1 fails, the thesis remains a geometry/oracle contract and one-step scorer
study. If M3 target matching is sparse, target #RRI remains an oracle diagnostic
while scene-level scoring is reported honestly. If $Q_H$ fails to beat myopic
controls, the thesis reports whether the limiting factor is candidate support,
target observability, reward definition, or model capacity. None of these
fallbacks reclassifies the mandatory $Q_H$ attempt as optional.
