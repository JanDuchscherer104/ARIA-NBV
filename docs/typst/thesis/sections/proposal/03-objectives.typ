#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Objectives and Hypotheses

The thesis objective is a reproducible target-aware #NBV stack whose selected
views improve target reconstruction quality under a fixed budget. The primary
hypothesis is policy-level: after oracle re-evaluation, $Q_(H,theta)$ should
achieve higher $J_(e,H)$ and $G_0^(H)$ than myopic learned or greedy one-step
selection on identical roots, candidates, masks, and budgets.

*O1: Trustworthy oracle labels.* M1 must validate offline-store indexing,
splits, frames, CW90 display handling, candidate-label order, rendered depth,
backprojection, invalidity reasons, and Rerun inspection before any target or
rollout scale-up.

*O2: Leakage-safe target utility.* V1 uses observed/predicted target descriptors
for selection and model input. GT crops and GT boxes are restricted to labels,
upper bounds, and evaluation. Scene #RRI, target #RRI, and acquisition cost are
reported separately.

*O3: Myopic control.* The target-conditioned one-step scorer predicts target
#RRI over the same candidate table as $Q_H$. It is evaluated by rank
correlation, top-$k$ oracle hit rate, calibration, target visibility, invalid
fraction, and failure groups.

*O4: Replayable multi-step data.* Rollouts include random-valid,
oracle-greedy/lookahead, and oracle-scored temperature-softmax traces with
seeds, policy ids, masks, reasons, target identity, and candidate provenance.
Gumbel-Top-k is later diversity evidence after deterministic lookahead is
trusted.

*O5: Mandatory finite-candidate $Q_H$.* The planner emits one masked value per
candidate:

$ bold(u)_(t,i) = "Tr"_theta(bold(S)_t, bold(T)_e, bold(H)_t, bold(C)_(t,i)),
  quad
  Q_(H,theta)(bold(s)_t^"cf0", bold(z)_e, bold(q)_(t,i)) =
  bold(w)_Q^top bold(u)_(t,i), $

$ a_t^theta =
  op("argmax", limits: #true)_(i in cal(A)_t)
  Q_(H,theta)(bold(s)_t^"cf0", bold(z)_e, bold(q)_(t,i)). $

Success is measured by oracle-rescored selected actions, not by predicted
values. If $Q_H$ does not beat one-step controls, the thesis reports the
failing gate rather than replacing the core claim with unvalidated continuous
RL.

#figure(
  table(
    columns: (0.86fr, 1.2fr, 1.65fr),
    table.header([*Claim*], [*Metric*], [*Evidence rule*]),
    [Target utility],
    [$J_(e,H)$, $G_t^(H)$, scene #RRI, cost],
    [Endpoint gain and cumulative return are separate; budgets use quality-cost curves.],
    [Target input safety],
    [selector rank, match score, support, leakage checks],
    [V1 actor input is observed/predicted; GT produces labels/evaluation only.],
    [Candidate space],
    [valid fraction, reasons, target visibility, provenance],
    [Hard masks and strategy provenance are present before scorer or $Q_H$ training.],
    [Planning gain],
    [oracle-evaluated target #RRI under equal budget],
    [$Q_H$ is compared to random-valid, one-step greedy, learned one-step scoring, and oracle lookahead.],
    [Scale],
    [scenes, snippets, targets, trajectories, seeds, gaps],
    [Full ASE GT-mesh coverage or an exact held-out subset report.],
  ),
  caption: [Objective-to-evidence matrix.],
) <tab:proposal-objective-evidence>
