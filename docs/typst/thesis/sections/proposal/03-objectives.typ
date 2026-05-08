#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Objectives and Hypotheses

The thesis objective is a leakage-safe target-aware #NBV stack whose selected
views reduce target point-mesh error under a fixed acquisition budget. The
primary endpoint metric is $J_e^(H)$; the additive return $G_t^(H)$ is the
training target for value learning.

== Aim 1: Target Oracle and Leakage Boundary

Define target-specific oracle #RRI while keeping target selection and model
input actor-visible. V1 uses observed or predicted target descriptors
$bold(z)_e$; GT crops, boxes, meshes, and all-candidate renders are restricted
to labels, upper bounds, and evaluation. The required evidence is target
eligibility, match score, unmatched/ambiguous counts, endpoint $J_e^(H)$, and
separate scene #RRI and acquisition cost.

== Aim 2: Target-Conditioned One-Step Control

Train a VIN-style myopic scorer over the same candidate table that later feeds
$Q_H$. The scorer predicts target #RRI from actor-visible scene, target, and
candidate features. It is the required learned one-step control, evaluated by
rank correlation, top-$k$ oracle hit rate, calibration, selected-candidate
oracle #RRI, target visibility, invalid fraction, and grouped failures.

== Aim 3: Non-Myopic Finite-Candidate Planning

First estimate whether bounded oracle lookahead has headroom over one-step
oracle greedy:

$
  Delta_"look" =
  J_e^(H)(pi_"oracle-look") - J_e^(H)(pi_"oracle-1").
$

Only if this headroom is positive is $Q_H$ expected to recover part of it from
offline rollout traces. The candidate-query model emits one masked value per
candidate:

// TODO: subscript issue in "op("Transformer")_theta(bold(X)_t)_i," - (bold(X)_t)_i are part of the subscript, not argument to op Transformer!
$
  bold(u)_(t,i) =
  op("Transformer")_theta(bold(X)_t)_i,
$

$
  Q_(H,theta)(bold(s)_t^"cf0", bold(z)_e, bold(q)_(t,i)) =
  bold(w)_Q^top bold(u)_(t,i),
$

$
  a_t^theta =
  op("argmax", limits: #true)_(i in cal(A)_t)
  Q_(H,theta)(bold(s)_t^"cf0", bold(z)_e, bold(q)_(t,i)).
$

Success is measured by oracle-rescored selected actions, not predicted values.
If oracle lookahead itself has little headroom, the thesis reports that the
current objective and candidate distribution are effectively myopic. If
lookahead has headroom but $Q_H$ fails to recover it, the analysis reports
whether the limiting factor is target observability, candidate support, rollout
coverage, reward definition, or model capacity.

#figure(
  table(
    columns: (0.86fr, 1.32fr, 1.48fr),
    table.header([*Claim*], [*Primary evidence*], [*Decision rule*]),
    [Target utility],
    [$J_e^(H)$, $G_t^(H)$, scene #RRI, cost],
    [$J_e^(H)$ decides endpoint quality; $G_t^(H)$ trains and ranks rollouts.],

    [Input safety],
    [actor-visible $bold(z)_e$, match score, support, leakage checks],
    [GT is label/evaluation only for the V1 result.],

    [Myopic control],
    [target-rank metrics, selected-candidate oracle #RRI, calibration],
    [One-step target scoring is the comparator for $Q_H$, not the final policy claim.],

    [Planning headroom],
    [$Delta_"look"$ and recovered fraction $eta_Q$],
    [$Q_H$ is meaningful only relative to measured oracle-lookahead headroom.],
  ),
  caption: [Objective-to-evidence matrix.],
) <tab:proposal-objective-evidence>
