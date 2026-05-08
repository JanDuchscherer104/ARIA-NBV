#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Method

== Geometry, Targets, and One-Step Control

The geometry stage establishes the candidate poses, rig poses, rendered depths,
backprojected point clouds, semi-dense support, EVL fields, and Rerun
diagnostics that must agree in frame and indexing. Oracle labels use the
implemented point-mesh terms

$ Delta_t = cal(A)_t + cal(C)_t,
  quad
  Delta_t^e = cal(A)_t^e + cal(C)_t^e,
  quad
  r_t^e = (Delta_t^e - Delta_(t+1)^e) / (Delta_t^e + epsilon). $

The target stage adds the V1 contract. The selector returns a small top-$K$ set
of observed or predicted targets and refuses GT OBB input for the main result.
Target labels are attached only after matching to GT crops. The one-step
target-conditioned scorer keeps the current VIN-style ordinal setup, including
CORAL variants @CORAL-cao2019, and becomes the myopic learned control.

== Candidate and Replay Contract

Each decision state carries a finite candidate table $bold(Q)_t$, hard mask
$bold(m)_t$, invalid-reason vector $bold(rho)_t$, target descriptor
$bold(z)_e$, selected-view history, and remaining budget. Candidate generation
is logged as a mixture over target-point, radial-shell, and forward-rig
families. Each row stores provenance, pose, target/scene support, path
increment, validity, reason code, and oracle labels. Candidate order has no
semantics, so shuffled-candidate evaluation is required.

The minimum replay row is

$ ("scene", "snippet", "target", t, bold(s)_t^"cf0", bold(z)_e,
   bold(Q)_t, bold(m)_t, bold(rho)_t, a_t, r_t^e,
   bold(s)_(t+1)^"cf0", bold(Q)_(t+1), bold(m)_(t+1), "policy", "seed"). $

This row reproduces the mask, selected transition, value target, and oracle
re-evaluation.

== Rollout Policies and $Q_H$

The planning evaluation first compares deterministic one-step oracle greedy and
bounded oracle lookahead:

$ pi_"oracle-1"(bold(s)_t) =
  op("argmax", limits: #true)_(i in cal(A)_t) r_t^e(i), $

$ pi_"oracle-look"(bold(s)_t) =
  pi_0(
    op("argmax", limits: #true)_(a_(t:t+H-1))
    sum_(k=0)^(H-1) gamma^k r_(t+k)^e
  ). $

Temperature-softmax traces are rollout-data diversity, not the final policy:

$ P(a_t=i | bold(s)_t)
  =
  exp(beta ell_(t,i)) / (sum_(j in cal(A)_t) exp(beta ell_(t,j))). $

The learned planner is a candidate-query Transformer over scene, target,
history, and candidate tokens @Transformer-vaswani2017 @DeepSets-zaheer2017
@SetTransformer-lee2019:

$ bold(u)_(t,i) = op("Transformer")_theta(bold(X)_t)_i, $

$ Q_(H,theta)(bold(s)_t^"cf0", bold(z)_e, bold(q)_(t,i)) =
  bold(w)_Q^top bold(u)_(t,i). $

Invalid candidates are filled with $-infinity$ before argmax, softmax, loss
targets, and bootstrap maximization. With online parameters $theta$ and target
parameters $theta^-$, the first backup is masked Double-Q
#cite(label("DBLP:journals/corr/MnihKSGAWR13")) @DoubleDQN-vanHasselt2015:

$ i^star =
  op("argmax", limits: #true)_(i in cal(A)_(t+1))
  Q_(H,theta)(bold(s)_(t+1)^"cf0", bold(z)_e, bold(q)_(t+1,i)), $

$ y_t =
  r_t^e
  + gamma (1 - d_t)
    Q_(H,theta^-)(bold(s)_(t+1)^"cf0", bold(z)_e, bold(q)_(t+1,i^star)), $

$ cal(L)_Q(theta) =
  (1)/(|cal(D)|)
  sum_((s,a,r,s') in cal(D))
  m_(t,a)
  (Q_(H,theta)(bold(s)_t^"cf0", bold(z)_e, bold(q)_(t,a)) - y_t)^2. $

IQL, CQL, BCQ, sequence decoding, soft/energy policies, PPO, and SAC remain
later comparison references unless the finite-candidate rollout store is stable
@IQL-kostrikov2021 @CQL-kumar2020 @BCQ-fujimoto2019
@DecisionTransformer-chen2021 @DeepEnergyPolicies-haarnoja2017
@PPO-schulman2017 @SAC-haarnoja2018.

== Evaluation

All selected actions are oracle-evaluated under the same acquisition and
candidate budgets. The proposal uses one canonical comparison table:

#figure(
  table(
    columns: (0.92fr, 0.7fr, 0.7fr, 0.64fr, 1.22fr),
    table.header([*Policy*], [*Actor input?*], [*GT decision?*], [*Horizon*], [*Primary role*]),
    [$pi_"rand"$],
    [yes],
    [no],
    [1],
    [lower reference over valid candidates],
    [$pi_"learned-1"$],
    [yes],
    [no],
    [1],
    [myopic learned target scorer],
    [$pi_"oracle-1"$],
    [no],
    [yes],
    [1],
    [one-step oracle upper bound],
    [$pi_"oracle-look"$],
    [no],
    [yes],
    [$H$],
    [non-myopic headroom estimate],
    [$pi_Q$],
    [yes],
    [no],
    [$H$],
    [learned recovery of lookahead headroom],
  ),
  caption: [Policy comparison and leakage boundary. Report $J_e^(H)$, $G_0^(H)$, scene #RRI, cost, invalidity, runtime, and coverage for each row.],
) <tab:proposal-policy-eval>
