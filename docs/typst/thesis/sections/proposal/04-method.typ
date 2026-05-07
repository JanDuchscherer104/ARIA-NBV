#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Method

The method is a sequence of empirical gates. Each gate either produces a
trusted artifact for the next gate or records a blocker.

== M1/M2: Geometry and One-Step Baseline

First, candidate poses, rig poses, rendered depths, backprojected point clouds,
semidense support, EVL fields, and Rerun diagnostics must agree in frame and
indexing. The oracle labels use the implemented point-mesh diagnostics

$ Delta_t = cal(A)_t + cal(C)_t,
  quad
  Delta_t^e = cal(A)_t^e + cal(C)_t^e,
  quad
  r_t^e = (Delta_t^e - Delta_(t+1)^e) / (Delta_t^e + epsilon). $

The one-step VIN-style scorer is then trained as the myopic comparator. Its
purpose is to test whether actor-visible state and candidate features predict
oracle #RRI well enough to justify planning.

== M3/M4: Target Contract and Target Scoring

The selector returns a small top-$K$ set of observed/predicted targets. V1
refuses GT OBB input; the labeler matches selected targets to GT crops by class
compatibility, OBB overlap, visibility/support, projected area, and
semidense/EVL support. Ambiguous matches are excluded and counted.

The target-conditioned scorer receives the same finite candidate set as $Q_H$.
Its ablations are bounded to target descriptor choice, surface input, CORAL
variant, auxiliary regression, candidate-relative pose encoding, and
calibration/stage features.

== Candidate and Replay Contract

Candidate generation is logged as a finite mixture over target-point, radial
shell, and forward-rig continuity families. Optional projection/frontier
shortlists are diagnostics until calibrated. Every candidate row stores
provenance, pose, target/scene support, path increment, validity mask, invalid
reason, and oracle labels. Candidate order is not semantic; shuffled-candidate
evaluation is required for $Q_H$.

The minimum replay row is

$ ("scene", "snippet", "target", t, bold(s)_t^"cf0", bold(z)_e,
   bold(Q)_t, bold(m)_t, bold(rho)_t, a_t, r_t^e,
   bold(s)_(t+1)^"cf0", bold(Q)_(t+1), bold(m)_(t+1), "policy", "seed"). $

This is enough to reproduce the mask, value target, and oracle re-evaluation.

== M5: Rollouts and $Q_H$

Generate deterministic oracle greedy and bounded oracle lookahead first:

$ a_t^"greedy" =
  op("argmax", limits: #true)_(i in cal(A)_t) r_t^e(i), $

$ a_t^"look" =
  pi_0(
    op("argmax", limits: #true)_(a_(t:t+H-1))
    sum_(k=0)^(H-1) gamma^k r_(t+k)^e
  ). $

Temperature-softmax traces then widen support:

$ P(a_t=i | bold(s)_t)
  =
  exp(beta ell_(t,i)) / (sum_(j in cal(A)_t) exp(beta ell_(t,j))). $

The candidate-query Transformer consumes scene, target, history, and candidate
tokens $bold(X)_t=[bold(S)_t,bold(T)_e,bold(H)_t,{bold(C)_(t,i)}]$ and decodes one value per candidate:

$ bold(u)_(t,i) = "Tr"_theta(bold(X)_t)_i,
  quad
  Q_(H,theta)(bold(s)_t^"cf0", bold(z)_e, bold(q)_(t,i)) =
  bold(w)_Q^top bold(u)_(t,i). $

Invalid candidates are filled with $-infinity$ before argmax, softmax, and
bootstrap maximization.

With online parameters $theta$ and target parameters $theta^-$, the first
target is masked Double-Q:

$ i^star =
  op("argmax", limits: #true)_(i in cal(A)_(t+1))
  Q_(H,theta)(bold(s)_(t+1)^"cf0", bold(z)_e, bold(q)_(t+1,i)), $

$ y_t =
  r_t^e + gamma Q_(H,theta^-)(bold(s)_(t+1)^"cf0", bold(z)_e, bold(q)_(t+1,i^star)), $

$ cal(L)_Q(theta) =
  (1)/(|cal(D)|)
  sum_((s,a,r,s') in cal(D))
  m_(t,a) (Q_(H,theta)(bold(s)_t^"cf0", bold(z)_e, bold(q)_(t,a)) - y_t)^2. $

CQL, BCQ, IQL, Decision Transformer, soft/energy policies, PPO, and SAC remain
comparison references or later ablations @CQL-kumar2020 @BCQ-fujimoto2019
@IQL-kostrikov2021 @DecisionTransformer-chen2021
@DeepEnergyPolicies-haarnoja2017 @PPO-schulman2017 @SAC-haarnoja2018.

== Evaluation

All selected actions are oracle-evaluated under the same budget. The main table
compares random-valid, one-step oracle greedy, learned one-step target scorer,
bounded oracle lookahead, temperature-softmax rollouts, and learned $Q_H$ on
scene-level splits. Required columns are $J_(e,H)$, $G_0^(H)$, scene #RRI, view
count, path length, invalid-action rate, runtime, scenes, snippets, targets,
trajectories, rollout seeds, transitions, and coverage gaps.

#figure(
  table(
    columns: (0.7fr, 1.25fr, 1.55fr),
    table.header([*Gate*], [*Artifact*], [*Fail-safe*]),
    [M1],
    [geometry/oracle contract report],
    [block target/RL scale-up if frames or labels are not trustworthy],
    [M3],
    [V0/V1 target #RRI and observed-only selector],
    [report target oracle only on validated subsets if matching is sparse],
    [M4],
    [target-conditioned scorer],
    [freeze as myopic baseline with calibration and failure groups],
    [M5],
    [rollout store, oracle lookahead, $Q_H$],
    [report the exact blocker if $Q_H$ does not beat myopic controls],
    [M6+],
    [online/IQL/actor-critic/simulator bridge],
    [do not promote bridge work without M5 evidence],
  ),
  caption: [Method gates and fail-safe interpretation.],
) <tab:proposal-method-gates>
