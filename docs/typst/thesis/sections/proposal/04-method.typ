#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Method

The method is a sequence of gates. Each gate either produces a trusted artifact
for the next gate or records an explicit blocker.

== M1/M2: Data, Geometry, and One-Step Baseline

The first phase freezes the geometry contracts. Candidate poses, rig poses,
rendered depths, backprojected point clouds, semidense support, EVL fields, and
Rerun diagnostics must agree in frame and indexing. The oracle computes
scene-level and target-level labels from

#eqs.rri.cd

and

#eqs.rri.target_rri

without aggressive mesh or point-cloud downsampling in fine-detail runs. The
one-step VIN-style scorer then serves as the learned myopic baseline, not as
the final policy. Its main purpose is to test whether actor-visible state and
candidate features predict oracle #RRI well enough to justify planning.

== M3/M4: Target Contract and Target-Conditioned Scoring

Target selection produces a small top-$K$ set of actor-visible observed/predicted
targets per snippet. For V1, the selector scores predicted OBB confidence,
projected visibility, semidense/EVL support, support deficit, distance, and
class eligibility; it refuses GT OBB input. The labeler then matches selected
targets to GT crops using class compatibility, OBB overlap, visibility/support,
projected area, and semidense/EVL support. Ambiguous matches and the exact
near-solved-target threshold remain advisor-facing details.

The target-conditioned one-step scorer receives the same finite candidate set
as $Q_H$ and predicts target #RRI. Its ablations are bounded: target descriptor
choice, surface reconstruction input, CORAL variant, auxiliary regression,
candidate-relative pose encoding, and calibration/stage features. This keeps
model work from becoming unbounded architecture search before the planning
question is answered.

== M5: Bounded Rollout and $Q_H$ Training

For each root state, deterministic oracle one-step greedy and bounded oracle
lookahead are generated first. Let $B$ be branch factor, $H$ horizon, and
$cal(A)(s_t)={i:m_(t,i)=1}$. Greedy selection is

$ a_t^"greedy"
  = op("argmax", limits: #true)_(i in cal(A)(s_t)) r_t^e(i). $

Bounded lookahead selects the first action of the best rollout:

$ a_t^"look"
  = pi_0(
      op("argmax", limits: #true)_(a_(t:t+H-1))
      sum_(k=0)^(H-1) gamma^k r_(t+k)^e
    ), $

where $pi_0(.)$ extracts the first action. Temperature-softmax traces then add
support:

$ P(a_t=i | s_t)
  = exp(beta u_(t,i)) / sum_(j: m_(t,j)=1) exp(beta u_(t,j)), $

where $u_(t,i)$ may be oracle immediate #RRI, oracle lookahead value, or a
trusted scorer. The rollout store preserves logits, probabilities, entropy,
temperature, selected action, RNG seed, target identity, candidate provenance,
and invalid reasons.

$Q_H$ is trained as masked fitted Double-Q over finite candidates. With online
parameters $theta$ and target parameters $bar(theta)$,

$ i^star
  = op("argmax", limits: #true)_(i: m_(t+1,i)=1)
    Q_(H,theta)(s_(t+1)^"cf0", z_e, q_(t+1,i)), $

$ y_t
  = r_t^e
    + gamma Q_(H,bar(theta))(s_(t+1)^"cf0", z_e, q_(t+1,i^star)), $

$ cal(L)_Q(theta)
  = (1)/(|cal(D)|)
    sum_((s,a,r,s') in cal(D))
    m_(t,a)
    (Q_(H,theta)(s_t^"cf0", z_e, q_(t,a)) - y_t)^2. $

This target family uses the Double-DQN selector/evaluator separation to reduce
max-operator overestimation @DoubleDQN-vanHasselt2015. CQL, BCQ, IQL, Decision
Transformer, soft/energy policies, PPO, and SAC are comparison references or
later ablations because they require stronger support assumptions, sequence
decoding, or an online simulator @CQL-kumar2020 @BCQ-fujimoto2019
@IQL-kostrikov2021 @DecisionTransformer-chen2021
@DeepEnergyPolicies-haarnoja2017 @PPO-schulman2017 @SAC-haarnoja2018.

== Evaluation

All selected actions are oracle-evaluated under the same budget. The main table
compares random-valid, one-step oracle greedy, learned one-step target scorer,
bounded oracle lookahead, temperature-softmax rollouts, and learned $Q_H$ on
scene-level splits. Required columns are endpoint target gain $J_e^(H)$,
cumulative target return $G_0^(H)$, scene #RRI, view count, path length,
invalid-action rate, runtime, scenes, snippets, targets, trajectories, rollout
seeds, transitions, and coverage gaps. Quality-cost curves are used when
budgets differ.

#figure(
  table(
    columns: (0.72fr, 1.28fr, 1.55fr),
    table.header([*Gate*], [*Artifact*], [*Fail-safe*]),
    [M1],
    [contract report, Rerun examples, throughput and frame checks],
    [block target/RL scale-up if geometry or labels are not trustworthy],
    [M3],
    [V0/V1 target #RRI and observed-only target selector],
    [fall back to target oracle diagnostics if actor-visible matching is sparse],
    [M4],
    [target-conditioned one-step scorer],
    [use as myopic baseline even if accuracy is modest, provided calibration is reported],
    [M5],
    [rollout store, oracle lookahead, and candidate-query $Q_H$],
    [report the exact blocker if $Q_H$ does not beat myopic controls],
    [M6+],
    [online/IQL/actor-critic/simulator bridge design],
    [do not promote bridge work to thesis core without M5 evidence],
  ),
  caption: [Method gates and fail-safe interpretation.],
) <tab:proposal-method-gates>
