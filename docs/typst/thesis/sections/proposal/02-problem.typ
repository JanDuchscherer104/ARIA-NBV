#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Problem and Research Contract

ARIA-NBV is a target-conditioned, finite-candidate #NBV problem in the
Project Aria / ASE / EFM observation regime. The thesis is deliberately not a
first-order continuous-control claim: it tests whether bounded planning over a
finite valid candidate table improves target reconstruction quality beyond
myopic view selection.

The actor-visible state at decision step $t$ is

$ bold(s)_t^"obs" =
  (bold(P)_t^"semi", bold(F)_t^"EVL", bold(O)_t^"pred",
   bold(h)_t, b_t), $

where $bold(P)_t^"semi"$ is the accumulated semi-dense or fused point evidence,
$bold(F)_t^"EVL"$ is the frozen local EFM/EVL evidence field, $bold(O)_t^"pred"$ are
observed or predicted target hypotheses, $bold(h)_t$ is selected-view history, and
$b_t$ is remaining budget. The oracle state augments this with ASE GT assets:

$ bold(s)_t^"oracle" =
  (bold(s)_t^"obs", bold(M)_"GT", {bold(M)_e^"GT"}_(e in cal(E)),
   {bold(P)_(t,i)^"cand"}_(i=1)^(N_q)). $

GT meshes, GT OBBs, GT crops, GT semantic labels, and all-candidate GT renders
are label/evaluation assets only. They are not V1 actor inputs.

The planner state is

$ bold(s)_t^"cf0" =
  (bold(F)_0^"EVL", bold(P)_t, bold(z)_e, bold(h)_t, b_t,
   bold(Q)_t, bold(m)_t, bold(rho)_t), $

where $bold(Q)_t={bold(q)_(t,i)}_(i=1)^(N_q)$ is the finite candidate table,
$m_(t,i) in {0,1}$ is a hard validity mask, and $rho_(t,i)$ is an invalid
reason. The admissible actions are candidate indices:

$ cal(A)_t = {i in {1,dots,N_q}: m_(t,i)=1}. $

Invalidity is a constraint, not low utility. Masks apply before argmax,
softmax, loss targets, and bootstrap maximization.

The V1 target protocol is OBS-SEL / PRED-Q / GT-EVAL. Target selection and
model input use an actor-visible descriptor

$ bold(z)_e =
  phi(hat(bold(B))_e, hat(bold(y))_e, hat(p)_e, A_e^"proj",
      n_e^"semi", n_e^"EVL", bold(T)_e^"rel"), $

covering predicted/observed OBB geometry, class, confidence, projected area,
semi-dense support, EVL support, and relative pose. GT crops $bold(M)_e$ are used only
after matching $bold(z)_e$ to GT by class compatibility, OBB overlap, and support.
Unmatched or ambiguous targets are reported as target-invalid cases.

For target $e$, the implemented oracle error is the point-mesh accuracy plus
mesh-to-point completeness diagnostic:

$ Delta_t^e = cal(A)_t^e + cal(C)_t^e. $

This is the `pm_acc_*` / `pm_comp_*` distance used by `aria_nbv`, not generic
point-cloud Chamfer distance. If valid action $a_t=i$ selects $bold(q)_(t,i)$, then

$ bold(P)_(t+1) = bold(P)_t union bold(P)_(t,i)^"cand",
  quad
  r_t^e = (Delta_t^e - Delta_(t+1)^e) / (Delta_t^e + epsilon). $

The value-learning return and endpoint metric stay separate:

$ G_t^(H) = sum_(k=0)^(H-1) gamma^k r_(t+k)^e,
  quad
  J_(e,H) = (Delta_0^e - Delta_H^e) / (Delta_0^e + epsilon). $

The log-gain companion
$L_(e,H)=log(Delta_0^e+epsilon)-log(Delta_H^e+epsilon)$ is only an ablation.

#thesis-box([Main question])[
  Can a target-conditioned candidate-query Transformer
  $Q_(H,theta)(bold(s)_t^"cf0", bold(z)_e, bold(q)_(t,i))$ choose valid candidate views whose
  oracle-evaluated cumulative target #RRI beats one-step greedy/model scoring
  under equal acquisition and candidate budgets?
]

The hard quantitative core is observed target selection, target #RRI labels, a
target-conditioned one-step scorer, replayable oracle rollouts, and mandatory
$Q_H$. Online discrete $Q_H$, IQL, actor-critic control, external simulators,
3DGS control, SceneScript, VLM planning, and real-device guidance are bridge or
future-work surfaces unless M5 evidence justifies escalation.
