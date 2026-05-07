#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Problem and Research Contract

ARIA-NBV is formulated as a finite-candidate, target-conditioned #NBV problem.
At rollout step $t$, the actor-visible counterfactual state is

$ s_t^"cf0"
  = (
      F_v^"root",
      P_t,
      Q_t,
      m_t,
      rho_t,
      z_e,
      b_t
    ) $

where $F_v^"root"$ is the frozen EVL/EFM voxel-field context at the snippet
root, $P_t$ is accumulated semi-dense or rendered/fused geometry, $Q_t =
{q_(t,i)}_(i=1)^(N_q)$ is the finite candidate table, $m_(t,i) in {0,1}$ is the
validity mask, $rho_(t,i)$ is the invalidity reason, $z_e$ is the actor-visible
descriptor of the selected target $e$, and $b_t$ is the remaining acquisition
budget. The oracle-only state augments this with the #ASE GT mesh $M$ and target
mesh crop $M_e$; these are label/evaluation assets and are not actor inputs in
the main V1 protocol.

Target-conditioned evaluation uses OBS-SEL / PRED-Q / GT-EVAL. Target selection
uses only observed or predicted evidence. Scoring and $Q_H$ receive the
predicted/observed target descriptor $z_e$. Labels and final evaluation are
computed by matching that actor-visible target to a GT target crop $M_e$.
V0 may use GT boxes as a sanity or upper-bound path, but V1 is the main thesis
claim.

For a target crop operator $C_e(.)$, define the target distance

$ D_e(P) = "CD"(C_e(P), M_e), $

with the same accuracy/completeness surface-distance family used by the shared
#RRI equations. If action $a_t=i$ selects valid candidate $q_(t,i)$, the
counterfactual geometry transition is

$ P_(t+1) = P_t union P_(q_(t,i)). $

The immediate target reward is state-relative target #RRI:

$ r_t^e
  = (D_e(P_t) - D_e(P_(t+1))) / (D_e(P_t) + epsilon). $

The endpoint quality metric for a rollout sequence
$tau=(a_0,dots,a_(H-1))$ is instead root-relative:

$ J_e^(H)(tau)
  = (D_e(P_0) - D_e(P_H)) / (D_e(P_0) + epsilon). $

The learning return is additive:

$ G_t^(H)
  = sum_(k=0)^(H-1) gamma^k r_(t+k)^e. $

Thus $G_t^(H)$ trains the finite-horizon value function, while $J_e^(H)$ reports
the fraction of initial target error removed at the fixed budget. Negative
target #RRI remains a valid signal when a view worsens target distance; invalid
actions are constraints and are hard-masked before argmax, softmax, loss, and
bootstrap operations.

The central research question is:

#thesis-box([Main question])[
  Can ARIA-NBV train a target-conditioned candidate-query Transformer
  $Q_(H,theta)(s_t^"cf0", z_e, q_(t,i))$ that predicts one masked bounded-horizon
  value per finite candidate and whose selected actions, when re-evaluated by
  the #ASE oracle, beat one-step greedy/model scoring on cumulative target #RRI
  under equal acquisition and candidate budgets?
]

The contract is deliberately conservative. The implemented substrate covers
scene-level oracle #RRI, immutable VIN-style offline stores, candidate
generation, one-step scoring, early rollout scaffolding, and Rerun inspection.
The prerequisite evidence protocol covers M1 data/oracle correctness, V0/V1
target contracts, invalidity masks/reasons, Zarr-first rollout/Q storage,
LRZ-scale generation gates, scene-level splits, and exact coverage reporting.
The hard quantitative core is observed target selection, mixed candidate sets,
target #RRI labels, a target-conditioned one-step scorer, trusted oracle
rollouts, and mandatory $Q_H$. Online discrete $Q_H$, IQL, actor-critic,
continuous control, SceneScript, 3DGS simulators, and real-device guidance are
bridge or future-work surfaces unless the M5 evidence justifies escalation.
