#import "template/layout/proposal_template.typ": *
#import "metadata.typ": *
#import "../shared/macros.typ": *
#import "../shared/glossary.typ": *
#import "../shared/symbols.typ": symb
#import "../shared/equations.typ": eqs
#import "sections/proposal/_style.typ": *

#let proposalTitleEnglish = "ARIA-NBV: Advisor Research Contract"
#let proposalTitleGerman = "ARIA-NBV: Betreuer-Forschungsvertrag"

#let advisor-style(body) = {
  set text(size: 10.2pt)
  set par(leading: 0.92em, justify: true)
  set math.equation(numbering: "(1)")
  show heading.where(level: 1): set block(above: 1.25em, below: 0.62em)
  show heading.where(level: 1): set text(size: 14.2pt, weight: 700, fill: proposal-blue)
  show heading.where(level: 2): set block(above: 0.9em, below: 0.36em)
  show heading.where(level: 2): set text(size: 11.1pt, weight: 650, fill: proposal-red)
  show figure.caption: set text(size: 8.2pt, fill: proposal-muted)
  show math.equation: set text(size: 10pt, weight: 400)
  show table.cell: it => {
    set text(size: 8.15pt)
    set par(justify: false, leading: 0.72em)
    it
  }
  show table.cell.where(y: 0): it => {
    set text(size: 8.15pt, weight: 680, fill: proposal-blue)
    set par(justify: false)
    it
  }
  set table(inset: (x: 5pt, y: 4pt))
  body
}

#let claim-box(title, body) = block(above: 0.5em, below: 0.7em, breakable: true)[
  #rect(
    width: 100%,
    radius: 3pt,
    inset: (x: 7pt, y: 6pt),
    fill: proposal-blue.lighten(95%),
    stroke: 0.45pt + proposal-blue.lighten(55%),
  )[
    #text(size: 8.1pt, weight: 700, fill: proposal-red)[#smallcaps(title)]
    #v(3pt)
    #body
  ]
]

#let display-eq(body) = block(above: 0.48em, below: 0.58em, breakable: false)[#body]

#let gantt-bar(start, duration, critical: false) = {
  let total = 154
  let tail = total - start - duration
  grid(
    columns: (start * 1fr, duration * 1fr, tail * 1fr),
    column-gutter: 0pt,
    [],
    rect(
      height: 7pt,
      radius: 2pt,
      fill: if critical { proposal-red } else { proposal-blue },
    ),
    [],
  )
}

#let gantt-row(label, start, duration, critical: false) = grid(
  columns: (0.27fr, 0.73fr),
  column-gutter: 6pt,
  align: horizon,
  {
    set par(justify: false, leading: 0.78em)
    text(size: 8.1pt, weight: if critical { 700 } else { 500 }, fill: if critical { proposal-red } else { proposal-ink })[#label]
  },
  gantt-bar(start, duration, critical: critical),
)

#set document(title: proposalTitleEnglish, author: author)
#set text(font: "New Computer Modern")

#show: proposal.with(
  title: proposalTitleEnglish,
  titleGerman: proposalTitleGerman,
  thesisKindEnglish: thesisKindEnglish + " Advisor Handout",
  thesisKindGerman: "Betreuer-Handout zur " + thesisKindGerman,
  academicDegree: academicDegree,
  program: program,
  specialization: specialization,
  universityEnglish: universityEnglish,
  universityGerman: universityGerman,
  facultyEnglish: facultyEnglish,
  facultyGerman: facultyGerman,
  firstExaminer: firstExaminer,
  secondExaminer: "",
  supervisors: supervisors,
  author: author,
  email: email,
  matriculationNumber: matriculationNumber,
  startDate: startDate,
  submissionDate: datetime(day: 8, month: 5, year: 2026),
  submissionDateText: "8 May 2026",
  transparency_ai_tools: [
    AI-assisted tools were used to consolidate repository documentation, check proposal consistency, and draft this advisor handout. The author remains responsible for the final research scope, technical claims, citations, and submitted document.
  ],
)

#show: proposal-style
#show: advisor-style

= Current State and Thesis Claim

ARIA-NBV studies target-conditioned, quality-driven next-best-view selection for
egocentric indoor reconstruction. The implemented seminar substrate already
contains scene-level oracle #RRI labels, immutable VIN offline-store paths, a
VINv3-style one-step candidate scorer on frozen EVL evidence, candidate
generation, rollout scaffolding, and Rerun diagnostics. It is not yet an
end-to-end multi-step #NBV policy.

The thesis extension is the target-specific and non-myopic layer: observed
target selection, target-cropped oracle labels, replayable ASE counterfactual
rollouts, and a finite-horizon value model over valid candidate rows. The first
quantitative result stays inside the Project Aria / #ASE mesh-oracle loop
@projectaria-engel2023 @ProjectAria-ASE-2025 @EFM3D-straub2024. Habitat/Isaac,
continuous actor-critic control, SceneScript-style global memory, VLM planning,
and real-device guidance remain bridge designs unless the finite-candidate
result justifies escalation.

#claim-box([Thesis claim])[
  Given a logged Project Aria / #ASE snippet, an actor-visible target descriptor,
  a finite candidate table, and a fixed acquisition budget, bounded oracle
  lookahead first tests whether target-#RRI has non-myopic headroom. If headroom
  is positive, a masked candidate-query #symb.rl.qh model is evaluated by oracle
  re-scoring of its selected actions and by recovered headroom over the learned
  one-step target scorer.
]

= Formal Model and Research Questions

At decision step $t$, the actor-visible state is

#display-eq($
  #symb.rl.s_obs
  =
  (
    #symb.obs.points_semi_t,
    #symb.vin.field_evl_t,
    #symb.entity.target_hyp_pred_t,
    bold(h)_t,
    b_t
  ).
$)

The privileged oracle state augments this with GT geometry and all-candidate
oracle render/evaluation assets:

#display-eq($
  #symb.rl.s_oracle
  =
  (
    #symb.rl.s_obs,
    bold(M)_"GT",
    {bold(M)_e^"GT"}_(e in cal(E)),
    {#symb.obs.points_cand_ti}_(i=1)^(N_q)
  ).
$)

For the V1 main result, GT meshes, GT OBBs, GT crops, and all-candidate GT
renders are label and evaluation assets only. V0 may use GT OBB input only as a
sanity or upper-bound path, not as the main actor-visible result. The
counterfactual state keeps root EVL fixed:

#display-eq($
  #symb.rl.s_cf0
  =
  (
    #symb.vin.field_evl_0,
    #symb.obs.points_t,
    #symb.entity.target_desc,
    bold(h)_t,
    b_t,
    #symb.rl.candidate_table,
    #symb.rl.candidate_mask,
    #symb.rl.invalid_reasons
  ).
$)

The finite action set is the masked subset of candidate rows:

#display-eq($
  cal(A)_t
  =
  {i in {1, dots, N_q}: m_(t,i) = 1},
  quad
  bold(q)_t = bold(q)_(t,a_t).
$)

V1 uses OBS-SEL / PRED-Q / GT-EVAL. The descriptor #symb.entity.target_desc is
built from observed or predicted OBB geometry, class, confidence, projected
area, semidense support, EVL support, and relative pose. The GT crop is selected
only after matching:

#display-eq($
  e^star
  =
  op("argmax", limits: #true)_(e in cal(E))
  mu(hat(e), e).
$)

#display-eq($
  "accept"
  "iff"
  mu(hat(e), e^star) >= tau_mu
  "and"
  mu_1 - mu_2 >= tau_"gap".
$)

The target error is the implemented point-mesh accuracy plus mesh-to-point
completeness diagnostic, not point-cloud Chamfer distance:

#display-eq[#eqs.entity.target_error]

#display-eq[#eqs.entity.target_rri_reward]

#display-eq[#eqs.entity.finite_horizon_return]

#display-eq[#eqs.entity.endpoint_gain]

After selecting a valid candidate, the acquired geometry is added to the
current geometry:

#display-eq($
  #symb.obs.points_next
  =
  #symb.obs.points_t
  union
  #symb.obs.points_cand_ti.
$)

The next candidate table $bold(Q)_(t+1)$ is regenerated around the updated
geometry, history, and remaining budget with the same logged mixture families,
while #symb.vin.field_evl_0 remains fixed. Invalidity is always a hard mask with
a reason code, never a low-#RRI label.

The three thesis questions are tests:

#figure(
  table(
    columns: (0.28fr, 1.72fr),
    table.header([*RQ*], [*Advisor-facing test*]),
    [RQ1],
    [Can target-specific point-mesh #RRI be measured and matched without target leakage?],
    [RQ2],
    [Does an actor-visible target descriptor improve one-step target-#RRI ranking over scene-level scoring?],
    [RQ3],
    [Does bounded oracle lookahead expose positive headroom, and can #symb.rl.qh recover part of it from offline rollout support?],
  ),
  caption: [Research questions as falsifiable tests rather than deliverable slogans.],
) <tab:advisor-rqs>

The headroom and recovery quantities are

#display-eq[#eqs.entity.lookahead_headroom]

#display-eq[#eqs.entity.q_recovery]

= Architecture Contract

The architecture inspiration review argues against starting from a generic
"Transformer because Transformer" design. The first learned planner should
reuse VINv3's working geometry path and add only the missing planning biases:
target-local spatial reads, selected-view history, directional observation
memory, frustum-aware candidate features, permutation-equivariant candidate-set
reasoning, and mask-aware value heads.

The recommended first architecture is a residual finite-horizon model on top of
the target-conditioned one-step scorer:

#display-eq($
  #symb.rl.qh_theta (#symb.rl.s_cf0, #symb.entity.target_desc, bold(q)_(t,i))
  =
  hat(r)_psi^e (#symb.rl.s_cf0, #symb.entity.target_desc, bold(q)_(t,i))
  +
  A_theta^H (#symb.rl.s_cf0, #symb.entity.target_desc, bold(Q)_t, bold(h)_t, i).
$)

The residual advantage is initialized near zero, so the first #symb.rl.qh model
behaves like the myopic scorer until rollout evidence proves that planning adds
value. The scene memory uses the frozen EVL field, accumulated actor-visible
geometry, target support, and directional memory:

#display-eq($
  bold(F)_t^"scene"
  =
  op("Conv3D")_"small" (
    bold(F)_0^"EVL",
    bold(V)(bold(P)_t),
    bold(V)_"dir"(bold(P)_t),
    bold(V)_"target"(#symb.entity.target_desc)
  ).
$)

The target token should not be a descriptor vector alone; it should read a
target-local crop from the scene memory:

#display-eq($
  bold(T)_e
  =
  op("MLP")_phi (
    op("concat") (
      #symb.entity.target_desc,
      op("ROIAlign3D") (bold(F)_t^"scene", hat(bold(B))_e)
    )
  ).
$)

Directional observation memory is the proposed compact substitute for a large
raw view-frequency histogram over $S^2$:

#display-eq($
  bold(c)_(ell m)(bold(v))
  =
  sum_(o in cal(O) (bold(v)))
  w_o Y_(ell m)(bold(d)_o),
  quad
  ell <= L.
$)

#display-eq($
  n_"dir"(bold(v), bold(q)_(t,i))
  =
  1
  -
  sigma(
    sum_(ell,m)
    bold(c)_(ell m)(bold(v))
    Y_(ell m)(bold(d)_i(bold(v)))
  ).
$)

Each candidate row then receives pose, target-relative, frustum, belief-render,
directional novelty, mask/reason, and history features. The feature contract is
kept in small blocks so the rendered equations stay inspectable:

#display-eq($
  bold(p)_(t,i)
  =
  op("concat") (
    phi_"pose"(bold(q)_(t,i)),
    phi_"target-rel"(bold(q)_(t,i), #symb.entity.target_desc)
  ).
$)

#display-eq($
  bold(g)_(t,i)
  =
  op("concat") (
    phi_"frustum"(bold(F)_t^"scene", bold(q)_(t,i)),
    phi_"belief"(bold(P)_t, bold(F)_0^"EVL", bold(q)_(t,i)),
    phi_"dir"(bold(c), bold(q)_(t,i))
  ).
$)

#display-eq($
  bold(x)_(t,i)
  =
  op("concat") (
    bold(p)_(t,i),
    bold(g)_(t,i),
    phi_"valid"(m_(t,i), rho_(t,i)),
    bold(H)_t
  ).
$)

Candidate reasoning must be permutation-equivariant. The first implementation
can use Set Transformer attention over candidate tokens @SetTransformer-lee2019,
with an edge-conditioned spatial GNN as the first architectural ablation:

#display-eq($
  {bold(u)_(t,i)}_(i=1)^(N_q)
  =
  E_"set" (
    {
      op("concat") (bold(x)_(t,i), bold(T)_e, bold(H)_t)
    }_(i=1)^(N_q),
    bold(m)_t
  ).
$)

The head should be dueling and mask-normalized over valid actions:

#display-eq($
  #symb.rl.qh_theta (#symb.rl.s_cf0, #symb.entity.target_desc, bold(q)_(t,i))
  =
  V_theta (#symb.rl.s_cf0, #symb.entity.target_desc, bold(H)_t)
  +
  A_(theta,i)^H
  -
  (1) / (abs(cal(A)_t))
  sum_(j in cal(A)_t) A_(theta,j)^H.
$)

Dense GT candidate renders may be exploited only as privileged training signals,
not as V1 actor input:

#display-eq($
  cal(L)_"distill"
  =
  sum_(i in cal(A)_t)
  (
    Q_i^"student"
    -
    op("stopgrad") (Q_i^"priv")
  )^2.
$)

The thesis-safe order is therefore: target-conditioned one-step scorer,
residual #symb.rl.qh without a view-render branch, actor-visible belief-render
branch, $S^2$ directional memory, candidate-set interaction ablations,
distributional #symb.rl.qh head, and only then privileged-teacher distillation.
Point backbones, sparse convolutions, and continuous control are useful later
ablations, not the first success criterion.

= Learning Objective and Evaluation

The planned value model remains finite-candidate and masked. The candidate token,
candidate value, and selected action are

#display-eq[#eqs.rl.qh_candidate_token]

#display-eq[#eqs.rl.qh_candidate_value]

#display-eq[#eqs.rl.qh_masked_argmax]

The first backup is fitted masked Double-Q
#cite(label("DBLP:journals/corr/MnihKSGAWR13")) @DoubleDQN-vanHasselt2015.
Here $d_t=1$ at horizon termination, budget termination, or when no valid
successor action exists:

#display-eq[#eqs.rl.qh_doubleq_index]

#display-eq[#eqs.rl.qh_doubleq_target]

#display-eq[#eqs.rl.qh_loss]

All selected actions are re-evaluated by the oracle under identical roots,
candidate budgets, and acquisition budgets. The minimum final evidence report
must contain symbolic thresholds that are locked with the advisor before final
experiments:

#display-eq($
  (S_min, T_min, N_min)
  =
  (S_"scenes", T_"targets", N_"roots").
$)

Here the three symbols denote the minimum number of scenes, matched targets, and
rollout roots. Coverage is reported against the full scale bar of 100 GT-mesh
ASE scenes and 4,608 snippet windows, or against an explicit scene-level
held-out subset if scale is blocked.

#figure(
  table(
    columns: (0.72fr, 0.58fr, 0.58fr, 0.52fr, 1.3fr),
    table.header([*Policy*], [*Actor input*], [*GT decision*], [*H*], [*Role*]),
    [$pi_"rand"$], [yes], [no], [1], [lower reference over valid candidates],
    [$pi_"learned-1"$], [yes], [no], [1], [myopic learned target scorer],
    [$pi_"oracle-1"$], [no], [yes], [1], [one-step oracle upper bound],
    [$pi_"oracle-look"$], [no], [yes], [$H$], [non-myopic headroom estimate],
    [$pi_Q$], [yes], [no], [$H$], [learned recovery after oracle re-scoring],
  ),
  caption: [Leakage-aware policy comparison. Report #symb.entity.endpoint_gain, #symb.entity.return_h, scene #RRI, cost, invalidity, runtime, and coverage for each row.],
) <tab:advisor-policy-comparison>

= What We Adopt From Prior Work

#figure(
  table(
    columns: (0.86fr, 1.24fr, 1.48fr, 1.32fr),
    table.header([*Source family*], [*Role*], [*Adopt*], [*Do not adopt as core*]),
    [VIN-NBV @VIN-NBV-frahm2025],
    [Closest quality-driven finite-candidate precedent.],
    [Oracle point-mesh #RRI labels, ordinal one-step ranking, learned myopic control.],
    [Treating one-step greedy as enough for multi-step claims.],
    [Project Aria / #ASE / EFM3D @projectaria-engel2023 @ProjectAria-ASE-2025 @EFM3D-straub2024],
    [Logged egocentric sensing and mesh-supervised oracle substrate.],
    [Poses, calibration, semidense points, frozen EVL/EFM evidence, OBB predictions.],
    [GT geometry, GT OBBs, or dense labels as V1 actor input.],
    [Greedy sensing @KrauseSensorPlacement2008 @AdaptiveSubmodularity-golovin2011],
    [Reason greedy may be strong under diminishing returns.],
    [Measure oracle-lookahead headroom before claiming planning gain.],
    [Assume non-myopic learning is useful without a headroom test.],
    [CORAL and set models @CORAL-cao2019 @DeepSets-zaheer2017 @SetTransformer-lee2019],
    [Ranking and permutation-aware candidate scoring.],
    [Ordinal one-step target scorer; candidate-token #symb.rl.qh with masked set output.],
    [Unbounded architecture search before target/RL evidence is stable.],
    [Offline value learning #cite(label("DBLP:journals/corr/MnihKSGAWR13")) @DoubleDQN-vanHasselt2015 @IQL-kostrikov2021 @CQL-kumar2020 @BCQ-fujimoto2019],
    [Replay, overestimation control, and offline support constraints.],
    [Masked fitted Double-Q first; IQL/CQL/BCQ only as later ablations.],
    [Optimizing invalid or unsupported actions.],
    [GenNBV, Hestia, 3DGS, SceneScript @GenNBV-chen2024 @Hestia-lu2026 @FisherRF-jiang2024 @SceneScript-avetisyan2024],
    [Continuous, hierarchical, uncertainty, semantic, and global-memory contrast.],
    [Design pressure for post-#symb.rl.qh bridge work and discussion.],
    [Replacing target point-mesh #RRI with coverage, uncertainty, or semantic proxies as the main result.],
  ),
  caption: [Literature role ledger: each source family justifies a design choice or a deferred bridge; it does not broaden the thesis claim.],
) <tab:advisor-adoption-ledger>

= Roadmap, Risks, and Decisions

#figure(
  rect(width: 100%, inset: 7pt, radius: 3pt, stroke: 0.4pt + proposal-rule)[
    #grid(columns: (0.27fr, 0.73fr), column-gutter: 6pt, align: horizon,
      [],
      grid(columns: (12fr, 19fr, 31fr, 30fr, 31fr, 31fr), column-gutter: 0pt,
        text(size: 7.2pt, fill: proposal-muted)[May],
        text(size: 7.2pt, fill: proposal-muted)[Jun],
        text(size: 7.2pt, fill: proposal-muted)[Jul],
        text(size: 7.2pt, fill: proposal-muted)[Aug],
        text(size: 7.2pt, fill: proposal-muted)[Sep],
        text(size: 7.2pt, fill: proposal-muted)[freeze],
      ),
    )
    #v(3pt)
    #gantt-row([M0 proposal contract], 0, 12)
    #gantt-row([M1 oracle/geometry trust], 12, 21)
    #gantt-row([M2 one-step baseline/scale], 33, 21)
    #gantt-row([M3 V1 target oracle/selector], 54, 21)
    #gantt-row([M4 target-conditioned scorer], 75, 28)
    #gantt-row([M5 headroom and #symb.rl.qh], 103, 21, critical: true)
    #gantt-row([M6 bridge design], 124, 14)
    #gantt-row([M7 final experiments/writing], 138, 14, critical: true)
    #gantt-row([M8 release freeze], 152, 2, critical: true)
  ],
  caption: [Typst-native thesis Gantt, 2026-04-29 to 2026-09-30. Red bars are interpretation-critical: M5 headroom/#symb.rl.qh evidence and final freeze.],
) <fig:advisor-gantt>

Three risks decide interpretation. If M1 geometry or oracle labels fail, the
thesis becomes a contract and one-step-scoring study. If target matching is
sparse or ambiguous, target #RRI is reported only on validated subsets with
unmatched counts. If #symb.entity.lookahead_headroom is near zero, the thesis
reports an effectively myopic candidate/objective regime rather than promoting
continuous RL as a substitute.

Open advisor decisions are deliberately narrow: final scene-level split,
numeric minimum scale $(S_min,T_min,N_min)$, pass/fail threshold for
#symb.entity.q_recovery, exact target-match thresholds $(tau_mu,tau_"gap")$,
and the first actor-visible crop descriptor ablation. These choices change the
evidence bar, not the thesis spine.
