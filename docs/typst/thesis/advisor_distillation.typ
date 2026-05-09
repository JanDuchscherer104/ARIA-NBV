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

#let rq-node(title, body, critical: false) = rect(
  width: 100%,
  radius: 3pt,
  inset: (x: 7pt, y: 5pt),
  fill: if critical { proposal-red.lighten(93%) } else { proposal-blue.lighten(96%) },
  stroke: 0.45pt + if critical { proposal-red.lighten(45%) } else { proposal-blue.lighten(58%) },
)[
  #text(size: 8pt, weight: 700, fill: if critical { proposal-red } else { proposal-blue })[#title]
  #h(4pt)
  #text(size: 8pt, fill: proposal-ink)[#body]
]

#let rq-down() = align(center)[
  #text(size: 9.5pt, fill: proposal-muted)[#sym.arrow.b]
]

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
    text(
      size: 8.1pt,
      weight: if critical { 700 } else { 500 },
      fill: if critical { proposal-red } else { proposal-ink },
    )[#label]
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

Next-best-view selection is often optimized through coverage, uncertainty, or information-gain proxies. These proxies help exploration, but they do not directly test whether a selected view improves the geometry of a specific target. This gap matters for egocentric indoor reconstruction: broad scene coverage can still leave a target poorly reconstructed because support is sparse, occluded, or observed from unfavorable directions.

ARIA-NBV studies a finite-candidate, target-conditioned version of this problem on Project Aria / #ASE data @projectaria-engel2023 @ProjectAria-ASE-2025 with frozen EFM3D/EVL evidence @EFM3D-straub2024. The prior seminar implementation provides scene-level oracle #RRI labels and a one-step scoring substrate. The thesis extends this substrate to leakage-safe target labels, fixed-budget oracle lookahead, actor-visible finite-horizon value learning, and a scaling analysis that preserves mesh/oracle target-#RRI supervision. Online discrete interaction and continuous target-then-pose policies are lower-priority escalation RQs after the finite-candidate evidence, not substitutes for it.

#claim-box([Thesis claim])[
  Target-specific reconstruction-quality improvement can serve as a finite-horizon #NBV objective for egocentric indoor reconstruction. The thesis tests this by training #symb.rl.qh to predict bounded cumulative target-#RRI for a target of interest and by evaluating selected trajectories with endpoint target-quality gain. Oracle lookahead first measures whether non-myopic headroom exists; only then is actor-visible #symb.rl.qh recovery interpreted under the same candidate and acquisition budgets.
]

= Formal Model and Research Questions

Let #symb.obs.points_t denote the accumulated actor-visible geometry proxy at decision step $t$, initialized from the logged semi-dense reconstruction and updated during counterfactual rollouts. Let #symb.vin.field_evl_0 be the frozen root egocentric evidence field, and let #symb.entity.target_desc be the actor-visible descriptor of the selected target. The actor-visible state is

$
  #symb.rl.s_obs
  =
  (
    #symb.obs.points_t,
    #symb.vin.field_evl_0,
    #symb.entity.target_desc,
    bold(h)_t,
    b_t
  ).
$

The privileged oracle state augments this with GT geometry and all-candidate oracle render/evaluation assets:

$
  #symb.rl.s_oracle
  =
  (
    #symb.rl.s_obs,
    #symb.ase.mesh,
    {#symb.ase.mesh_target}_(e in cal(E)),
    {#symb.obs.points_cand_ti}_(i=1)^(N_q)
  ).
$

For the V1 main result, GT meshes, GT OBBs, GT crops, and all-candidate GT renders are label and evaluation assets only. V0 may use GT OBB input only as a sanity or upper-bound path, not as the main actor-visible result. The counterfactual state keeps root EVL fixed:

$
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
$

The finite action set is the masked subset of candidate rows:

$
  cal(U)_t
  =
  {i in {1, dots, N_q}: m_(t,i) = 1},
  quad
  q_t = q_(t,a_t).
$

Invalidity is a constraint, not weak supervision: masks apply before argmax, temperature softmax, loss targets, and bootstrap maximization. True infeasibility or absent evaluation samples are hard-invalid; low immediate target support is a diagnostic unless no meaningful oracle/evaluation sample exists. V1 uses OBS-SEL, PRED-Q, and GT-EVAL. The actor-visible descriptor is

$
  #eqs.entity.target_descriptor
$

It bundles observed or predicted OBB geometry, class, confidence, projected area, semidense support, EVL support, and relative pose. A compact actor-visible crop descriptor is the first target-input ablation after this OBB-level path, not a GT crop. GT crops are selected only after protocol-level matching:

$
  #eqs.entity.target_match_score
$

$
  #eqs.entity.target_match_selection
$

$
  #eqs.entity.target_match_acceptance
$

Targets that fail acceptance, or whose top-1/top-2 scores are ambiguous, are counted as target-invalid protocol cases rather than low target-#RRI examples.

Let $C_e (#symb.obs.points_t)$ denote the oracle-only crop of accumulated points to the matched target region. The target error is the implemented point-mesh accuracy plus mesh-to-point completeness diagnostic on this target crop, not point-cloud Chamfer distance:

$
  #eqs.entity.target_error
$

$
  #eqs.entity.target_rri_reward
$

$
  #eqs.entity.finite_horizon_return
$

$
  #eqs.entity.endpoint_gain
$

$
  #eqs.entity.log_gain
$

#symb.entity.endpoint_gain is the primary endpoint metric, #symb.entity.return_h is the rollout training return, and #symb.entity.log_gain is an ablation for scale sensitivity only. Since each immediate #RRI term is normalized by the current target error, the additive return is not algebraically identical to endpoint gain; the former supports value fitting and the latter supports fixed-budget interpretation.

After selecting a valid candidate, the acquired geometry is added to the current geometry:

$
  #symb.obs.points_next
  =
  #symb.obs.points_t
  union
  #symb.obs.points_cand_ti.
$

The next candidate table $cal(Q)_(t+1)$ is regenerated from the updated geometry, selected-view history, and remaining budget with the same logged mixture families, while #symb.vin.field_evl_0 remains fixed.

The six research questions form a dependency chain: the objective defines what counts as improvement, leakage-safe target matching defines which target is evaluated, candidate and rollout-support design defines the finite action evidence, and #symb.rl.qh is interpretable only after oracle headroom and scale are measured.

#figure(
  block(width: 100%)[
    #rq-node([RQ1 Objective and metrics], [learn target-conditioned finite-candidate multi-step #NBV and separate #symb.entity.endpoint_gain for endpoint evaluation, #symb.entity.return_h for value learning, and #symb.entity.log_gain as ablation])
    #rq-down()
    #rq-node([RQ2 Target and matching], [OBB plus support is the V1 baseline; actor-visible crop descriptors are the first ablation; #symb.entity.target_desc is matched to GT only for labels and evaluation])
    #rq-down()
    #rq-node([RQ3 Candidate and rollout support], [mixed target-centric plus exploration candidates, branch/beam knobs, and stochastic support traces define the finite action rows available to learning])
    #rq-down()
    #rq-node([RQ4 Headroom and #symb.rl.qh], [oracle lookahead optimizes cumulative target-#RRI, and actor-visible #symb.rl.qh must beat validated myopic scoring when headroom is positive], critical: true)
    #rq-down()
    #rq-node([RQ5 Scaling], [scale ASE finite-candidate evidence first, then external mesh/oracle-compatible substrates and online discrete #symb.rl.qh if the supervision contract remains comparable])
    #rq-down()
    #rq-node([RQ6 Online and continuous escalation], [continuous target-then-pose actor-critic is time-permitting after online training evidence; imitation variants are deferred beyond planned #RRI + #symb.rl.qh])
  ],
  caption: [Causal RQ dependency chain. Later questions should not be used to rescue failed earlier contracts.],
) <fig:advisor-rq-dag>

Oracle lookahead selects by cumulative target-#RRI and endpoint gain evaluates the resulting trajectory. Report #symb.entity.q_recovery only when its denominator is positive and above the advisor-set minimum effect threshold. The headroom and recovery quantities are

$
  #eqs.entity.lookahead_headroom
$

$
  #eqs.entity.q_recovery
$

= Planned Value-Model Design

This section is subordinate to the oracle and headroom tests. The research object is the finite-candidate value model #symb.rl.qh, with the candidate-query Transformer as the first implementation. The learned model must map each valid candidate row to a finite-horizon value using only actor-visible scene, target, history, and candidate features. It must be permutation-equivariant over candidate rows, apply the hard mask before action selection and training targets, and be evaluated by oracle re-scoring rather than by its own predicted values.

The recommended first architecture is a residual finite-horizon model on top of the target-conditioned one-step scorer:

$
  #symb.rl.qh_theta (#symb.rl.s_cf0, #symb.entity.target_desc, q_(t,i))
  =
  hat(r)_psi^e (#symb.rl.s_cf0, #symb.entity.target_desc, q_(t,i))
  +
  A_theta^H (#symb.rl.s_cf0, #symb.entity.target_desc, #symb.rl.candidate_table, bold(h)_t, i).
$

The residual advantage is initialized near zero, so the first #symb.rl.qh model behaves like the myopic scorer until rollout evidence proves that planning adds value. The scene memory uses the frozen EVL field, accumulated actor-visible geometry, target support, and directional memory:

$
  bold(F)_t^"scene"
  =
  op("Conv3D")(
    bold(F)_0^"EVL",
    bold(V) (#symb.obs.points_t),
    bold(V)_"dir" (#symb.obs.points_t),
    bold(V)_"target" (#symb.entity.target_desc)
  ).
$

The target token should not be a descriptor vector alone; it should read a target-local crop from the scene memory:

$
  bold(T)_e
  =
  op("MLP")_phi (
    op("concat") (
      #symb.entity.target_desc,
      op("ROIAlign3D") (bold(F)_t^"scene", hat(bold(B))_e)
    )
  ).
$

Candidate orientation and accumulated visibility are separate signals. The pose branch uses the continuous R6D rotation representation @zhou2019continuity:

$
  #symb.vin.candidate_pose_feat (q_(t,i))
  =
  op("concat") (
    bold(t)_(t,i),
    bold(R)_(t,i)^"6D",
    bold(t)_(t,i) - bold(t)_e,
    alpha_(t,i)^e,
    l_(t,i),
    c_(t,i)^"strategy"
  ).
$

Accumulated visibility is a separate actor-visible directional memory on $bb(S)^2$. The first implementation should use a moment memory:

$
  #eqs.features.direction_unit
$

$
  #eqs.features.direction_memory_moment
$

$
  #eqs.features.direction_novelty
$

Low-order spherical harmonics are a richer directional-memory ablation, not a precondition for the first value-model test. Each candidate row then receives pose, target-relative, frustum, belief-render, directional novelty, mask/reason, and history features:

$
  bold(p)_(t,i)
  =
  op("concat") (
    #symb.vin.candidate_pose_feat (q_(t,i)),
    phi_"target-rel" (q_(t,i), #symb.entity.target_desc)
  ).
$

$
  bold(g)_(t,i)
  =
  op("concat") (
    phi_"frustum" (bold(F)_t^"scene", q_(t,i)),
    phi_"belief" (#symb.obs.points_t, #symb.vin.field_evl_0, q_(t,i)),
    phi_"dir" (#symb.vin.dir_moment, q_(t,i))
  ).
$

$
  bold(x)_(t,i)
  =
  op("concat") (
    bold(p)_(t,i),
    bold(g)_(t,i),
    phi_"valid" (m_(t,i), rho_(t,i)),
    bold(H)_t
  ).
$

Candidate reasoning must be permutation-equivariant; a Set Transformer candidate encoder is the first planned implementation @SetTransformer-lee2019:

$
  {bold(u)_(t,i)}_(i=1)^(N_q)
  =
  E_"set" (
    {
      op("concat") (bold(x)_(t,i), bold(T)_e, bold(H)_t)
    }_(i=1)^(N_q),
    bold(m)_t
  ).
$

The head should be dueling and mask-normalized over valid actions:

$
  #symb.rl.qh_theta (#symb.rl.s_cf0, #symb.entity.target_desc, q_(t,i))
  =
  V_theta (#symb.rl.s_cf0, #symb.entity.target_desc, bold(H)_t)
  +
  A_(theta,i)^H
  -
  (1) / (abs(cal(U)_t))
  sum_(j in cal(U)_t) A_(theta,j)^H.
$

The thesis-safe order is therefore: target-conditioned one-step scorer, residual #symb.rl.qh without a view-render branch, actor-visible belief-render branch, R6D pose features, $bb(S)^2$ moment directional memory, low-order spherical harmonic memory, candidate-set interaction ablations, edge-conditioned spatial GNNs, distributional #symb.rl.qh heads, and privileged-teacher distillation. Dense GT candidate renders may be used only as privileged training signals in later ablations, never as V1 actor input. Point backbones, sparse convolutions, online discrete interaction, and continuous control are lower-priority scaling or escalation steps, not the first success criterion.

= Learning Objective and Evaluation

The planned value model remains finite-candidate and masked. The candidate token, candidate value, and selected action are

$
  #eqs.rl.qh_candidate_token
$

$
  #eqs.rl.qh_candidate_value
$

$
  #eqs.rl.qh_masked_argmax
$

The first backup is fitted masked Double-Q #cite(label("DBLP:journals/corr/MnihKSGAWR13")) @DoubleDQN-vanHasselt2015. Here $d_t=1$ at horizon termination, budget termination, or when no valid successor action exists:

$
  #eqs.rl.qh_doubleq_index
$

$
  #eqs.rl.qh_doubleq_target
$

$
  #eqs.rl.qh_loss
$

Before #symb.rl.qh is interpreted as planning, the learned one-step target scorer must pass the myopic-control evidence gate: held-out ranking, oracle-evaluated model-selected rollouts, calibration and stage-shift diagnostics, and Rerun examples of representative successes and failures.

All selected actions are re-evaluated by the oracle under identical roots, candidate budgets, and acquisition budgets. Equal budget means equal selected-view horizon $H$, candidate count $N_q$, candidate-generation distribution, and validity constraints; path length, runtime, and oracle evaluation count are reported separately. The minimum final evidence report must contain symbolic thresholds that are locked with the advisor before final experiments:

$
  (S_min, T_min, N_min)
  =
  (S_"scenes", T_"targets", N_"roots").
$

Here the three symbols denote the minimum number of scenes, matched targets, and rollout roots. Coverage is reported against the full scale bar of 100 GT-mesh ASE scenes and 4,608 snippet windows, or against an explicit scene-level held-out subset if scale is blocked. Final splits are scene-level; sample-level splitting across snippets from the same scene is not valid for final claims.

#figure(
  table(
    columns: (0.62fr, 1.78fr),
    table.header([*Surface*], [*Required evidence*]),
    [One-step scorer],
    [held-out rank correlation, top-$k$ oracle hit, calibration, stage-shift diagnostics, selected-candidate oracle #RRI, target visibility, Rerun success/failure examples],

    [Candidate and replay],
    [strategy provenance, path increments, scene/target support fields, validity masks, reason codes, with/without-mask rank metrics, policy metadata, seed metadata, shuffled-candidate evaluation],

    [Scale and storage],
    [scene-level splits, no silent coverage changes, scale axes reported separately, Zarr asset references, LRZ/Slurm/DSS/resume/storage gates],
  ),
  caption: [Compact evidence contract: quality is reported both as ranking performance and as replay/candidate integrity.],
) <tab:advisor-evidence-contract>

#figure(
  table(
    columns: (0.72fr, 0.58fr, 0.58fr, 0.52fr, 1.3fr),
    table.header([*Policy*], [*Actor input*], [*GT decision*], [*H*], [*Role*]),
    [$pi_"rand"$], [yes], [no], [1], [lower reference over valid candidates],
    [$pi_"learned-1"$], [yes], [no], [1], [myopic learned target scorer],
    [$pi_"oracle-1"$], [no], [yes], [1], [one-step oracle upper bound],
    [$pi_"oracle-look"$], [no], [yes], [$H$], [cumulative-#RRI headroom estimate],
    [$pi_Q$], [yes], [no], [$H$], [learned recovery; must beat myopic scoring when headroom is positive],
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

    [CORAL and set models @CORAL-cao2019 @SetTransformer-lee2019],
    [Ranking and permutation-aware candidate scoring.],
    [Ordinal one-step target scorer; finite-candidate #symb.rl.qh, first as masked candidate-query Transformer.],
    [Unbounded architecture search before target/RL evidence is stable.],

    [Offline value learning #cite(label("DBLP:journals/corr/MnihKSGAWR13")) @DoubleDQN-vanHasselt2015 @IQL-kostrikov2021 @CQL-kumar2020 @BCQ-fujimoto2019],
    [Replay, overestimation control, and offline support constraints.],
    [Masked fitted Double-Q first; IQL/CQL/BCQ only as later ablations.],
    [Optimizing invalid or unsupported actions.],

    [GenNBV, Hestia, 3DGS, SceneScript @GenNBV-chen2024 @Hestia-lu2026 @FisherRF-jiang2024 @SceneScript-avetisyan2024],
    [Continuous, hierarchical, uncertainty, semantic, and global-memory contrast.],
    [Design pressure for lower-priority scaling and online/continuous escalation.],
    [Replacing target point-mesh #RRI with coverage, uncertainty, or semantic proxies as the main result.],
  ),
  caption: [Literature role ledger: each source family justifies a design choice or a deferred extension; it does not broaden the thesis claim.],
) <tab:advisor-adoption-ledger>

= Roadmap, Risks, and Decisions

#figure(
  rect(width: 100%, inset: 7pt, radius: 3pt, stroke: 0.4pt + proposal-rule)[
    #grid(
      columns: (0.27fr, 0.73fr),
      column-gutter: 6pt,
      align: horizon,
      [],
      grid(
        columns: (12fr, 19fr, 31fr, 30fr, 31fr, 31fr),
        column-gutter: 0pt,
        text(size: 7.2pt, fill: proposal-muted)[May],
        text(size: 7.2pt, fill: proposal-muted)[Jun],
        text(size: 7.2pt, fill: proposal-muted)[Jul],
        text(size: 7.2pt, fill: proposal-muted)[Aug],
        text(size: 7.2pt, fill: proposal-muted)[Sep],
        text(size: 7.2pt, fill: proposal-muted)[submit],
      ),
    )
    #v(3pt)
    #gantt-row([M0 advisor agreement], 0, 12)
    #gantt-row([M1 oracle/geometry validation], 12, 21)
    #gantt-row([M2 myopic baselines + scale], 33, 21)
    #gantt-row([M3 target-RRI protocol], 54, 21)
    #gantt-row([M4 actor-visible scorer], 75, 28)
    #gantt-row([M5 lookahead + #symb.rl.qh recovery], 103, 21, critical: true)
    #gantt-row([M6 scale/escalation analysis], 124, 14)
    #gantt-row([M7 final evidence + writing], 138, 14, critical: true)
    #gantt-row([M8 submission freeze], 152, 2, critical: true)
  ],
  caption: [Typst-native thesis Gantt, 2026-04-29 to 2026-09-30. Red bars mark interpretation-critical evidence and submission freeze.],
) <fig:advisor-gantt>

Three risks decide interpretation. If M1 geometry or oracle labels fail, the thesis becomes a validation and one-step-scoring study. If target matching is sparse or ambiguous, target #RRI is reported only on validated subsets with unmatched counts. If #symb.entity.lookahead_headroom is near zero, the thesis reports no measurable non-myopic headroom for the evaluated split, target set, horizon, branch factor, and candidate distribution, then uses scaling or online-discrete tests only if they preserve the same target-#RRI supervision contract.

Open advisor decisions are deliberately narrow: final scene-level split, numeric minimum scale $(S_min,T_min,N_min)$, pass/fail threshold for #symb.entity.q_recovery, exact target-match thresholds $(tau_mu,tau_"gap")$, the first actor-visible crop descriptor ablation, and whether any external or online scaling substrate preserves comparable mesh/oracle target-#RRI labels. These choices change the evidence bar, not the thesis spine.
