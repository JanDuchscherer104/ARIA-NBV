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

ARIA-NBV starts from the implemented seminar substrate: Project Aria / #ASE snippets with poses, calibration, semidense geometry, GT meshes, EFM3D/EVL scene encodings @projectaria-engel2023 @ProjectAria-ASE-2025 @EFM3D-straub2024, scene-level oracle #RRI labels, finite candidate tables, and a VIN-style one-step scorer inspired by quality-driven #NBV ranking @VIN-NBV-frahm2025. What is not yet thesis evidence is target-specific supervision, actor-visible target selection, trusted multi-step rollout data, or a learned finite-horizon value model or policy.

The thesis tests whether this substrate can be upgraded from scene-level myopic scoring to _target-conditioned multi-step #NBV _. The core path is leakage-safe target-#RRI, an adapted target-conditioned myopic scorer over counterfactual state rows, oracle lookahead headroom, and residual dueling #symb.rl.qh over finite candidates. Online discrete interaction, continuous target-then-pose policies, and simulator-backed actor-critic remain bridge decisions after the finite-candidate result.

// TODO: How is qh actor style? also point to the question: How can we scale the offline learning approach? And do we get further improvement by on-policy online training + optionally by a continuous-control policy? The ~...~ sentence should be revised. don't point to a negative outcome!
#claim-box([Thesis claim])[
  Given an #ASE snippet, an actor-visible target record, a finite valid candidate table, and a fixed acquisition budget, ARIA-NBV should be able to measure target-specific point-mesh improvement, test whether oracle lookahead exposes headroom over a myopic model, and train actor-visible #symb.rl.qh to recover part of that headroom under oracle re-evaluation. ~A negative outcome is still informative when it localizes failure to target matching, one-step target signal, candidate support, or absence of measurable lookahead headroom.~
]

= Formal Model and Research Questions
// TODO: shouldn't we start with defining the MDP? Drop / demote aquisition budget for now!

// TODO: what does #symb.obs.points_t represent more precisely? we need to denote that! do we mean just the point cloud or pointcloud with enriched features (i.e. S^2 frequency histogram!?)
// TODO: what do we mean with target descriptor z_e, h_t... provide a bit more context!
Let #symb.obs.points_t denote the accumulated actor-visible geometry proxy at decision step $t$, initialized from logged semidense reconstruction and updated during counterfactual rollouts. Let #symb.vin.field_evl_0 be the frozen root egocentric evidence field, #symb.entity.target_desc the actor-visible target descriptor, $bold(h)_t$ a compact selected-view history embedding, and $b_t$ the remaining acquisition budget. The actor-visible state is


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

// TODO: why {#symb.ase.mesh_target}_(e in cal(E)) instead of #symb.ase.mesh_target_e given that its only one targe per state!?
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

// TODO(^1): we need to clarify s^cf0 vs s^obs - i.e. difference in state space for myopic single step w.r.t. efm snippet and geometry only encodings for the counterfactuals, and also concisely point out the issue of different stat spaces for historic ego traj and counterfactual rollouts!
// TODO: don't use terms like V1 / V0 if not clearly defined!
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

// TODO(^1): difference between historic trajectory and counterfactual views. clearly term counter factual state space - how should we optimally denote it given that it is a superset of the historic ego traj state space?
// TODO: what do we mean by candidate rows? we mean the available candidate views at a given step t!
The counterfactual state uses the same root EVL field but updates geometry, history, budget, candidate rows, masks, and reason metadata after selected views. The finite action set is the masked subset of candidate rows:

$
  #eqs.rl.finite_action_set
$
// TODO: what is meant by OBS-SEL, PRED-Q, GT-EVAL? We need to clarify these terms (also those must be gls entries!)
// TODO: Invalidity is a constraint, not weak supervision in the baseline model!
Invalidity is a constraint, not weak supervision: masks apply before argmax, temperature softmax, loss targets, and bootstrap maximization. True infeasibility or absent evaluation samples are hard-invalid; low immediate target support is a diagnostic unless no meaningful oracle/evaluation sample exists. V1 uses OBS-SEL, PRED-Q, and GT-EVAL. The actor-visible target-descriptor is

$
  #eqs.entity.target_descriptor
$
// TODO: elaborate on the different terms in the target descriptor! so list the symbols  together with the following NLP description / terms.

It bundles observed or predicted OBB geometry, class, confidence, projected area, semidense support, EVL support, and relative pose. A compact actor-visible crop of spatial feature fields could be the first target-input ablation after this OBB-level path.
// TODO: GT crops are selected sounds stupid. rather Target selection is based on ..., protocal-level matching also sounds sloppy!
GT crops are selected only after protocol-level matching. In the match score, $kappa$ is class compatibility and $sigma$ is an observed-support compatibility term over projected area, semidense support, and EVL support:

$
  #eqs.entity.target_match_score
$

// TODO: I don't quite understand this equation. Shouldn't we do temperature-softmax over the scores to allow for a greater variety?
$
  #eqs.entity.target_match_selection
$

// TODO: Do we really need the target_match_acceptance predicate?
$
  #eqs.entity.target_match_acceptance
$

Here $mu_1$ is the best match score, $mu_2$ the runner-up score, and $g_mu$ the top-1/top-2 gap. The binary predicate $a_"match"$ records whether the observed target is class-compatible, sufficiently supported, and unique under the matching protocol. Unmatched, unsupported, or ambiguous targets are target-invalid protocol cases rather than low target-#RRI examples.

// TODO: include link t the implementation / cite the cloud to mesh metric source!
// TODO: also point out that #eqs.entity.target_error is pretty much target cropped RRI and cite vin paper!
// TODO: point out that it's fixed horizon metrics!
Let $C_e (#symb.obs.points_t)$ denote the oracle-only crop of accumulated points to the matched target region. The target error is the implemented point-mesh accuracy plus mesh-to-point completeness diagnostic on this target crop:

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

// TODO: how did we find inspiration for #symb.entity.log_gain? We need a citation for this one!
#symb.entity.endpoint_gain is the primary endpoint metric, #symb.entity.return_h is the rollout training return, and #symb.entity.log_gain is an ablation for scale sensitivity only. Since each immediate #RRI term is normalized by the current target error, the additive return is not algebraically identical to endpoint gain; the former supports value fitting and the latter supports fixed-budget interpretation.

// TODO: clarify - what do we mean by *selecting* a candidate here? point out that for now softmax+temperature, greed -> potentially gumbel-top-k (get context from proposal or other sources)
After selecting a valid candidate, the acquired geometry is added to the current geometry:


$
  #symb.obs.points_next
  =
  #symb.obs.points_t
  union
  #symb.obs.points_cand_ti.
$
// TODO: what do we mean by "mixture families" - also include a brief description of target generation - i.e. mixture of look-away, look-at,
The next candidate table $cal(Q)_(t+1)$ is regenerated from the updated geometry, selected-view history, and remaining budget with the same logged mixture families, while #symb.vin.field_evl_0 remains fixed.

// TODO: bad wording "In advisor-facing terms". "can target labels be measured without leakage" is stupid and vague.
Six research questions form a dependency chain. In advisor-facing terms they reduce to three empirical dependencies: can target labels be measured without leakage, can actor-visible myopic scoring rank target-RRI candidates, and does non-myopic evidence justify and support #symb.rl.qh beyond that myopic control?

// TODO: the RQs are too vague! Consider @questions.qmd and .agents/work/proposal-review/questions-drift-feedback-gpt55pro.md and clarify! Also not all contents of these "RQs" actually appears to be RQs in the meaning of the word - $grill-me! there again seemes to have been some regression w.r.t. our research questions!
#figure(
  block(width: 100%)[
    #rq-node(
      [RQ1 Objective and metrics],
      [learn target-conditioned finite-candidate multi-step #NBV and separate #symb.entity.endpoint_gain for endpoint evaluation, #symb.entity.return_h for value learning, and #symb.entity.log_gain as ablation],
    )
    #rq-down()
    #rq-node(
      [RQ2 Target and matching],
      [OBB plus support is the V1 baseline; actor-visible crop descriptors are the first ablation; #symb.entity.target_desc is matched to GT only for labels and evaluation],
    )
    #rq-down()
    #rq-node(
      [RQ3 Candidate and rollout support],
      [mixed target-centric plus exploration candidates, branch/beam knobs, and stochastic support traces define the finite action rows available to learning],
    )
    #rq-down()
    #rq-node(
      [RQ4 Headroom and #symb.rl.qh],
      [oracle lookahead optimizes cumulative target-#RRI, and actor-visible #symb.rl.qh must beat validated myopic scoring when headroom is positive],
      critical: true,
    )
    #rq-down()
    #rq-node(
      [RQ5 Scaling],
      [scale ASE finite-candidate evidence first, then external mesh/oracle-compatible substrates and online discrete #symb.rl.qh if the supervision contract remains comparable],
    )
    #rq-down()
    #rq-node(
      [RQ6 Online and continuous escalation],
      [continuous target-then-pose actor-critic is time-permitting after online training evidence; imitation variants are deferred beyond planned #RRI + #symb.rl.qh],
    )
  ],
  caption: [Causal RQ dependency chain. Later questions should not be used to rescue failed earlier contracts.],
) <fig:advisor-rq-dag>

Oracle lookahead selects by cumulative target-#RRI and endpoint gain evaluates the resulting trajectory. Report #symb.entity.q_recovery only when the oracle-lookahead denominator is positive; otherwise report raw endpoint gains and the no-headroom condition. The headroom and recovery quantities are

$
  #eqs.entity.lookahead_headroom
$

$
  #eqs.entity.q_recovery
$

= Planned Value-Model Design

This section is subordinate to the oracle and headroom tests. The research object is the finite-candidate value model #symb.rl.qh, with a permutation-equivariant candidate-set encoder as the first implementation. The learned model must map each valid candidate row to a finite-horizon value using only actor-visible scene, target, selected-history, budget, candidate, mask, and reason-code features. It is evaluated by oracle re-scoring rather than by its own predicted values.

The first implementation should not reuse the current seminar VIN checkpoint as-is. It should adapt the VIN-style one-step scorer to counterfactual rollout rows: #symb.obs.points_t, #symb.rl.candidate_table, #symb.rl.candidate_mask, selected history, directional memory, and budget all reflect the current rollout state. The scorer remains myopic and predicts immediate target-#RRI evidence for each candidate. The finite-horizon model is residual:

$
  #eqs.rl.qh_residual
$

Here $hat(r)_psi^e$ is the adapted target-conditioned one-step scorer, and $A_theta^H$ is the finite-horizon residual advantage for candidate row $i$. The residual advantage is initialized near zero, so the first #symb.rl.qh behaves like the myopic scorer until rollout evidence supports a planning correction. The CORAL interface is an explicit advisor decision: use calibrated ordinal output as the scalar myopic base, pass CORAL logits/probabilities as candidate features, or use both. The default handout contract is to retain ordinal probabilities as candidate features and optionally derive $hat(r)_psi^e$ by monotone calibration:

$
  #eqs.rl.qh_coral_interface
$

The scene memory uses the frozen EVL field, accumulated actor-visible geometry, target support, and directional memory. The tensors $bold(V)(#symb.obs.points_t)$, $bold(V)_"dir"(#symb.obs.points_t)$, and $bold(V)_"target"(#symb.entity.target_desc)$ denote voxelized accumulated geometry, directional-observation channels, and target-support channels:

$
  #eqs.features.qh_scene_memory
$

The target token should not be a descriptor vector alone; it reads a target-local actor-visible crop from scene memory. In this equation $hat(bold(B))_e$ is the observed or predicted target OBB, not a GT crop:

$
  #eqs.features.qh_target_token
$

Candidate orientation and accumulated visibility are separate signals. The pose branch uses the continuous R6D rotation representation @zhou2019continuity. In the pose feature, $bold(t)_(t,i)$ is candidate translation, $bold(R)_(t,i)^"6D"$ is candidate orientation, $bold(t)_e$ is the target center or support centroid, $alpha_(t,i)^e$ is the target-bearing or incidence proxy, $l_(t,i)$ is acquisition/path cost, and $c_(t,i)^"strategy"$ encodes candidate-strategy provenance:

$
  #eqs.features.candidate_pose_features
$

Accumulated visibility is a separate actor-visible directional memory on $bb(S)^2$. Let $bold(v)$ be a voxel, point cell, or target-local cell; $bold(c)_k$ the camera center selected at step $k$; and $w_k(bold(v))$ a visibility/support weight. The first implementation uses a second-moment directional memory:

$
  #eqs.features.direction_unit
$

$
  #eqs.features.direction_memory_moment
$

Candidate novelty then asks whether the candidate observes the cell from a direction already represented in the memory:

$
  #eqs.features.direction_novelty
$

Low-order spherical harmonics provide the richer directional-memory ablation @e3nn-SphericalHarmonics-2025:

$
  #eqs.features.direction_memory_sh
$

Each candidate row then receives pose, target-relative, frustum, belief-render, directional novelty, mask/reason, and history features. Here $bold(p)_(t,i)$ is pose/target context, $bold(g)_(t,i)$ is geometry/frustum context, $rho_(t,i)$ is the invalid-reason code, and $bold(H)_t$ is a learned selected-history token representation distinct from the horizon scalar $H$:

$
  #eqs.features.candidate_pose_context
$

$
  #eqs.features.candidate_geometry_context
$

$
  #eqs.features.candidate_row_features
$

Candidate reasoning must be permutation-equivariant; a Set Transformer candidate encoder is the first planned implementation @SetTransformer-lee2019:

$
  #eqs.features.qh_set_encoder
$

The first planned head is a dueling residual decomposition over valid actions @DuelingDQN-wang2016. $V_theta$ is shared scene-target-history value; $A_(theta,i)^H$ is candidate-specific finite-horizon advantage; the mean advantage is subtracted only over #symb.rl.action_set_t:

$
  #eqs.rl.qh_dueling_residual
$

The thesis-safe order is therefore: target-conditioned one-step scorer on counterfactual rows, residual dueling #symb.rl.qh without a dense future-render branch, actor-visible belief-render branch, R6D pose features, $bb(S)^2$ moment directional memory, low-order spherical-harmonic memory, candidate-set interaction ablations, edge-conditioned spatial GNNs, distributional #symb.rl.qh heads, and privileged-teacher distillation. Dense GT candidate renders may be used only as privileged training signals in later ablations, never as V1 actor input. Point backbones, sparse convolutions, online discrete interaction, and continuous control are lower-priority scaling or escalation steps, not the first success criterion.

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

The replay dataset $cal(D)$ contains selected-action transition rows with state, action, immediate target reward, successor state, validity masks, and terminal flags:

$
  #eqs.rl.qh_loss
$

Before #symb.rl.qh is interpreted as planning, the learned one-step target scorer must pass the myopic-control evidence gate: held-out ranking, oracle-evaluated model-selected rollouts, calibration and stage-shift diagnostics, and Rerun examples of representative successes and failures.

All selected actions are re-evaluated by the oracle under identical roots, candidate budgets, and acquisition budgets. Equal budget means equal selected-view horizon $H$, candidate count $N_q$, candidate-generation distribution, and validity constraints; path length, runtime, and oracle evaluation count are reported separately. Coverage is reported against the full scale bar of 100 GT-mesh ASE scenes and 4,608 snippet windows, or against an explicit scene-level held-out subset if scale is blocked. Final splits are scene-level; sample-level splitting across snippets from the same scene is not valid for final claims.

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

The important transfers are narrow. VIN-NBV motivates point-mesh #RRI labels and one-step ranking; #ASE / Project Aria / EFM3D provide the egocentric mesh-supervised substrate; greedy/submodular sensing motivates measuring oracle headroom before claiming planning gain; CORAL, set models, and Double-Q provide the ranking, candidate-set, and backup machinery; Hestia contributes directional observability and target/look-at then pose factorization as a bridge, not as the thesis-core reward.

#figure(
  table(
    columns: (0.9fr, 1.72fr, 1.18fr),
    table.header([*Source family*], [*Adopt for ARIA-NBV*], [*Boundary*]),
    [VIN-NBV @VIN-NBV-frahm2025],
    [Oracle point-mesh #RRI labels, ordinal target-RRI ranking, learned myopic control.],
    [One-step greedy is not a multi-step thesis claim.],

    [Project Aria / #ASE / EFM3D @projectaria-engel2023 @ProjectAria-ASE-2025 @EFM3D-straub2024],
    [Poses, calibration, semidense points, frozen EVL/EFM evidence, OBB predictions.],
    [GT geometry and GT OBBs are labels/evaluation only.],

    [Greedy sensing @KrauseSensorPlacement2008 @AdaptiveSubmodularity-golovin2011],
    [Diminishing-returns intuition; oracle-lookahead headroom as a required test.],
    [Do not assume non-myopic learning is useful without headroom.],

    [CORAL, set models, Double-Q @CORAL-cao2019 @SetTransformer-lee2019 #cite(label("DBLP:journals/corr/MnihKSGAWR13")) @DoubleDQN-vanHasselt2015],
    [Ordinal target scorer, permutation-equivariant candidate reasoning, masked fitted backups.],
    [IQL/CQL/BCQ and distributional heads are later ablations.],

    [GenNBV, Hestia, 3DGS, SceneScript @GenNBV-chen2024 @Hestia-lu2026 @FisherRF-jiang2024 @SceneScript-avetisyan2024],
    [Continuous-control pressure, directional face/visibility memory, semantic/global-memory bridge ideas.],
    [Do not replace target point-mesh #RRI with proxy coverage or semantics.],
  ),
  caption: [Literature role ledger. Each source family justifies one design choice or deferred bridge; it does not broaden the thesis claim.],
) <tab:advisor-adoption-ledger>

Hestia's transferable hierarchy is a post-#symb.rl.qh bridge: first propose a target or look-at point, then choose a feasible pose conditioned on it @Hestia-lu2026. ARIA-NBV would keep target-#RRI as supervision/evaluation and use feasibility projection or masks as constraints:

$
  #eqs.rl.target_pose_factorization
$

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

Open advisor decisions are deliberately narrow: final scene-level split, target-match acceptance protocol, the CORAL-to-#symb.rl.qh interface, the first actor-visible crop descriptor ablation, final evidence scale, and whether any external or online scaling substrate preserves comparable mesh/oracle target-#RRI labels. These choices change the evidence bar, not the thesis spine.
