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

The thesis tests whether this substrate can be upgraded from scene-level myopic scoring to target-conditioned multi-step #NBV. The core path is leakage-safe target-#RRI, an adapted target-conditioned myopic scorer over counterfactual state rows, oracle lookahead headroom, and residual dueling #symb.rl.qh over finite candidates. Online discrete interaction, continuous target-then-pose policies, and simulator-backed actor-critic are scaling and escalation research questions after the finite-candidate result, not substitutes for it.

#claim-box([Thesis claim])[
  Given an #ASE snippet, an actor-visible target record, a finite valid candidate table, and a fixed horizon, ARIA-NBV should measure target-specific point-mesh improvement, test whether oracle lookahead exposes headroom over myopic selection, and train actor-visible #symb.rl.qh to recover that headroom under oracle re-evaluation. The learned value model induces a finite-action policy $pi_Q(s_t)=op("argmax", limits: #true)_(i in cal(A)_t) #symb.rl.qh_theta (#symb.rl.s_cf0, #symb.entity.target_desc, #symb.rl.candidate_qti)$; RQ5/RQ6 then ask whether scale, online discrete interaction, or continuous target-then-pose control improves beyond this offline finite-candidate result.
]

The quantitative success rule is conditional. If #symb.entity.lookahead_headroom is positive on the evaluated split, #symb.rl.qh must improve over the learned myopic target scorer under matched oracle re-evaluation. If headroom is near zero, the thesis reports a negative planning result for the evaluated candidate distribution, horizon, branch factor, target set, and split rather than treating architecture changes as a rescue path.

= Formal Model and Research Questions

The first non-myopic experiment is a masked finite-horizon candidate-decision process, not a continuous-control problem. After fixing the root snippet, target, candidate generator, RNG seed, and transition rule, it can be treated as an MDP over valid candidate indices; without those fixtures it is better viewed as controlled counterfactual replay with stochastic candidate regeneration:

$
  cal(M)_"NBV"
  =
  (
    cal(S)^"hist",
    cal(S)^"cf0",
    cal(S)^"oracle",
    {cal(A)_t},
    T,
    r_t^e,
    gamma,
    H
  ).
$

The notation separates logged snippet state, counterfactual actor state, and privileged oracle state because the modalities available on the real egocentric trajectory differ from the geometry-only state that can be replayed after synthetic view choices. The raw historic state is the actor-visible state available on the logged #ASE trajectory:

$
  #symb.rl.s_hist
  =
  (
    #symb.obs.img_rgb,
    #symb.obs.pose,
    #symb.obs.points_semi_t,
    #symb.vin.field_evl_t,
    #symb.entity.target_hyp_pred_t
  ).
$

The symbol #symb.obs.points_t is only the accumulated fused point-set proxy after selected logged or rendered views. It does not include the frozen EVL tensor, target descriptor, directional $bb(S)^2$ memory, candidate feature table, or learned history tokens. The minimal counterfactual actor state freezes the root EVL field and updates only the fused geometry proxy, selected-view history $bold(h)_t$, remaining horizon metadata $b_t$, target descriptor, candidates, masks, and reason codes:

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

The privileged oracle state augments this current counterfactual state with GT geometry, the matched target mesh, all-candidate rendered points, and oracle labels. It is used for label generation, upper-bound planning, and evaluation only:

$
  #symb.rl.s_oracle
  =
  (
    #symb.rl.s_cf0,
    #symb.ase.mesh,
    #symb.ase.mesh_target,
    {#symb.obs.points_cand_ti}_(i=1)^(N_q),
    {r_t^e (i)}_(i=1)^(N_q)
  ).
$

A candidate row is one available candidate view at step $t$: pose, view direction, provenance, support fields, validity mask, invalid reason, and candidate features. The action is the valid row index, and $q_t=q_(t,a_t)$ is the selected candidate pose:

$
  #eqs.rl.finite_action_set
$

Invalidity is a hard constraint in the baseline model, not weak supervision: masks apply before argmax, temperature softmax, loss targets, and bootstrap maximization. True infeasibility or absent evaluation samples are hard-invalid; low immediate target support is a diagnostic unless no meaningful oracle/evaluation sample exists.

The target protocol has two named regimes. V0 uses GT OBB target input as a sanity or upper-bound path. V1 is the main actor-visible protocol: Observed Target Selection (OBS-SEL) chooses from observed or predicted target hypotheses; Predicted-Target Q (PRED-Q) conditions the one-step scorer or #symb.rl.qh on actor-visible target descriptors; Ground-Truth Target Evaluation (GT-EVAL) uses GT OBBs and target mesh crops only for labels, matching checks, and evaluation. The actor-visible descriptor is

$
  #eqs.entity.target_descriptor
$

In this descriptor, $hat(bold(B))_e$ is observed or predicted OBB geometry; $hat(bold(y))_e$ is class probabilities or class embedding; $hat(p)_e$ is confidence; $A_e^"proj"$ is projected area; $n_e^"semi"$ and $n_e^"EVL"$ are semidense and EVL support counts; and $bold(T)_e^"rel"$ is relative target pose. A compact actor-visible crop of spatial feature fields inside $hat(bold(B))_e$ is the first target-input ablation after this OBB-level path.

Target selection and target-to-GT matching are different operations. For training/root diversity, OBS-SEL may sample actor-visible target hypotheses by top-$K$ or temperature-softmax over selector scores:

$
  P(hat(e)_j | #symb.rl.s_hist)
  =
  (exp(beta u_(t,j)^"tar"))
  /
  (sum_(l in cal(E)_t^"obs") exp(beta u_(t,l)^"tar")).
$

After a target hypothesis $hat(e)$ is selected, GT-EVAL deterministically matches it to a GT target for labels and evaluation. In the match score, $kappa$ is class compatibility and $sigma$ is an observed-support compatibility term over projected area, semidense support, and EVL support:

$
  #eqs.entity.target_match_score
$

$
  #eqs.entity.target_match_selection
$

Here $mu_1$ is the best match score, $mu_2$ the runner-up score, and $g_mu$ the top-1/top-2 gap. The validated target subset is defined by symbolic acceptance filters:

$
  #eqs.entity.target_match_acceptance
$

The numeric values of $tau_mu$, $tau_"gap"$, and $tau_"support"$ are advisor decisions. Unmatched, unsupported, or ambiguous targets are target-invalid protocol cases rather than low target-#RRI examples.

Let $C_e (#symb.obs.points_t)$ denote the oracle-only crop of accumulated points to the matched target region. The target error is the target-cropped version of the VIN-NBV #RRI idea @VIN-NBV-frahm2025: point-to-mesh accuracy plus mesh-to-point completeness on the crop. In code, this is implemented by `OracleRRI.score` and `chamfer_point_mesh_batched` in `aria_nbv.rri_metrics`, with target cropping in `aria_nbv.pose_generation.target_counterfactuals`; PyTorch3D point-mesh face distances are the closest implementation reference for the low-level primitive. Completeness should be area-weighted or evaluated on uniformly sampled target-surface points so target-#RRI is not biased by mesh tessellation density.

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

#symb.entity.endpoint_gain is the primary fixed-horizon endpoint metric, #symb.entity.return_h is the rollout training return, and #symb.entity.log_gain is only an algebraic scale-sensitivity ablation. Since each immediate #RRI term is normalized by the current target error, the additive return is not algebraically identical to endpoint gain; the former supports value fitting and the latter supports fixed-horizon interpretation.

Selecting a candidate means choosing a valid index $a_t=i in cal(A)_t$ and then rendering or retrieving only $q_(t,i)$ for the transition. Greedy selection chooses the highest current score; temperature-softmax widens rollout support over valid candidate scores; Gumbel-Top-$K$ is a later diversity method if temperature-softmax produces insufficient branch diversity:

$
  P(a_t = i | s_t)
  =
  (exp(beta ell_(t,i)))
  /
  (sum_(j in cal(A)_t) exp(beta ell_(t,j))).
$

After selection, the acquired geometry is added to the current geometry:


$
  #symb.obs.points_next
  =
  #symb.obs.points_t
  union
  #symb.obs.points_cand_ti.
$

The next candidate table $cal(Q)_(t+1)$ is regenerated from the updated geometry, selected-view history, and remaining horizon metadata with the same logged mixture families, while #symb.vin.field_evl_0 remains fixed. The first mixture vocabulary is target-point or look-at candidates from the actor-visible target record, radial-towards/radial-away shell candidates for local exploration, forward-rig candidates that preserve logged trajectory direction, and bounded yaw/pitch/roll jitter with per-row strategy provenance.

The research questions form a dependency chain: objective definition, target representation, candidate support, finite-horizon planning, scale, and optional online/continuous escalation.

#figure(
  block(width: 100%)[
    #rq-node(
      [RQ1 Objective and metrics],
      [Can target-conditioned finite-candidate #NBV improve endpoint target quality under equal budget while training on additive target-#RRI returns?],
    )
    #rq-down()
    #rq-node(
      [RQ2 Target and matching],
      [Can actor-visible target descriptors and deterministic GT matching produce leakage-safe target-#RRI labels at sufficient coverage?],
    )
    #rq-down()
    #rq-node(
      [RQ3 Candidate and rollout support],
      [Do mixed target-centric/exploration candidates and stochastic rollout support produce enough valid, diverse, non-myopic training data?],
    )
    #rq-down()
    #rq-node(
      [RQ4 Headroom and #symb.rl.qh],
      [Does bounded oracle lookahead expose non-myopic headroom, and can masked residual dueling #symb.rl.qh recover it over the learned myopic scorer?],
      critical: true,
    )
    #rq-down()
    #rq-node(
      [RQ5 Scaling],
      [Which scaling path preserves mesh/oracle target-#RRI supervision beyond small trusted subsets?],
    )
    #rq-down()
    #rq-node(
      [RQ6 Online and continuous escalation],
      [When, after finite-candidate evidence, is online discrete or continuous target-then-pose control justified?],
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

The first implementation should not reuse the current seminar VIN checkpoint as-is. It should adapt the VIN-style one-step scorer to counterfactual rollout rows: #symb.obs.points_t, #symb.rl.candidate_table, #symb.rl.candidate_mask, selected history, directional memory, and budget all reflect the current rollout state. The scorer remains myopic and predicts immediate target-#RRI evidence for each candidate. #symb.rl.qh is residual around this calibrated myopic base, but the canonical handout definition is the dueling residual head below.

The CORAL interface is an explicit advisor decision. The default handout contract uses CORAL threshold logits correctly: sigmoid outputs are cumulative threshold probabilities, not softmax class probabilities. They may be passed as uncertainty/ranking features, while the scalar base $hat(r)_psi^e$ is obtained by converting threshold probabilities into class marginals and taking an expected value over monotone bin representatives:

$
  #eqs.rl.qh_coral_interface
$

Training is staged to preserve the residual interpretation: first train and calibrate $hat(r)_psi^e$, then freeze or slow-finetune it while fitting the residual #symb.rl.qh, and finally ablate whether end-to-end fine-tuning improves oracle-evaluated policy performance.

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

R6D+LFF encodes each candidate token independently. The default candidate token already includes relative pose to the target and selected-history summaries. Inspired by QCNet's query-centric relative encoding @zhou2023query, the first candidate-set interaction ablation adds target, selected-history, and other-candidate relations in candidate $i$'s local frame. Here $bold(c)_(t,i)$ and $bold(R)_(t,i)$ are the candidate camera center and orientation; $bold(p)_j$ and $bold(R)_j$ are a neighboring candidate, history, target, or scene-token representative point and orientation:

$
  #eqs.features.candidate_query_local_frame
$

The relative positional encodings are Fourier/MLP features over candidate-candidate, target-candidate, and history-candidate relations:

$
  #eqs.features.candidate_query_rpe
$

These pairwise features modulate attention keys and values rather than replacing candidate content features:

$
  #eqs.features.edge_conditioned_attention
$

This query-centric RPE and the R6D pose branch are pose-relation signals. The $bb(S)^2$ memory below is different: it summarizes which viewing directions have already observed a target-local cell and is therefore an accumulated visibility signal rather than a pairwise pose encoding.

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

Directional-memory ablations should report endpoint #symb.entity.endpoint_gain and whether selected views increase angular diversity around the matched target crop, otherwise the memory is only an architectural feature rather than tested evidence.

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

Candidate reasoning must be permutation-equivariant; a Set Transformer or masked self-attention candidate encoder is the first planned interaction model @SetTransformer-lee2019:

$
  #eqs.features.qh_set_encoder
$

No-interaction candidate MLP scoring and pooled DeepSets aggregation @DeepSets-zaheer2017 are required baselines before attributing gains to Set Transformer interaction or QCNet-style RPE.

The first planned head is a dueling residual decomposition over valid actions @DuelingDQN-wang2016. $hat(r)_psi^e$ is the calibrated myopic base; $V_theta$ is shared scene-target-history value; $A_(theta,i)^H$ is candidate-specific finite-horizon advantage; and the mean advantage is subtracted only over #symb.rl.action_set_t. The decomposition fits finite candidate tables because many candidate views can be near-duplicates in value while still requiring candidate-specific corrections:

$
  #eqs.rl.qh_dueling_residual
$

The thesis-safe order is therefore: target-conditioned one-step scorer on counterfactual rows; residual dueling #symb.rl.qh without a dense future-render branch; R6D+LFF candidate pose features; $bb(S)^2$ moment directional memory as the default accumulated-visibility signal; required MLP/DeepSets/Set-Transformer candidate encoder comparisons; QCNet-style candidate-query RPE as the first interaction ablation; low-order spherical-harmonic or histogram directional-memory ablations; richer scene/cell attention; edge-conditioned spatial GNNs; distributional #symb.rl.qh heads; and privileged-teacher distillation. Dense GT candidate renders may be used only as privileged training signals in later ablations, never as V1 actor input. Point backbones, sparse convolutions, online discrete interaction, and continuous control are lower-priority scaling or escalation steps, not the first success criterion.

= Learning Objective and Evaluation

The planned value model remains finite-candidate and masked. Candidate tokens are decoded only over valid actions:

$
  #eqs.rl.qh_candidate_token
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

The one-step scorer is trained on all valid candidate rows with oracle immediate target-#RRI labels. The bootstrapped #symb.rl.qh backup is trained only on expanded transition rows for which the selected action, successor counterfactual state, successor candidate table, successor mask, and terminal flag are materialized. If all-candidate successor expansion is unavailable, non-expanded candidates contribute myopic supervision but not bootstrapped finite-horizon targets.

Before #symb.rl.qh is interpreted as planning, the learned one-step target scorer must pass the myopic-control evidence gate: held-out ranking, oracle-evaluated model-selected rollouts, calibration and stage-shift diagnostics, and Rerun examples of representative successes and failures.

All selected actions are re-evaluated by the oracle under identical roots, candidate budgets, and acquisition budgets. Equal budget means equal selected-view horizon $H$, candidate count $N_q$, candidate-generation distribution, and validity constraints; path length, runtime, and oracle evaluation count are reported separately. Coverage is reported against the full scale bar of 100 GT-mesh ASE scenes and 4,608 snippet windows, or against an explicit scene-level held-out subset if scale is blocked. Final splits are scene-level; sample-level splitting across snippets from the same scene is not valid for final claims.

Policy comparisons are paired by root snippet, target, candidate seed, candidate budget, and horizon. Report mean, median, bootstrap confidence intervals, and per-scene win rates for #symb.entity.endpoint_gain, #symb.entity.return_h, and invalidity; aggregate tables should not hide scene-level failures behind global averages.

#figure(
  table(
    columns: (0.62fr, 1.78fr),
    table.header([*Surface*], [*Required evidence*]),
    [One-step scorer],
    [held-out rank correlation, top-$k$ oracle hit, calibration, stage-shift diagnostics, selected-candidate oracle #RRI, target visibility, Rerun success/failure examples],

    [Candidate and replay],
    [strategy provenance, path increments, scene/target support fields, validity masks, reason codes, with/without-mask rank metrics, policy metadata, seed metadata, shuffled-candidate evaluation],

    [Scale and storage],
    [scene-level splits, paired policy comparisons, bootstrap confidence intervals, per-scene win rates, no silent coverage changes, scale axes reported separately, Zarr asset references, LRZ/Slurm/DSS/resume/storage gates],
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

The important transfers are narrow. VIN-NBV motivates point-mesh #RRI labels and one-step ranking; #ASE / Project Aria / EFM3D provide the egocentric mesh-supervised substrate; greedy/submodular sensing motivates measuring oracle headroom before claiming planning gain; CORAL, set models, QCNet, and Double-Q provide the ranking, candidate-set, relative-encoding, and backup machinery; SCONE and MACARONS contrast coverage/online exploration with ARIA-NBV's mesh-supervised target-#RRI objective; Hestia contributes directional observability and target/look-at then pose factorization as a bridge, not as the thesis-core reward.

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

    [CORAL, set models, QCNet, Double-Q @CORAL-cao2019 @SetTransformer-lee2019 @zhou2023query #cite(label("DBLP:journals/corr/MnihKSGAWR13")) @DoubleDQN-vanHasselt2015],
    [Ordinal target scorer, MLP/DeepSets/Set Transformer candidate controls, query-centric relative-encoding ablation, masked fitted backups.],
    [Do not adopt QCNet trajectory decoding, motion-forecasting losses, or streaming online claims as thesis core; IQL/CQL/BCQ and distributional heads are later ablations.],

    [SCONE, MACARONS @SCONE-guedon2022 @MACARONS-guedon2023],
    [Coverage/gain prediction over partial 3D evidence and online RGB-only coverage anticipation as contrast.],
    [Do not replace mesh-supervised target-#RRI with surface coverage, occupancy, or RGB-only self-supervision.],

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

Three risks decide interpretation. If M1 geometry or oracle labels fail, the thesis becomes a validation and one-step-scoring study. If target matching is sparse or ambiguous, target #RRI is reported only on validated subsets with unmatched counts and the advisor-chosen acceptance filters. If #symb.entity.lookahead_headroom is near zero, the thesis reports no measurable non-myopic headroom for the evaluated split, target set, horizon, branch factor, and candidate distribution, then uses scaling or online-discrete tests only if they preserve the same target-#RRI supervision contract.

Open advisor decisions are deliberately narrow: final scene-level split; symbolic target-match thresholds $(tau_mu, tau_"gap", tau_"support")$; the CORAL-to-#symb.rl.qh interface; the first actor-visible crop descriptor ablation; final evidence scale and subset rule; and whether any external or online scaling substrate preserves comparable mesh/oracle target-#RRI labels. These choices change the evidence bar, not the thesis spine.
