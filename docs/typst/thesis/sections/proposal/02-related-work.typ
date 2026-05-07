#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Related Work and Positioning

The local paper manifest and shared bibliography were treated as a design
corpus, not as a citation list to pad the proposal. The scientific pattern that
emerges is consistent: older active-perception and view-planning work motivates
action-conditioned sensing; modern #NBV work splits between continuous
coverage-driven policies, quality-driven finite candidate ranking, projection
shortlists, and radiance-field uncertainty; offline RL contributes support and
overestimation warnings for finite-candidate value learning.

#figure(
  table(
    columns: (1.08fr, 1.36fr, 1.65fr),
    table.header([*Corpus family*], [*Main technical signal*], [*ARIA-NBV use*]),
    [Active perception and classical view planning @ActivePerception-bajcsy1988 @ActiveVision-aloimonos1988 @ViewPlanningSurvey-scott2003 @NBVSystem-banta2000],
    [Viewpoint choice changes what constraints are observable; candidate generation and feasibility are first-class parts of perception.],
    [Keep a finite candidate table, explicit validity, and acquisition budget instead of hiding planning behind a black-box policy.],
    [Receding, projection, and finite-candidate #NBV @RecedingHorizonNBV-bircher2016 @ShadowcastingNBV-batinovic2022 @PB-NBV-jia2025 @VIN-NBV-frahm2025],
    [Evaluator cost, horizon, branch factor, and reconstruction-quality labels determine whether deeper search is meaningful.],
    [Use mesh-supervised #RRI as the utility label; use projection/frontier ideas only as proposal or diagnostic channels.],
    [Egocentric substrate @projectaria-engel2023 @ProjectAria-ASE-2025 @EFM3D-straub2024],
    [Project Aria supplies calibrated egocentric streams; #ASE adds synthetic scale, GT meshes, OBBs, trajectories, and EVL-style 3D state.],
    [Use predicted/observed OBB and EVL support as actor-visible target state; keep GT meshes/boxes for labels and evaluation only.],
    [Continuous and hierarchical learned #NBV @GenNBV-chen2024 @Hestia-lu2026 @PPO-schulman2017],
    [Coverage-driven PPO and hierarchical look-at-then-fly policies become powerful when simulator state, reward, and dynamics are mature.],
    [Use as bridge design and baseline context after finite-candidate #RRI evidence shows headroom.],
    [Radiance-field / 3DGS active selection @NeRF-mildenhall2020 @GaussianSplatting-kerbl2023 @ActiveNeRF-pan2022 @FisherRF-jiang2024 @NextBestSense-strong2024 @li2025bestviewselectionssemantic @ObjectCentricNBV-jeong2026 @FOVHPE-bae2025],
    [Uncertainty, Fisher information, depth, semantic, dynamic, object-specific, and downstream-task utilities can be logged separately.],
    [Adopt utility-channel separation and target focus; do not replace target #RRI with 3DGS uncertainty before calibration.],
    [Offline value learning and sequence planning @TrajectoryTransformer-janner2021 @GumbelTopK-kool2019 #cite(label("DBLP:journals/corr/MnihKSGAWR13")) @DoubleDQN-vanHasselt2015 @CQL-kumar2020 @BCQ-fujimoto2019 @DecisionTransformer-chen2021 @IQL-kostrikov2021 @DeepEnergyPolicies-haarnoja2017 @SAC-haarnoja2018],
    [Replay, beam decoding, stochastic sampling without replacement, max-over-action overestimation, and offline support mismatch are the relevant hazards.],
    [Train masked fitted Double-Q $Q_H$ first; keep IQL, sequence decoding, soft/energy policies, and actor-critic control as gated ablations.],
    [Structured semantic scene representations @SceneScript-avetisyan2024 @HITL-SceneScript-xie2025],
    [Autoregressive scene languages and editable layouts support semantic/global memory and human correction.],
    [Use as future global planner context only after observed target contracts and target #RRI are stable.],
  ),
  caption: [Source-backed literature positioning for the ARIA-NBV thesis scope.],
) <tab:proposal-source-positioning>

The thesis therefore does not claim novelty through a larger continuous action
space. Its scientific contribution is the controlled replacement of proxy
utility by target-conditioned reconstruction-quality utility under a finite
candidate contract. The closest methodological lineage is

$ "coverage / uncertainty NBV"
  -> "oracle RRI candidate ranking"
  -> "target RRI"
  -> "bounded rollout"
  -> "masked finite-candidate " Q_H. $

This lineage makes three negative claims explicit. First, coverage is a useful
diagnostic but not the thesis objective. Second, GT meshes and GT target boxes
are oracle assets, not actor inputs. Third, offline/continuous RL methods are
not safe shortcuts until rollout support, masks, and evaluation are reliable.

#figure(
  image("../../figures/proposal_system_flow.png", width: 100%),
  caption: [Mermaid flowchart of the thesis evidence chain. Rectangular nodes are thesis-core gates; dashed nodes are bridge paths after $Q_H$ evidence.],
) <fig:proposal-system-flow>
