#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Related Work and Positioning

The thesis sits inside the older active-perception tradition in which the sensing action is part of the perceptual computation rather than a passive data-collection step. Bajcsy's formulation of active perception and the active-vision analysis of Aloimonos, Weiss, and Bandyopadhyay both frame motion as a way to change the constraints of perception @ActivePerception-bajcsy1988 @ActiveVision-aloimonos1988. Classical view-planning work then specialized this idea to 3D acquisition: surveys such as Scott, Roth, and Rivest and object-scanning systems such as Banta and coauthors describe the dominant generate-and-test pattern in which candidate views are sampled, scored, and selected under sensing and registration constraints @ViewPlanningSurvey-scott2003 @NBVSystem-banta2000. ARIA-NBV keeps this discretized candidate-set structure, but replaces generic unknown-area or coverage gain with an oracle reconstruction-quality target.

Modern robotic #NBV methods extend the same generate-and-test pattern toward online exploration and computational efficiency. Receding-horizon NBV plans over a sampled tree to explore 3D environments, while shadowcasting and projection-based variants reduce the cost of estimating view utility in large maps or voxel structures @RecedingHorizonNBV-bircher2016 @ShadowcastingNBV-batinovic2022 @PB-NBV-jia2025. These works are important because they show that horizon, branch factor, and evaluator cost are first-order design variables. They also clarify why ARIA-NBV should start with bounded candidate rollouts: the expensive part is not choosing an RL algorithm, but making the view utility trustworthy enough that a deeper search tree means something.

Learning-based #NBV work has recently split into continuous policy learning and discrete quality-driven ranking. GenNBV demonstrates a generalizable continuous 5-DoF policy trained with PPO-style reinforcement learning and coverage rewards @GenNBV-chen2024 @PPO-schulman2017. Hestia improves that line by decomposing continuous control into a look-at proposal and a feasible camera pose while using directional voxel-face visibility and close-greedy rewards @Hestia-lu2026. VIN-NBV instead remains closer to candidate ranking: it samples views, computes oracle #RRI, and learns to rank candidates by expected reconstruction improvement @VIN-NBV-frahm2025. The thesis follows VIN-NBV for the first contribution because #RRI is aligned with mesh-supervised reconstruction quality, then uses GenNBV and Hestia as controlled references for later continuous or hierarchical extensions.

Radiance-field and Gaussian-splatting work broadens the representation side of the literature. NeRF and 3D Gaussian Splatting made differentiable or explicit radiance-field representations central to view synthesis and reconstruction @NeRF-mildenhall2020 @GaussianSplatting-kerbl2023. ActiveNeRF, FisherRF, Next Best Sense, and recent semantic, dynamic, and object-centric 3DGS #NBV papers select views by uncertainty, Fisher information, depth uncertainty, semantics, dynamics, or object-specific utility @ActiveNeRF-pan2022 @FisherRF-jiang2024 @NextBestSense-strong2024 @li2025bestviewselectionssemantic @ObjectCentricNBV-jeong2026. These papers are not direct replacements for ARIA-NBV's oracle labels, but they support the target-aware thesis direction: utility should be decomposed by geometry, object, and task rather than collapsed into undifferentiated scene coverage.

The reinforcement-learning literature enters the thesis as a finite-candidate planning scaffold, not as a commitment to immediate actor-critic control. Trajectory Transformer shows that offline control can be decoded as a sequence problem with beam search, and Gumbel-Top-k provides a stochastic top-k mechanism for sampling diverse high-scoring sequences without replacement @TrajectoryTransformer-janner2021 @GumbelTopK-kool2019. Double DQN supplies the first target-construction lesson for ARIA-NBV: selector/evaluator separation is useful when training a masked candidate-query Transformer $Q_H$ over finite candidate sets from ASE oracle rollout data @DoubleDQN-vanHasselt2015. CQL, BCQ, Decision Transformer, IQL, deep energy-based policies, PPO, and SAC remain gated follow-up or bridge references because they expose support, reward-loop, and simulator assumptions that the thesis core should not hide @CQL-kumar2020 @BCQ-fujimoto2019 @DecisionTransformer-chen2021 @IQL-kostrikov2021 @DeepEnergyPolicies-haarnoja2017 @PPO-schulman2017 @SAC-haarnoja2018. ARIA-NBV therefore treats `ArgTopK -> ArgTop1_1 -> ... -> ArgTop1_H -> Q_H` as the thesis-grade non-myopic ladder before optional actor-critic or continuous-control work.

#figure(
  table(
    columns: (0.88fr, 1.5fr, 1.55fr),
    table.header([*Literature family*], [*Main lesson*], [*ARIA-NBV boundary*]),
    [Active perception and classical view planning],
    [Motion and viewpoint choice are part of perception; candidate views are usually generated and evaluated under constraints.],
    [Keep typed candidate sets and explicit geometry contracts, but use reconstruction-quality labels.],
    [Continuous and hierarchical #NBV policies],
    [Learned policies can operate in 5-DoF spaces when simulator, reward, and state are mature.],
    [Use as extension references after discrete RRI rollouts show headroom.],
    [Radiance-field and 3DGS active selection],
    [Uncertainty, Fisher information, semantics, dynamics, and object utility are separable channels.],
    [Borrow target-aware decomposition, not the objective, until calibrated against #RRI.],
    [Offline and stochastic planning],
    [Beam search, Gumbel-Top-k, offline value learning, and maximum-entropy policies define later baselines.],
    [Begin with deterministic bounded oracle rollout, train masked candidate-query Transformer $Q_H$ after rollout data are trusted, and gate continuous RL behind evidence.],
  ),
  caption: [Literature positioning for the thesis scope.],
) <tab:proposal-literature-positioning>
