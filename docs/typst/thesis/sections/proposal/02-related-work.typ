#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Related Work and Positioning

The literature is used here to assign roles, not to broaden the thesis claim.
Older active perception motivates action-conditioned sensing; VIN-NBV supplies
the quality-driven candidate-ranking precedent; Project Aria, #ASE, and EFM3D
define the egocentric state; and offline value learning supplies replay and
overestimation controls for the finite candidate table.

#figure(
  table(
    columns: (1.05fr, 1.32fr, 1.48fr),
    table.header([*Role*], [*Relevant signal*], [*Adopt / defer*]),
    [Quality-driven #NBV @VIN-NBV-frahm2025 @CORAL-cao2019],
    [Oracle #RRI and ordinal one-step candidate ranking are the closest implemented precedent.],
    [Adopt point-mesh #RRI labels and a learned one-step target scorer; test whether one-step ranking is enough.],
    [Egocentric substrate @projectaria-engel2023 @ProjectAria-ASE-2025 @EFM3D-straub2024],
    [Calibrated egocentric streams, semi-dense points, predicted OBBs, EVL fields, and GT meshes coexist.],
    [Use observed/predicted target descriptors as actor input; keep GT geometry for labels and evaluation.],
    [Greedy sensing and finite candidates @RecedingHorizonNBV-bircher2016 @PB-NBV-jia2025 @KrauseSensorPlacement2008 @AdaptiveSubmodularity-golovin2011 @SubmodularNBV-lauri2020],
    [When utility has diminishing returns, greedy selection can be strong; deeper search must earn its cost empirically.],
    [Measure oracle-lookahead headroom before claiming a learnable non-myopic advantage.],
    [Continuous and radiance-field #NBV @GenNBV-chen2024 @Hestia-lu2026 @ActiveNeRF-pan2022 @FisherRF-jiang2024 @ObjectCentricNBV-jeong2026],
    [Continuous policies, target-then-pose hierarchies, and uncertainty/semantic utility channels are useful comparisons.],
    [Use as follow-up design pressure; do not replace target #RRI with coverage or uncertainty rewards.],
    [Finite-action value learning #cite(label("DBLP:journals/corr/MnihKSGAWR13")) @DoubleDQN-vanHasselt2015 @IQL-kostrikov2021 @Transformer-vaswani2017 @DeepSets-zaheer2017 @SetTransformer-lee2019],
    [Replay, masked Bellman targets, overestimation control, offline support, and permutation-aware candidate-token modeling shape $Q_H$.],
    [Train masked fitted Double-Q first; keep IQL, sequence decoding, and continuous actor-critic variants as later ablations.],
  ),
  caption: [Source-backed literature roles for the proposal scope.],
) <tab:proposal-source-positioning>

The resulting lineage is deliberately narrow:

$ cal(U)_"cov/unc" -> hat(r)_t^e (i) -> r_t^e -> G_t^((H)) -> Q_(H,theta). $

Coverage and uncertainty remain diagnostics, not the thesis utility. GT meshes
and GT target boxes remain oracle assets, not V1 actor inputs. Offline and
continuous RL references become meaningful only after candidate support, masks,
and oracle re-evaluation are trustworthy.

#figure(
  align(center, image("../../figures/proposal_system_flow.png", width: 82%)),
  caption: [Compact evidence chain. The proposal moves from actor-visible $bold(s)_t^"obs"$ and $bold(z)_e$ through masked candidates, target #RRI, lookahead headroom $Delta_"look"$, and masked $Q_H$; the dashed path is post-$Q_H$ follow-up work.],
) <fig:proposal-system-flow>
