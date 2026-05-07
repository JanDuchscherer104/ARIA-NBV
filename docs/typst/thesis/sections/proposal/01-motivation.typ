#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Motivation

Active reconstruction is a coupled sensing and inference problem: the
reconstruction error after a fixed budget depends as much on the selected
viewpoints as on the downstream surface estimator. Classical active perception
therefore treats sensing actions as part of perception itself, and view-planning
surveys formalize the dominant generate-score-select loop for three-dimensional
inspection @ActivePerception-bajcsy1988 @ActiveVision-aloimonos1988
@ViewPlanningSurvey-scott2003. ARIA-NBV adopts that loop for egocentric indoor
data, but asks a narrower scientific question: can finite candidate views be
scored and planned by reconstruction-quality improvement rather than by coverage
or uncertainty proxies alone?

The key empirical precedent is VIN-NBV, which replaces pure coverage with
Relative Reconstruction Improvement (#RRI), an oracle label computed from the
reduction in Chamfer-style reconstruction error after adding a query view
@VIN-NBV-frahm2025. This is the right axis for ARIA-NBV because target indoor
surfaces can remain poorly reconstructed even when many voxels or pixels are
already covered. The thesis extends this idea from object-centric RGB-D
benchmarks to the Project Aria / #ASE ecosystem, where calibrated egocentric
streams, trajectories, semi-dense points, predicted object boxes, and a
mesh-supervised subset support controlled oracle labels @projectaria-engel2023
@ProjectAria-ASE-2025 @EFM3D-straub2024.

The current gap is not a lack of possible algorithms; it is the lack of a
trustworthy target-conditioned, multi-step evidence chain. GenNBV and Hestia
show that continuous 5-DoF and hierarchical #NBV policies are plausible when an
online simulator and coverage reward are mature @GenNBV-chen2024
@Hestia-lu2026. Radiance-field and Gaussian-splatting papers show that view
utility can be decomposed into uncertainty, Fisher information, semantics,
dynamics, object identity, or downstream task error @ActiveNeRF-pan2022
@FisherRF-jiang2024 @NextBestSense-strong2024
@li2025bestviewselectionssemantic @ObjectCentricNBV-jeong2026 @FOVHPE-bae2025.
ARIA-NBV should use those papers to sharpen the model, not to replace its
objective: the thesis utility remains scene and target #RRI, with validity and
cost reported as separate constraints.

#thesis-box([Thesis position])[
  The thesis contribution is a target-conditioned, quality-driven finite-candidate
  #NBV study on #ASE/EFM. The hard result is a candidate-query $Q_H$ model that
  predicts bounded cumulative target #RRI over valid candidates and improves on
  one-step greedy/model scoring under the same acquisition budget. Continuous
  actor-critic control, Gymnasium/SB3, Habitat/Isaac, SceneScript-style global
  memory, and real-device guidance are bridge or future work unless this
  finite-candidate result is already stable.
]
