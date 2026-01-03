= Discussion and Limitations

The current Aria-VIN-NBV system demonstrates a practical path to quality-driven
NBV for egocentric scenes, but several limitations remain.

== Local voxel extent

EVL produces a local voxel grid centered on the latest pose. Candidates outside
this extent receive limited voxel evidence. Semidense projection features and
frustum attention mitigate the issue, yet fully global representations remain an
open challenge.

== Stage dependence and label scaling

RRI distributions shift over the course of a trajectory. Early views typically
yield larger gains than late-stage refinements. Stage-aware features or
trajectory-conditioned binning may be required for consistent calibration.

== Computational cost of oracle labels

Oracle RRI computation requires depth rendering and point-to-mesh distances for
all candidates, which is expensive and limits dataset size. Approximate labels
or learned proxies could reduce this bottleneck.

== Entity-aware objectives

The current pipeline optimizes global reconstruction quality. ASE provides
object-level annotations and OBBs that enable entity-aware RRI. Incorporating
per-entity objectives is a promising direction for task-driven NBV.
