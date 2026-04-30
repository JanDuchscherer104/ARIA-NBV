= Discussion and Limitations

The current Aria-VIN-NBV system demonstrates a practical path to quality-driven
NBV for egocentric scenes, but several limitations remain.
// TODO(paper-cleanup): “demonstrates a practical path” is a strong framing; either point to
// concrete empirical evidence in this paper (figures/tables) or soften wording.

== Local voxel extent

EVL produces a local voxel grid centered on the latest pose. Candidates outside
this extent receive limited voxel evidence. Semidense projection features (VIN
v3) and, in VIN v2 ablations, frustum attention mitigate the issue, yet fully
global representations remain an open challenge.
// TODO(paper-cleanup): Reference the specific diagnostics/figures that quantify “outside extent”
// cases (voxel_valid_frac distributions) instead of qualitative phrasing.

== Stage dependence and label scaling

RRI distributions shift over the course of a trajectory. Early views typically
yield larger gains than late-stage refinements. Stage-aware features or
trajectory-conditioned binning may be required for consistent calibration.
// TODO(paper-cleanup): Either provide a concrete plot/table demonstrating stage shift or
// rephrase as a hypothesis / planned ablation.

== Computational cost of oracle labels

Oracle RRI computation requires depth rendering and point-to-mesh distances for
all candidates, which is expensive and limits dataset size. This cost also
makes on-policy training of continuous 5-DoF action policies impractical in our
current system: each policy step would require multiple oracle evaluations,
and our labeler is not yet optimized for large-scale multiprocessing.
// TODO(paper-cleanup): Quantify oracle cost (sec/candidate and sec/snippet) and avoid
// unsubstantiated claims about “impractical” without measured throughput numbers.
Discretizing the action space into a candidate set amortizes oracle cost and
provides dense supervision (RRI labels for all candidates), enabling stable
offline training in future work. A learned RRI predictor can then act as a fast
surrogate objective for continuous pose search or eventual on-policy
fine-tuning.

== Entity-aware objectives

The current pipeline optimizes global reconstruction quality. ASE provides
object-level annotations and OBBs that enable entity-aware RRI. Incorporating
per-entity objectives is a promising direction for task-driven NBV.
// TODO(paper-cleanup): Cross-reference @sec:entity-aware and align notation there with `#symb`
// (avoid introducing a third “RRI_total” notation variant).

// TODO: Add measured oracle runtimes (per snippet / per candidate) once
// profiling results are finalized.
