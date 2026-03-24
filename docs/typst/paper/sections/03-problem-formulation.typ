= Problem Formulation <sec:problem>

#import "../../shared/macros.typ": *

// TODO(paper-cleanup): This section partially duplicates the full oracle pipeline description
// now consolidated in Section @sec:oracle-rri (candidate generation/render/backproject/score).
// Keep Problem Formulation focused on the optimization problem + notation, and reference the
// oracle section for implementation details.

We consider an egocentric reconstruction episode with a sequence of captured
frames and poses. Let $#(symb.oracle.points) _t$ be the current reconstruction point set
at step $t$, and let $#symb.ase.mesh$ denote the ground-truth surface mesh for the
scene. At each step we sample a finite set of $#symb.shape.Nq$ candidate camera poses
$bold(q) in #symb.oracle.candidates subset "SE"(3)$ (with optional roll constraints), render
a candidate point set $#symb.oracle.points_q$ by rasterized depth-rendering
$#symb.ase.mesh$ from pose $bold(q)$, converting pixel centers to PyTorch3D's NDC
screen coordinates for unprojection, and score candidates by their expected
improvement in reconstruction quality. In practice, candidates are sampled
within a constrained shell (radius/elevation/azimuth), then pruned by collision
and free-space checks.
// <rm>
// Implementation details duplicated in Section 5 (Oracle RRI Computation). Keep Problem
// Formulation focused on the optimization problem + notation and refer to @sec:oracle-rri for
// candidate sampling / rendering / unprojection / pruning.
// </rm>

Our work focuses on two pieces: (i) computing these oracle per-candidate scores
(continuous RRI values and their corresponding ordinal bin labels), and (ii)
training a lightweight candidate scorer (VIN v3) that predicts the ordinal
labels from egocentric observations and the candidate pose. The resulting model
implements a one-step ranking policy (select the candidate with the highest
predicted expected RRI). Learning a multi-step NBV policy and/or continuous
action policy on top of this scorer remains future work.

== Chamfer distance and RRI

// <rm>
// TODO(paper-cleanup): The RRI/CD equations are repeated in Section @sec:oracle-rri; decide
// whether to keep them here (and shorten Section 5) or keep them only in Section 5 and
// reference them here.

We measure reconstruction quality using a Chamfer-style point #sym.arrow.l.r mesh distance between a point set $#symb.oracle.points$ and a mesh surface $#symb.ase.mesh$. We represent $#symb.ase.mesh$ by its triangular faces $#symb.ase.faces$ and evaluate both directional terms using squared point-to-triangle and triangle-to-point distances.

#block[#align(center)[#eqs.rri.cd]]

#block[#align(center)[#eqs.rri.acc]]

#block[#align(center)[#eqs.rri.comp]]

The Relative Reconstruction Improvement for candidate $bold(q)$ is then

#block[#align(center)[#eqs.rri.rri]]

Here $epsilon$ is a small stabilizer. A positive RRI means that adding the
candidate view decreases the Chamfer distance, thereby improving reconstruction
quality. A greedy one-step oracle policy would select
#block[#align(center)[#eqs.rri.greedy]]

This one-step selection rule is inherently myopic: it optimizes immediate
surface error reduction and ignores longer planning horizons. As a simple
thought experiment, consider a corridor with a doorway into an unseen room. A
candidate view that moves toward the doorway might yield little immediate RRI
(it still sees mostly the corridor), yet it enables a subsequent view inside
the room with a large gain. A greedy one-step rule can therefore prefer
refinements of already-observed surfaces over actions that open new regions.
Addressing this requires explicit lookahead or learning a multi-step policy,
which is outside the scope of this paper.
// </rm>

== Ordinal binning

// <rm>
// This subsection largely repeats the CORAL/quantile-binning story in Section 5 and Section 7.
// Keep only a short pointer here or remove.
Direct regression on RRI is sensitive to outliers and stage-dependent scaling
(early stages often yield larger gains). Following VIN-NBV, we discretize RRI
into $K$ ordered bins and solve an ordinal classification problem @VIN-NBV-frahm2025.
The continuous prediction is recovered by taking the expectation over the
estimated ordinal distribution.

In our implementation, we fit empirical *quantile edges* (equal-mass bins) and
assign the ordinal label by edge counting (Section @sec:oracle-rri):

#block[#align(center)[#eqs.binning.edges]]

#block[#align(center)[#eqs.binning.label]]
// </rm>
