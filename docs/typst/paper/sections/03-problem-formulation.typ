= Problem Formulation <sec:problem>

#import "../../shared/macros.typ": *

We consider an egocentric reconstruction episode with a sequence of captured
frames and poses. Let $#(s.points)_t$ be the current reconstruction point set
at step $t$, and let $#s.mesh$ denote the ground-truth surface mesh for the
scene. At each step we sample a finite set of $N$ candidate camera poses
$q in #s.candidates subset "SE"(3)$ (with optional roll constraints), render
a candidate point set $#(s.points)_q$ by rasterized depth-rendering
$#s.mesh$ from pose $q$, unprojecting it in normalized device coordinates,
and score candidates by their expected improvement in reconstruction quality.

Our work to date focuses on computing these oracle per-candidate scores: the
continuous RRI values and their corresponding ordinal bin labels (the CORAL
rank). Learning a next-best-view policy on top of these labels is left to
future work.

== Chamfer distance and RRI

We measure reconstruction quality using a Chamfer-style point #sym.arrow.l.r mesh distance between a point set $#s.points$ and a mesh surface $#s.mesh$. We represent $#s.mesh$ by its triangular faces $#s.faces$ and evaluate both directional terms using squared point-to-triangle and triangle-to-point distances.

#block[
  #align(center)[
    $
      "CD"(#s.points, #s.mesh) =
      #s.acc (#s.points, #s.mesh) + #s.comp (#s.points, #s.mesh)
    $
  ]
]

#block[
  #align(center)[
    $
      #s.acc (#s.points, #s.mesh) =
      (1)/(||#s.points||) sum_(bold(p) in #s.points) min_(bold(f) in #s.faces) d(bold(p), bold(f))^2
    $
  ]
]

#block[
  #align(center)[
    $
      #s.comp (#s.points, #s.mesh) =
      (1)/(||#s.faces||) sum_(bold(f) in #s.faces) min_(bold(p) in #s.points) d(bold(p), bold(f))^2
    $
  ]
]

The Relative Reconstruction Improvement for candidate $q$ is then

#block[
  #align(center)[
    $
      "RRI"(q) =
      ("CD"(#(s.points)_t, #s.mesh) - "CD"(#(s.points)_t union #(s.points)_q, #s.mesh))
      / ("CD"(#(s.points)_t, #s.mesh) + epsilon)
    $
  ]
]

Here $epsilon$ is a small stabilizer. A positive RRI means that adding the
candidate view decreases the Chamfer distance, thereby improving reconstruction
quality. A greedy one-step oracle policy would select
#block[
  #align(center)[
    $ q_star = op("argmax", limits: #true)_(q in #s.candidates) "RRI"(q) $
  ]
]

This one-step selection rule is inherently myopic: it optimizes immediate
surface error reduction and ignores longer planning horizons. As a simple
thought experiment, consider a corridor with a doorway into an unseen room. A
candidate view that moves toward the doorway might yield little immediate RRI
(it still sees mostly the corridor), yet it enables a subsequent view inside
the room with a large gain. A greedy one-step rule can therefore prefer
refinements of already-observed surfaces over actions that open new regions.
Addressing this requires explicit lookahead or learning a multi-step policy,
which is outside the scope of this paper.

== Ordinal binning

Direct regression on RRI is sensitive to outliers and stage-dependent scaling
(early stages often yield larger gains). Following VIN-NBV, we discretize RRI
into $K$ ordered bins and solve an ordinal classification problem @VIN-NBV-frahm2025.
The continuous prediction is recovered by taking the expectation over the
estimated ordinal distribution.
