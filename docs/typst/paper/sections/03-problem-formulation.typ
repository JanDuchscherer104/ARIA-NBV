= Problem Formulation

#import "/typst/shared/macros.typ": *

We consider an egocentric reconstruction episode with a sequence of captured frames and poses. Let $#sym_points _t$ be the current reconstruction point set at step $t$, and let $#sym_mesh$ denote the ground-truth surface mesh for the scene. At each step we sample a set of $N$ candidate camera poses $(q_i)_{i=1}^N in "SE"(3)$ (with optional roll constraints), render candidate point sets $#sym_points _(q_i)$, and select the view that maximizes the expected improvement in reconstruction quality.

== Chamfer distance and RRI

We measure reconstruction quality using the symmetric Chamfer distance between a point set $#sym_points$ and a mesh surface $#sym_mesh$. We sample points from the mesh surface and compute the bidirectional distance

#block[
  #align(center)[
    $
      "CD"(#sym_points, #sym_mesh) =
      (#sym_acc)(#sym_points, #sym_mesh) + (#sym_comp)(#sym_points, #sym_mesh)
    $
  ]
]

#block[
  #align(center)[
    $
      (#sym_acc)(#sym_points, #sym_mesh) =
      (1)/(n_P) sum_(bold(p) in #sym_points) min_(bold(m) in #sym_mesh) d(bold(p), bold(m))^2
    $
  ]
]

#block[
  #align(center)[
    $
      (#sym_comp)(#sym_points, #sym_mesh) =
      (1)/(n_M) sum_(bold(m) in #sym_mesh) min_(bold(p) in #sym_points) d(bold(m), bold(p))^2
    $
  ]
]

where $n_P$ and $n_M$ denote the point counts. The Relative Reconstruction Improvement for candidate $q$ is then

#block[
  #align(center)[
    $
      "RRI"(q) =
      ("CD"(#sym_points _t, #sym_mesh) - "CD"(#sym_points _t union #sym_points _q, #sym_mesh))
      / ("CD"(#sym_points _t, #sym_mesh) + epsilon)
    $
  ]
]

Here $epsilon$ is a small stabilizer. A positive RRI means that adding the
candidate view decreases the Chamfer distance, thereby improving reconstruction
quality. Our policy selects

#block[
  #align(center)[
    $ q_star = "argmax" "RRI"(q) $
  ]
]

== Ordinal binning

Direct regression on RRI is sensitive to outliers and stage-dependent scaling
(early stages often yield larger gains). Following VIN-NBV, we discretize RRI
into $K$ ordered bins and solve an ordinal classification problem @VIN-NBV-frahm2025.
The continuous prediction is recovered by taking the expectation over the
estimated ordinal distribution.
