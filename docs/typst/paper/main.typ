#import "charged_ieee_local.typ": ieee
#import "@preview/booktabs:0.0.4": *
#import "/typst/shared/macros.typ": *

#show: booktabs-default-table-style

#let figures_path = "/figures/"

#show: ieee.with(
  title: [Aria-VIN-NBV: Quality-Driven Next-Best-View Planning with Egocentric Foundation Models],
  abstract: [
    We present the current state of the Aria-VIN-NBV system, a quality-driven next-best-view (NBV)
    planner that predicts Relative Reconstruction Improvement (RRI) for candidate views in complex
    indoor scenes. Our approach leverages the Aria Synthetic Environments (ASE) dataset, a frozen
    Egocentric Voxel Lifting (EVL) backbone from EFM3D, and an oracle RRI pipeline built from
    ground-truth meshes and semi-dense SLAM points. We describe the end-to-end pipeline from
    candidate generation and depth rendering to ordinal regression with CORAL, including
    semidense view conditioning through projection statistics and frustum-aware attention, a
    trajectory encoder for history context, and voxel-reliability gating via `voxel_valid_frac`.
    The paper consolidates our implementation choices, diagnostics, and open design questions,
    and provides a reproducible blueprint for future ablations and entity-aware extensions.
  ],
  authors: (
    (
      name: "Jan Duchscherer",
      department: [Department of Computer Science & Mathematics],
      organization: [Munich University of Applied Sciences],
      location: [Munich, Germany],
      email: "j.duchscherer@hm.edu",
    ),
  ),
  index-terms: (
    "next-best-view",
    "relative reconstruction improvement",
    "egocentric foundation models",
    "EFM3D",
    "Aria Synthetic Environments",
    "ordinal regression",
  ),
  bibliography: bibliography("/references.bib"),
  figure-supplement: [Fig.],
  paper-size: "a4",
)


// #set text(font: "DejaVu Serif")
#set text(font: "New Computer Modern")


#import "/typst/shared/macros.typ": *

#include "sections/01-introduction.typ"
#include "sections/02-related-work.typ"
#include "sections/03-problem-formulation.typ"
#include "sections/04-dataset.typ"
#include "sections/05-coordinate-conventions.typ"
#include "sections/05-oracle-rri.typ"
#include "sections/06-architecture.typ"
#include "sections/08a-frustum-pooling.typ"
#include "sections/07-training-objective.typ"
#include "sections/07a-binning.typ"
#include "sections/07b-training-config.typ"
#include "sections/08-system-pipeline.typ"
#include "sections/09-diagnostics.typ"
#include "sections/09a-evaluation.typ"
#include "sections/09b-ablation.typ"
#include "sections/09c-wandb.typ"
#include "sections/10-discussion.typ"
#include "sections/10a-entity-aware.typ"
#include "sections/11-conclusion.typ"
#include "sections/12c-appendix-oracle-rri-labeler.typ"
#include "sections/12d-appendix-vin-v2-details.typ"
#include "sections/12-appendix-gallery.typ"
#include "sections/12b-appendix-extra.typ"
