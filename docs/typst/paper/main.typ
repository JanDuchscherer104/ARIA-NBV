#import "charged_ieee_local.typ": ieee
#import "@preview/booktabs:0.0.4": *
#import "/typst/shared/macros.typ": *

#show: booktabs-default-table-style

#let figures_path = "/figures/"

#show: ieee.with(
  title: [Aria-VIN-NBV: Quality-Driven Next-Best-View Planning with Egocentric Foundation Models],
  abstract: [
    Next-Best-View (NBV) planning for active 3D reconstruction must decide where to move next in
    order to maximize reconstruction quality under limited capture budgets. Existing learning-based
    planners often optimize geometric coverage proxies that can fail in cluttered indoor scenes,
    while VIN-NBV showed that directly predicting Relative Reconstruction Improvement (RRI) yields
    better view selection in object-centric settings. We describe Aria-VIN-NBV, a quality-driven NBV
    system that brings the RRI objective to egocentric indoor trajectories in the Aria ecosystem.
    Using Aria Synthetic Environments data, we build an oracle RRI pipeline that renders candidate
    depth from ground-truth meshes, fuses the resulting points with semi-dense SLAM reconstructions,
    and evaluates improvement with point-to-surface distances. On top of a frozen EFM3D EVL
    backbone, we train a lightweight ranking model that scores candidate poses with ordinal
    regression and combines local voxel evidence with view-conditioned cues derived from semidense
    projections, including mechanisms to down-weight unreliable voxel context when candidates leave
    the local grid. The paper summarizes the full pipeline, implementation diagnostics, and open
    design questions, providing a reproducible baseline for ablations and future entity-aware
    extensions.
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
