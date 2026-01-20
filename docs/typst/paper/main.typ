#import "charged_ieee_local.typ": ieee
#import "@preview/booktabs:0.0.4": *
#import "/typst/shared/macros.typ": *

#show: booktabs-default-table-style

#let figures_path = "/figures/"

#show: ieee.with(
  title: [Aria-VIN-NBV: Quality-Driven Next-Best-View Planning with Egocentric Foundation Models],
  abstract: [
    Next-Best-View (NBV) planning selects future viewpoints for active 3D reconstruction to
    maximize reconstruction quality under limited capture budgets. Many learning-based planners
    optimize coverage- or information-gain proxies that can fail in cluttered indoor scenes with
    occlusions and fine details. We introduce Aria-VIN-NBV, a quality-driven NBV system for
    egocentric indoor trajectories in Aria Synthetic Environments (ASE). Building on the
    view-introspection paradigm of predicting Relative Reconstruction Improvement (RRI) for
    candidate views, we compute oracle labels by rendering candidate depth from ASE ground-truth
    meshes, fusing the resulting points with semi-dense simultaneous localization and mapping
    (SLAM) reconstructions, and scoring
    candidates by the relative reduction in a bidirectional point↔mesh surface error (point→mesh
    accuracy plus mesh→point completeness). With a Streamlit-based diagnostics dashboard, we
    inspect candidate distributions, depth render quality, and RRI failure modes. We document
    the oracle pipeline, evaluation
    protocol, and reproducible configurations to provide a baseline for future learning-based NBV
    policies (e.g., training a VIN-style candidate scorer on these labels) and entity-aware
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
#include "sections/10-discussion.typ"
#include "sections/10a-entity-aware.typ"
#include "sections/11-conclusion.typ"
#include "sections/12c-appendix-oracle-rri-labeler.typ"
#include "sections/12d-appendix-vin-v2-details.typ"
#include "sections/12e-appendix-optuna-analysis.typ"
#include "sections/12-appendix-gallery.typ"
#include "sections/12b-appendix-extra.typ"
