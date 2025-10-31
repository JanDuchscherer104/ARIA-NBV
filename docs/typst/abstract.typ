// Abstract for NBV project in Typst
// This file will be included in the main Quarto document

#import "shared/macros.typ": *

#let abstract-content = [
  #heading(level: 1, outlined: false)[Abstract]

  Next-Best-View (#NBV) planning addresses the fundamental challenge of autonomous viewpoint selection in active 3D reconstruction. Traditional approaches rely on hand-crafted geometric heuristics that optimize for surface coverage but fail to account for reconstruction quality, leading to suboptimal scanning strategies in complex indoor environments.

  This project develops a foundation model-enhanced #NBV system that leverages pre-trained egocentric models to predict Relative Reconstruction Improvement (#RRI) for candidate viewpoints. We replace VIN-NBV's custom CNN backbone with #EFM3D/#EVL, a pre-trained egocentric foundation model that provides 3D voxel lifting, semantic scene understanding, and multi-modal processing capabilities. This integration enables better generalization to complex multi-room scenes compared to coverage-based methods.

  Our approach uses the Aria Synthetic Environments (#ASE) dataset, containing 100,000 synthetic indoor scenes with ground truth meshes, to train an #RRI prediction network. The system combines #EVL's pre-trained 3D spatial reasoning with a lightweight prediction head that estimates reconstruction quality improvement from candidate views. By directly optimizing for reconstruction quality rather than geometric coverage, our method aims to achieve the 30% improvement demonstrated by VIN-NBV while extending to realistic indoor environments.

  Key innovations include: (1) integration of pre-trained foundation models for improved scene understanding, (2) entity-aware #RRI computation enabling semantic prioritization, and (3) evaluation on large-scale synthetic indoor scenes. The work establishes a foundation for human-in-the-loop reconstruction systems where users can specify semantic importance weights for different scene entities.
]

// Export the content
abstract-content
