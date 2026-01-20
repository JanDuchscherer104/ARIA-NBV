= Conclusion

We presented an oracle supervision and diagnostics pipeline for quality-driven
NBV research in egocentric indoor scenes. The approach computes per-candidate
oracle RRI labels from ASE ground-truth meshes and semi-dense reconstructions,
and provides tooling to inspect candidate generation, depth rendering, and
surface-error behavior. We also outlined an ordinal label representation
(CORAL) and a candidate-scoring architecture as an implementation sketch.
Learning a next-best-view policy on top of these labels is left to future work,
including entity-aware objectives, stage-adaptive labeling, and more expressive
view-conditioned representations.
