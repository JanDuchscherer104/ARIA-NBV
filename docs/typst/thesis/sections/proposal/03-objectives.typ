#import "../../../shared/macros.typ": *

= Objectives

The thesis objective is a reproducible target-aware finite-candidate #NBV stack for egocentric reconstruction. The evidence gates are: a passable M1 offline-store/oracle contract; inspectable target #RRI under V0 sanity and V1 OBS-SEL / PRED-Q / GT-EVAL; a target-conditioned one-step scorer evaluated by held-out ranking, top-k oracle hit, calibration, and failure cases; and candidate-query $Q_H$ trained from random-valid, oracle-greedy/lookahead, and oracle-scored temperature-softmax traces. The exit condition is oracle-evaluated $Q_H$ actions beating one-step greedy/model scoring under equal acquisition budget, or a documented blocker.

The required first target input is observed or predicted OBB geometry plus class, confidence, projected area, semidense support, EVL support, and relative pose. Crop descriptors and entity tokens remain ablation-ready fields. The counterfactual state for $Q_H$ is geometry-only: frozen logged EFM/EVL context, accumulated rendered/fused points, candidate metadata, and selected-view history. Invalid candidates are hard-masked with reason codes; invalidity is not a low-#RRI class.

Coverage is part of the claim. Full thesis-scale generation targets the 100 GT-mesh ASE scenes and 4,608 snippet windows after small-subset correctness passes. A first thesis-grade result may use a scene-level held-out subset with multiple targets and trajectories, but every result must report scenes, snippets, targets, trajectories, rollout seeds, transitions, and missing gaps separately.
