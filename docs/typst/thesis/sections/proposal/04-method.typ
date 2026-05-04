#import "../../../shared/macros.typ": *

= Proposed Method

The proposed method follows a staged diagnostics-first pipeline. The first stage fixes the data substrate. ARIA-NBV reads fixed-length #ASE snippets through the active data-handling path, builds immutable VIN offline stores for expensive oracle products, and validates that train and validation samples expose the fields needed for reconstruction-quality supervision. The sample contract includes scene and snippet metadata, reference pose, candidate poses, camera intrinsics or PyTorch3D cameras, semi-dense points, candidate validity information, oracle #RRI, accuracy and completeness components, and optional diagnostics such as candidate point clouds, depth maps, meshes, object boxes, and voxel bounds.

The second stage computes scene-level and target-level oracle labels. Scene-level #RRI follows VIN-NBV by evaluating how much a candidate view reduces the Chamfer-style distance between the current reconstruction and the ground-truth mesh @VIN-NBV-frahm2025. Target-level #RRI uses the same metric but restricts the evaluation to the selected target crop. The first target definition is a ground-truth OBB crop because it is inspectable and directly available in the #ASE/EFM3D data. Predicted boxes, target tokens, or open-vocabulary selection are treated as realism ablations after the ground-truth target oracle is trusted.

#block[#align(center)[#eqs.entity.objective]]

The third stage trains and analyzes a VIN-style candidate scorer. The current implementation already uses frozen EVL/EFM3D features, candidate-relative pose information, semi-dense projection cues, and an ordinal prediction head. The thesis will keep this architecture bounded and evidence-driven. It will first establish the reproducible one-step scene-level baseline, then add target conditioning through explicit target fields. The model will be judged by ranking and calibration metrics on held-out snippets, not by loss alone. This is important because #NBV acts through ordering: a model that predicts approximate scalar values but ranks candidates poorly is not useful for view selection.

The fourth stage evaluates bounded non-myopic planning. For a state $s_t$ and valid candidate set $C(s_t)$, the one-step oracle baseline selects the best immediate candidate. A bounded rollout first chooses the top $K$ candidates under the one-step score and searches only those branches to horizon $H$. This is closer to receding-horizon and projection-efficient #NBV planning than to unrestricted policy learning: horizon and branch factor are explicit compute budgets, and every transition can be inspected @RecedingHorizonNBV-bircher2016 @PB-NBV-jia2025. The value recursion is

#block[#align(center)[
  $ V_h(s_t) = max_(q in "ArgTopK"(s_t)) (r(s_t, q) + V_(h-1)(T(s_t, q))) $
]]

and the selected action is

#block[#align(center)[
  $ "ArgTop1"_h(s_t) = op("argmax", limits: #true)_(q in "ArgTopK"(s_t)) (r(s_t, q) + V_(h-1)(T(s_t, q))) $.
]]

This formulation imports the useful part of sequence-model planning without immediately training a trajectory model. Trajectory Transformer shows that offline control can be treated as sequence decoding with beam search @TrajectoryTransformer-janner2021, while Gumbel-Top-k provides a principled stochastic beam extension for diverse samples without replacement @GumbelTopK-kool2019. ARIA-NBV will first use deterministic bounded rollout because it is easier to reproduce and inspect. Stochastic beams and learned values are follow-up baselines.

The fifth stage gates reinforcement-learning claims. If bounded oracle rollout shows headroom over one-step greedy and the VIN scorer ranks held-out candidates reliably, a masked discrete Q-function or offline value method may be added. Double Q-learning is relevant for controlling overestimation in max-based targets @DoubleDQN-vanHasselt2015, and IQL is relevant if learning is performed from fixed rollout data without evaluating unseen actions @IQL-kostrikov2021. PPO and SAC remain later references for simulator-backed continuous control rather than first thesis baselines, because they assume a reward loop and interaction regime that the current expensive oracle does not yet provide @PPO-schulman2017 @SAC-haarnoja2018. These methods will not be used to claim unrestricted continuous control unless the data support, reward speed, simulator access, and evaluation protocol are all sufficient.

Diagnostics are part of the method. A Rerun offline inspector will render semi-dense points, reference poses, candidate frusta, optional meshes, target boxes, and RRI-colored candidate layers from the immutable offline store. Candidate frusta should default to batched manual line strips until pose and camera regression tests prove that native camera logging is frame-safe. This inspector is intended to catch frame mistakes, invalid candidates, label pathologies, and missing fields before they contaminate training or rollout results.
