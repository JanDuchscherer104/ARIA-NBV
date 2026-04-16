# Pose Generation Ownership

`aria_nbv.pose_generation` owns candidate view sampling, orientation utilities,
feasibility rules, and counterfactual pose helpers.

It should expose explicit pose/frame semantics and keep visualization-only
corrections out of model, rendering, cache, and training contracts.
