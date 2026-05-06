#import "../../../shared/macros.typ": *
#import "_style.typ": *

= Problem Statement

The thesis studies view selection for an egocentric reconstruction episode in which the agent has already observed a partial trajectory and must choose future camera views from a finite candidate set. At step $t$, the available state consists of the current reference pose, calibrated camera geometry, accumulated semi-dense points, candidate poses, candidate validity information, and frozen egocentric features derived from the logged snippet. The privileged ground-truth mesh is used only to compute oracle labels and evaluation metrics. It is not an actor-visible input for the first planning formulation.

The core quality signal is #RRI. Given the current point set, a candidate view, and the ground-truth mesh, the oracle renders candidate geometry, fuses it with the current reconstruction state, and measures the relative reduction in reconstruction error. In the scene-level case this score evaluates improvement over the full mesh, while in the target-aware case it evaluates the same improvement over an object or region crop. The thesis keeps these two quantities visible rather than hiding the tradeoff inside a single opaque reward.

#block[#align(center)[#eqs.rri.rri]]

The central research problem is to learn and evaluate a target-conditioned scorer that ranks candidate views by target-specific reconstruction improvement, then train a finite-candidate fitted $Q_H$ selector from trusted rollout data. The one-step baseline selects the candidate with the best immediate oracle or predicted #RRI. The non-myopic baseline restricts the search tree to a small set of high-scoring valid candidates and recursively evaluates the first action under a finite horizon. This produces the experimental sequence $ "ArgTopK" -> "ArgTop1"_1 -> "ArgTop1"_2 -> ... -> "ArgTop1"_H $, where $"ArgTop1"_1$ is the greedy selector and $"ArgTop1"_H$ is the first action chosen by an $H$-step bounded rollout. The learned $Q_H$ model is then evaluated against one-step greedy and model scoring under the same acquisition budget.

#rollout-ladder()

The proposal deliberately narrows the problem to discrete candidate sets. Continuous policies such as GenNBV demonstrate that direct 5-DoF reinforcement learning is possible when a simulator, reward, and training distribution are available @GenNBV-chen2024. Hestia further shows that hierarchical action factorization and close-greedy reward design can improve continuous #NBV control @Hestia-lu2026. In ARIA-NBV, however, oracle #RRI evaluation is expensive, counterfactual modalities are incomplete, and the local offline store must first be made inspectable. The thesis therefore treats finite-candidate fitted Double-Q / $Q_H$ as the required value-learning result and continuous actor-critic learning as a follow-up direction rather than as the first claim.

The main failure mode the thesis must avoid is a visually plausible but geometrically inconsistent pipeline. Candidate frusta, camera frames, display rotations, invalid masks, and #RRI labels can all look reasonable while encoding the wrong transformation or supervision contract. For this reason, the proposal treats offline-store validation and Rerun-based inspection as part of the scientific method rather than as optional tooling. If a sample lacks required pose, camera, point, or metric fields, it should fail before rendering or training; if optional fields are absent, the diagnostics should state that limitation explicitly.
