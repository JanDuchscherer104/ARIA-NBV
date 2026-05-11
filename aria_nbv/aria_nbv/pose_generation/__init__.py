"""Finite-candidate generation and counterfactual rollout contracts.

The package owns the thesis action space: every NBV decision is a finite table
of candidate camera poses, validity masks, reason codes, and provenance fields.
`CandidateSamplingResult.views` is the compact valid table used for rendering
and scoring; `mask_valid`, `shell_poses`, strategy ids, mixture ids, and sampler
probabilities stay aligned to the full sampled shell so invalid actions remain
auditable instead of becoming low-RRI samples.

Target-conditioned rollout generation adds a runtime-only actor-visible target
context. `TARGET_POINT` candidates must use the selected observed/predicted
target center, not GT geometry. GT meshes and GT target crops are consumed only
by oracle scoring/evaluation.
"""

from .candidate_generation import CandidateViewGenerator, CandidateViewGeneratorConfig
from .candidate_mixture import (
    CandidateMixtureComponentConfig,
    CandidateMixtureViewGenerator,
    CandidateMixtureViewGeneratorConfig,
    candidate_strategy_id,
)
from .counterfactuals import (
    CounterfactualCandidateEvaluation,
    CounterfactualEvaluatorFn,
    CounterfactualOracleRriScorer,
    CounterfactualOracleRriScorerConfig,
    CounterfactualPoseGenerator,
    CounterfactualPoseGeneratorConfig,
    CounterfactualRolloutResult,
    CounterfactualSelectionPolicy,
    CounterfactualSelectionRecord,
    CounterfactualStepResult,
    CounterfactualTrajectory,
)
from .rollout_trace import (
    INVALID_REASON_CODES,
    INVALID_REASON_VERSION,
    RolloutLineage,
    RolloutStepTrace,
    RolloutTrace,
    build_synthetic_rollout_traces,
    read_rollout_traces,
    traces_from_rollout_result,
    write_rollout_traces,
)
from .target_counterfactuals import (
    SCENE_CROP_POLICY_SNIPPET_EXTENT_V1,
    TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1,
    CounterfactualTargetOracleRriScorer,
    CounterfactualTargetOracleRriScorerConfig,
    TargetRriInvalidError,
)
from .types import (
    CandidateGenerationRuntimeContext,
    CandidateSamplingResult,
    CollisionBackend,
    SamplingStrategy,
    ViewDirectionMode,
)
from .utils import (
    stats_to_markdown_table,
    summarise_dirs_ref,
    summarise_offsets_ref,
)

__all__ = [
    "CandidateViewGenerator",
    "CandidateViewGeneratorConfig",
    "CandidateMixtureComponentConfig",
    "CandidateMixtureViewGenerator",
    "CandidateMixtureViewGeneratorConfig",
    "CandidateGenerationRuntimeContext",
    "ViewDirectionMode",
    "candidate_strategy_id",
    "CounterfactualPoseGenerator",
    "CounterfactualPoseGeneratorConfig",
    "CounterfactualCandidateEvaluation",
    "CounterfactualEvaluatorFn",
    "CounterfactualOracleRriScorer",
    "CounterfactualOracleRriScorerConfig",
    "CounterfactualTargetOracleRriScorer",
    "CounterfactualTargetOracleRriScorerConfig",
    "SCENE_CROP_POLICY_SNIPPET_EXTENT_V1",
    "TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1",
    "TargetRriInvalidError",
    "CounterfactualRolloutResult",
    "CounterfactualSelectionRecord",
    "CounterfactualSelectionPolicy",
    "CounterfactualStepResult",
    "CounterfactualTrajectory",
    "RolloutLineage",
    "INVALID_REASON_CODES",
    "INVALID_REASON_VERSION",
    "RolloutStepTrace",
    "RolloutTrace",
    "build_synthetic_rollout_traces",
    "traces_from_rollout_result",
    "write_rollout_traces",
    "read_rollout_traces",
    "CandidateSamplingResult",
    "SamplingStrategy",
    "CollisionBackend",
    "summarise_offsets_ref",
    "summarise_dirs_ref",
    "stats_to_markdown_table",
]
