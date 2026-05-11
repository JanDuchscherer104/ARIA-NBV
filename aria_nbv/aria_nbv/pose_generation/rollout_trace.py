"""Durable trace schema for bounded counterfactual rollouts.

The trace format is intentionally smaller than a full simulator state. It stores
candidate tables, selected actions, lightweight per-step metrics, cumulative
rollout metrics, and deterministic lineage. Heavy observations such as rendered
depths or per-candidate point clouds remain optional and should be materialized
only for selected actions or retained chains.

Actor-visible and oracle-only fields are named explicitly so `rollouts.zarr`
can derive Q-training masks without leaking labels into observations. For
bounded value learning, per-step target RRI contributes to the training return,
while endpoint target-quality gain is measured by oracle re-scoring the selected
trajectory under the same budget.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
import torch
import trimesh  # type: ignore[import-untyped]
from efm3d.aria import CameraTW, PoseTW

from ..utils import BaseConfig, Console
from ..utils.typed_payloads import from_serializable, to_serializable
from .candidate_generation import CandidateViewGeneratorConfig
from .counterfactuals import (
    CounterfactualCandidateEvaluation,
    CounterfactualPoseGenerator,
    CounterfactualPoseGeneratorConfig,
    CounterfactualRolloutResult,
    CounterfactualSelectionPolicy,
    CounterfactualStepResult,
    CounterfactualTrajectory,
)
from .types import SamplingStrategy

ACTOR_VISIBLE_STEP_FIELDS = (
    "candidate_poses_world_cam",
    "candidate_valid",
    "candidate_invalid_reason_bitset",
    "candidate_primary_invalid_reason",
    "selected_shell_index",
    "selected_pose_world_cam",
)
"""Step fields that can be exposed to an actor or sequence model."""

ORACLE_ONLY_STEP_FIELDS = (
    "candidate_scores",
    "metric_vectors",
    "selected_metrics",
    "selection_logits",
    "selection_probabilities",
    "selection_log_probabilities",
    "selection_entropy",
    "selected_log_probability",
)
"""Step fields that should be treated as supervision/evaluation only."""

OPTIONAL_HEAVY_STEP_FIELDS = ("selected_point_cloud_world",)
"""Optional heavy diagnostics stored only for selected or retained actions."""

INVALID_REASON_CODES: dict[str, int] = {
    "VALID": 0,
    "POSE_NONFINITE": 1,
    "POSE_OUT_OF_EXTENT": 2,
    "CAMERA_OUT_OF_EXTENT": 3,
    "COLLISION_MESH": 4,
    "CLEARANCE_TOO_SMALL": 5,
    "PATH_SEGMENT_COLLISION": 6,
    "FRUSTUM_OUT_OF_BOUNDS": 7,
    "DEPTH_NO_HIT": 8,
    "DEPTH_TOO_SPARSE": 9,
    "BACKPROJECT_EMPTY": 10,
    "CANDIDATE_DUPLICATE": 11,
    "SAMPLER_RULE_REJECTED": 12,
    "TARGET_NOT_ACTOR_VISIBLE": 13,
    "TARGET_GT_UNMATCHED": 14,
    "TARGET_CROP_EMPTY": 15,
    "TARGET_SUPPORT_TOO_LOW": 16,
    "TARGET_VISIBILITY_TOO_LOW": 17,
    "SEMIDENSE_SUPPORT_TOO_LOW": 18,
    "EVL_EVIDENCE_MISSING": 19,
    "MESH_REFERENCE_MISSING": 20,
    "ORACLE_DISTANCE_FAILED": 21,
    "CANDIDATE_ORDER_GUARD_FAILED": 22,
    "RUNTIME_ERROR": 23,
}
"""Version-1 invalidity reason bit positions for rollout replay tables."""

INVALID_REASON_VERSION = "rollout-invalidity-v1"
"""Version label for `INVALID_REASON_CODES`."""

_RULE_REASON_BITS = {
    "FreeSpaceRule": INVALID_REASON_CODES["POSE_OUT_OF_EXTENT"],
    "MinDistanceToMeshRule": INVALID_REASON_CODES["CLEARANCE_TOO_SMALL"],
    "PathCollisionRule": INVALID_REASON_CODES["PATH_SEGMENT_COLLISION"],
}


@dataclass(slots=True)
class RolloutLineage:
    """Deterministic provenance for one rollout chain."""

    rollout_id: str
    """Stable rollout identifier inside a generated dataset."""

    chain_id: int = 0
    """Zero-based trajectory index when one rollout call returns a beam."""

    scene_id: str | None = None
    """Scene identifier, if the trace was generated from an ASE snippet."""

    snippet_id: str | None = None
    """Snippet identifier, if the trace was generated from an ASE snippet."""

    mesh_version: str | None = None
    """GT mesh version or digest used by oracle generation."""

    candidate_config_hash: str | None = None
    """Stable hash of candidate-generation settings."""

    oracle_config_hash: str | None = None
    """Stable hash of oracle/scorer settings when oracle labels are present."""

    model_checkpoint_hash: str | None = None
    """Stable hash of a model checkpoint when using learned scores."""

    random_seed: int | None = None
    """Seed controlling candidate or selection stochasticity."""

    rollout_policy: str = "unknown"
    """Policy used to select actions, for example ``farthest_from_history``."""

    source_cache_version: str | None = None
    """Source offline-cache or dataset version used to build the trace."""

    split: str | None = None
    """Dataset split inherited from the source snippet or rollout manifest."""

    source_offline_store_manifest_hash: str | None = None
    """Hash of the source offline-store manifest, when the rollout came from one."""

    split_manifest_hash: str | None = None
    """Hash of the scene/snippet split manifest used for this rollout."""

    rollout_config_hash: str | None = None
    """Stable hash of rollout-generation settings."""

    branch_schedule_id: str | None = None
    """Identifier for the branch schedule used by stochastic or beam rollout generation."""

    target_row_id: int | None = None
    """Optional row id of the actor-visible target record used by this rollout."""

    target_id: str | None = None
    """Stable source target id when available."""

    target_protocol_version: str | None = None
    """Target protocol used for actor input and GT evaluation."""

    target_crop_policy: str | None = None
    """Oracle/evaluation crop policy used for target-specific RRI labels."""

    reason_code_version: str = INVALID_REASON_VERSION
    """Version of candidate invalidity reason codes used by this trace."""

    selection_rng_state_hash: str | None = None
    """Optional digest of the selection RNG state after generation."""

    target_selection_policy: str | None = None
    """Actor-visible target selector policy used before this rollout, if known."""

    target_selection_rank: int | None = None
    """Zero-based selected target rank inside the selector's top-K table."""

    target_selection_score: float | None = None
    """Final actor-visible target selector score for the rollout target."""

    target_selection_probability: float | None = None
    """Selection probability for stochastic target policies, if applicable."""

    target_selection_temperature: float | None = None
    """Temperature used by stochastic target selection, if applicable."""

    target_invalid_reason_bitset: int | None = None
    """Target-selector invalidity bitset for the selected target row."""

    target_primary_invalid_reason: int | None = None
    """Dominant target-selector invalidity reason for the selected target row."""

    target_reason_code_version: str | None = None
    """Version of the target-selector invalidity reason-code dictionary."""

    matched_gt_target_row_id: int | None = None
    """Matched GT target row id used for oracle/evaluation labels, if any."""

    matched_gt_target_id: str | None = None
    """Matched GT target identifier used for oracle/evaluation labels, if any."""

    gt_match_iou: float | None = None
    """Sampled 3D IoU of the selected actor-visible target to the matched GT target."""

    gt_match_score: float | None = None
    """Selector-recorded GT match score, when distinct from IoU."""

    gt_match_status: str | None = None
    """GT match status such as ``matched``, ``unmatched_gt``, or ``ambiguous_gt``."""


@dataclass(slots=True)
class RolloutStepTrace:
    """Serializable candidate table and selected action for one horizon step.

    Actor-visible fields are poses, validity, and the selected action. Scores
    and metric vectors are oracle/model supervision and must not be treated as
    actor observations unless an experiment explicitly promotes them.
    """

    step_index: int
    """Zero-based horizon step."""

    selected_valid_index: int
    """Index inside the compact valid-candidate table."""

    selected_shell_index: int
    """Index inside the full sampled candidate shell before pruning."""

    selected_pose_world_cam: torch.Tensor
    """Selected world←camera pose tensor with shape ``(12,)``."""

    candidate_poses_world_cam: torch.Tensor
    """Full sampled candidate shell as world←camera poses with shape ``(N, 12)``."""

    candidate_valid: torch.Tensor
    """Boolean full-shell validity mask with shape ``(N,)``."""

    candidate_invalid_reason_bitset: torch.Tensor | None = None
    """Full-shell invalidity bitsets with shape ``(N,)`` and dtype ``uint32``."""

    candidate_primary_invalid_reason: torch.Tensor | None = None
    """Dominant invalidity reason bit with shape ``(N,)`` and dtype ``uint16``."""

    candidate_strategy_id: torch.Tensor | None = None
    """Full-shell candidate strategy ids with shape ``(N,)``."""

    candidate_mixture_id: torch.Tensor | None = None
    """Full-shell candidate mixture component ids with shape ``(N,)``."""

    candidate_sampler_probability: torch.Tensor | None = None
    """Full-shell sampler probabilities with shape ``(N,)``."""

    candidate_scores: torch.Tensor | None = None
    """Full-shell score vector; invalid or unscored candidates are ``NaN``."""

    selection_score: float = 0.0
    """Score used to choose the selected action."""

    selection_score_label: str = "score"
    """Semantic label for `selection_score`, for example ``oracle_rri``."""

    selection_policy: str = "unknown"
    """Selection policy used at this step."""

    score_source: str = "score"
    """Source of selection scores, for example ``oracle_rri`` or ``model_score``."""

    selection_temperature: float | None = None
    """Temperature used for softmax selection, if applicable."""

    selection_logits: torch.Tensor | None = None
    """Full-shell selection logits; invalid candidates are ``NaN``."""

    selection_probabilities: torch.Tensor | None = None
    """Full-shell selection probabilities; invalid candidates are exactly zero."""

    selection_log_probabilities: torch.Tensor | None = None
    """Full-shell log probabilities for the selection draw."""

    selection_entropy: float | None = None
    """Entropy of the valid-candidate selection distribution."""

    selected_log_probability: float | None = None
    """Log probability assigned to the selected action."""

    selection_rng_seed: int | None = None
    """Seed used for stochastic rollout selection, when configured."""

    transition_id: str | None = None
    """Stable selected-action transition id inside one rollout trace."""

    metric_vectors: dict[str, torch.Tensor] = field(default_factory=dict)
    """Full-shell metric vectors aligned with `candidate_valid`."""

    selected_metrics: dict[str, float] = field(default_factory=dict)
    """Metrics for the selected action only."""

    cumulative_score: float = 0.0
    """Running sum of selection scores through this step."""

    cumulative_rri: float | None = None
    """Running sum of selected RRI values when an RRI metric is present."""

    selected_point_cloud_world: torch.Tensor | None = None
    """Optional selected-action point cloud in world coordinates."""

    @classmethod
    def from_step(cls, step: CounterfactualStepResult) -> "RolloutStepTrace":
        """Build a serializable trace step from an in-memory rollout step."""

        candidate_valid = step.candidates.mask_valid.detach().cpu().to(dtype=torch.bool).reshape(-1)
        candidate_poses = step.candidates.shell_poses.tensor().detach().cpu()
        selected_pose = step.selected_pose_world.tensor().detach().cpu().reshape(-1)

        candidate_scores = None
        if step.selection_scores is not None:
            candidate_scores = _full_candidate_vector(step.selection_scores, candidate_valid)

        selection_logits = None
        if step.selection_logits is not None:
            selection_logits = _full_candidate_vector(step.selection_logits, candidate_valid)
        selection_probabilities = None
        if step.selection_probabilities is not None:
            selection_probabilities = _full_candidate_vector(
                step.selection_probabilities,
                candidate_valid,
                fill_value=0.0,
            )
        selection_log_probabilities = None
        if step.selection_log_probabilities is not None:
            selection_log_probabilities = _full_candidate_vector(
                step.selection_log_probabilities,
                candidate_valid,
                fill_value=float("-inf"),
            )
        reason_bitset, primary_reason = _candidate_invalid_reasons(step.candidates)

        return cls(
            step_index=int(step.step_index),
            selected_valid_index=int(step.selected_valid_index),
            selected_shell_index=int(step.selected_shell_index),
            selected_pose_world_cam=selected_pose,
            candidate_poses_world_cam=candidate_poses,
            candidate_valid=candidate_valid,
            candidate_invalid_reason_bitset=reason_bitset,
            candidate_primary_invalid_reason=primary_reason,
            candidate_strategy_id=_full_shell_or_default(
                step.candidates.strategy_id,
                candidate_valid,
                fill_value=-1,
            ),
            candidate_mixture_id=_full_shell_or_default(
                step.candidates.mixture_id,
                candidate_valid,
                fill_value=-1,
            ),
            candidate_sampler_probability=_full_shell_or_default(
                step.candidates.sampler_probability,
                candidate_valid,
                fill_value=float("nan"),
            ),
            candidate_scores=candidate_scores,
            selection_score=float(step.selection_score),
            selection_score_label=str(step.selection_score_label),
            selection_policy=str(step.selection_policy),
            score_source=str(step.selection_score_label),
            selection_temperature=step.selection_temperature,
            selection_logits=selection_logits,
            selection_probabilities=selection_probabilities,
            selection_log_probabilities=selection_log_probabilities,
            selection_entropy=step.selection_entropy,
            selected_log_probability=step.selected_log_probability,
            selection_rng_seed=step.selection_rng_seed,
            metric_vectors={
                name: _full_candidate_vector(values, candidate_valid) for name, values in step.metric_vectors.items()
            },
            selected_metrics=dict(step.selected_metrics),
            cumulative_score=0.0,
            cumulative_rri=None,
            selected_point_cloud_world=(
                None if step.selected_point_cloud_world is None else step.selected_point_cloud_world.detach().cpu()
            ),
        )


@dataclass(slots=True)
class RolloutTrace:
    """Durable replay metadata for one bounded rollout trajectory."""

    lineage: RolloutLineage
    """Deterministic provenance and dataset identifiers."""

    root_pose_world: torch.Tensor
    """Root world pose tensor with shape ``(12,)``."""

    horizon: int
    """Configured rollout horizon."""

    branch_factor: int
    """Configured branch factor."""

    beam_width: int | None
    """Configured retained beam width."""

    selection_policy: str
    """Selection policy used by the generator."""

    score_label: str
    """Score label reported by the evaluator."""

    termination_reason: str
    """Why this trajectory stopped: ``fixed_horizon``, ``terminated_early``, or ``incomplete_rollout``."""

    steps: list[RolloutStepTrace] = field(default_factory=list)
    """Per-step candidate tables and selected actions."""

    final_cumulative_score: float = 0.0
    """Final cumulative selection score."""

    final_cumulative_rri: float | None = None
    """Final cumulative RRI when available."""

    def to_serializable(self) -> dict[str, Any]:
        """Convert this trace into a msgspec-compatible payload."""

        return to_serializable(self)

    @classmethod
    def from_serializable(cls, payload: dict[str, Any], *, device: torch.device | None = None) -> "RolloutTrace":
        """Reconstruct a trace from `to_serializable` output."""

        return from_serializable(cls, payload, device=device)

    @classmethod
    def from_trajectory(
        cls,
        *,
        result: CounterfactualRolloutResult,
        trajectory: CounterfactualTrajectory,
        lineage: RolloutLineage,
    ) -> "RolloutTrace":
        """Build one trace from a rollout result and retained trajectory."""

        running_score = 0.0
        running_rri: float | None = None
        steps: list[RolloutStepTrace] = []
        for step in trajectory.steps:
            running_score += float(step.selection_score)
            step_rri = step.selected_metrics.get("rri")
            if step_rri is not None:
                running_rri = float(step_rri) if running_rri is None else float(running_rri + step_rri)
            trace_step = RolloutStepTrace.from_step(step)
            trace_step.cumulative_score = float(running_score)
            trace_step.cumulative_rri = running_rri
            trace_step.transition_id = (
                f"{lineage.rollout_id}:step={trace_step.step_index}:shell={trace_step.selected_shell_index}"
            )
            steps.append(trace_step)

        return cls(
            lineage=lineage,
            root_pose_world=result.root_pose_world.tensor().detach().cpu().reshape(-1),
            horizon=int(result.horizon),
            branch_factor=int(result.branch_factor),
            beam_width=result.beam_width,
            selection_policy=_policy_name(result.selection_policy),
            score_label=str(result.score_label),
            termination_reason=_termination_reason(result, trajectory),
            steps=steps,
            final_cumulative_score=float(trajectory.cumulative_score),
            final_cumulative_rri=trajectory.cumulative_rri,
        )


def traces_from_rollout_result(
    result: CounterfactualRolloutResult,
    *,
    rollout_id_prefix: str,
    scene_id: str | None = None,
    snippet_id: str | None = None,
    mesh_version: str | None = None,
    candidate_config_hash: str | None = None,
    oracle_config_hash: str | None = None,
    model_checkpoint_hash: str | None = None,
    random_seed: int | None = None,
    source_cache_version: str | None = None,
    split: str | None = None,
    source_offline_store_manifest_hash: str | None = None,
    split_manifest_hash: str | None = None,
    rollout_config_hash: str | None = None,
    branch_schedule_id: str | None = None,
    target_row_id: int | None = None,
    target_id: str | None = None,
    target_protocol_version: str | None = None,
    target_crop_policy: str | None = None,
    reason_code_version: str = INVALID_REASON_VERSION,
    selection_rng_state_hash: str | None = None,
    target_selection_policy: str | None = None,
    target_selection_rank: int | None = None,
    target_selection_score: float | None = None,
    target_selection_probability: float | None = None,
    target_selection_temperature: float | None = None,
    target_invalid_reason_bitset: int | None = None,
    target_primary_invalid_reason: int | None = None,
    target_reason_code_version: str | None = None,
    matched_gt_target_row_id: int | None = None,
    matched_gt_target_id: str | None = None,
    gt_match_iou: float | None = None,
    gt_match_score: float | None = None,
    gt_match_status: str | None = None,
) -> list[RolloutTrace]:
    """Convert all retained trajectories from one rollout call into traces."""

    traces: list[RolloutTrace] = []
    for chain_id, trajectory in enumerate(result.trajectories):
        lineage = RolloutLineage(
            rollout_id=f"{rollout_id_prefix}-{chain_id:06d}",
            chain_id=chain_id,
            scene_id=scene_id,
            snippet_id=snippet_id,
            mesh_version=mesh_version,
            candidate_config_hash=candidate_config_hash,
            oracle_config_hash=oracle_config_hash,
            model_checkpoint_hash=model_checkpoint_hash,
            random_seed=random_seed,
            rollout_policy=_policy_name(result.selection_policy),
            source_cache_version=source_cache_version,
            split=split,
            source_offline_store_manifest_hash=source_offline_store_manifest_hash,
            split_manifest_hash=split_manifest_hash,
            rollout_config_hash=rollout_config_hash,
            branch_schedule_id=branch_schedule_id,
            target_row_id=target_row_id,
            target_id=target_id,
            target_protocol_version=target_protocol_version,
            target_crop_policy=target_crop_policy,
            reason_code_version=reason_code_version,
            selection_rng_state_hash=selection_rng_state_hash,
            target_selection_policy=target_selection_policy,
            target_selection_rank=target_selection_rank,
            target_selection_score=target_selection_score,
            target_selection_probability=target_selection_probability,
            target_selection_temperature=target_selection_temperature,
            target_invalid_reason_bitset=target_invalid_reason_bitset,
            target_primary_invalid_reason=target_primary_invalid_reason,
            target_reason_code_version=target_reason_code_version,
            matched_gt_target_row_id=matched_gt_target_row_id,
            matched_gt_target_id=matched_gt_target_id,
            gt_match_iou=gt_match_iou,
            gt_match_score=gt_match_score,
            gt_match_status=gt_match_status,
        )
        traces.append(RolloutTrace.from_trajectory(result=result, trajectory=trajectory, lineage=lineage))
    return traces


def write_rollout_traces(path: Path | str, traces: list[RolloutTrace]) -> Path:
    """Write rollout traces as one MessagePack payload.

    Args:
        path: Destination ``.msgpack`` file.
        traces: Traces to persist.

    Returns:
        Resolved output path.
    """

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [trace.to_serializable() for trace in traces]
    output_path.write_bytes(msgspec.msgpack.encode(payload))
    return output_path


def read_rollout_traces(path: Path | str, *, device: torch.device | None = None) -> list[RolloutTrace]:
    """Read rollout traces written by `write_rollout_traces`."""

    payload = msgspec.msgpack.decode(Path(path).expanduser().resolve().read_bytes())
    if not isinstance(payload, list):
        raise TypeError("Rollout trace payload must decode to a list.")
    return [RolloutTrace.from_serializable(item, device=device) for item in payload]


def build_synthetic_rollout_traces(
    *,
    horizon: int = 2,
    num_samples: int = 8,
    seed: int = 0,
) -> list[RolloutTrace]:
    """Build greedy, random-valid, and temperature-softmax traces on a synthetic box scene.

    This helper backs the trace smoke CLI. It is not a simulator; it only verifies
    that the current rollout generator and trace writer can produce replayable
    one-scene artifacts without ASE data.
    """

    traces: list[RolloutTrace] = []
    for policy in (
        CounterfactualSelectionPolicy.ORACLE_GREEDY,
        CounterfactualSelectionPolicy.RANDOM_VALID,
        CounterfactualSelectionPolicy.TEMPERATURE_SOFTMAX,
    ):
        cfg = CounterfactualPoseGeneratorConfig(
            candidate_config=CandidateViewGeneratorConfig(
                num_samples=num_samples,
                min_radius=0.5,
                max_radius=0.5,
                ensure_collision_free=False,
                ensure_free_space=False,
                min_distance_to_mesh=0.0,
                sampling_strategy=SamplingStrategy.UNIFORM_SPHERE,
                view_max_azimuth_deg=0.0,
                view_max_elevation_deg=0.0,
                device="cpu",
                seed=seed,
                verbosity=0,
                is_debug=True,
            ),
            horizon=horizon,
            branch_factor=1,
            selection_policy=policy,
            selection_temperature=1.0,
            seed=seed,
            verbosity=0,
        )
        generator = CounterfactualPoseGenerator(cfg)
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        result = generator.generate(
            reference_pose=_identity_pose(),
            gt_mesh=mesh,
            mesh_verts=torch.as_tensor(mesh.vertices, dtype=torch.float32),
            mesh_faces=torch.as_tensor(mesh.faces, dtype=torch.int64),
            camera_calib_template=_dummy_camera(),
            occupancy_extent=torch.tensor([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=torch.float32),
            score_candidates=_smoke_scores,
        )
        traces.extend(
            traces_from_rollout_result(
                result,
                rollout_id_prefix=f"synthetic-{policy.value}",
                scene_id="synthetic_box",
                snippet_id="smoke",
                candidate_config_hash=_config_hash(cfg.candidate_config),
                oracle_config_hash="synthetic-smoke",
                rollout_config_hash=_config_hash(cfg),
                branch_schedule_id=cfg.branch_schedule_id,
                random_seed=seed,
                source_cache_version="synthetic",
                split="synthetic",
                source_offline_store_manifest_hash="synthetic",
                split_manifest_hash="synthetic",
                selection_rng_state_hash="synthetic",
                target_protocol_version="synthetic",
            )
        )
    return traces


def smoke_main(argv: list[str] | None = None) -> None:
    """Write a synthetic one-scene rollout trace for smoke verification."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(".artifacts") / "rollouts" / "synthetic_rollout_traces.msgpack",
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    traces = build_synthetic_rollout_traces(horizon=args.horizon, num_samples=args.num_samples, seed=args.seed)
    output_path = write_rollout_traces(args.output_path, traces)
    Console.with_prefix("rollout-trace-smoke").log(f"Wrote {len(traces)} traces to {output_path}")


def _full_candidate_vector(
    values: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    fill_value: float | int | None = None,
    require_full_shell: bool = False,
) -> torch.Tensor:
    valid_values = values.detach().cpu().reshape(-1)
    valid_mask = candidate_valid.detach().cpu().to(dtype=torch.bool).reshape(-1)
    if require_full_shell:
        if valid_values.numel() != valid_mask.numel():
            raise ValueError(f"Expected {valid_mask.numel()} full-shell values, got {valid_values.numel()}.")
        return valid_values
    valid_count = int(valid_mask.sum().item())
    if valid_values.numel() != valid_count:
        raise ValueError(f"Expected {valid_count} valid values, got {valid_values.numel()}.")
    if fill_value is None:
        fill_value = float("nan") if torch.is_floating_point(valid_values) else 0
    full = torch.full(
        valid_mask.shape,
        fill_value,
        dtype=valid_values.dtype,
        device=valid_values.device,
    )
    full[valid_mask] = valid_values
    return full


def _full_shell_or_default(
    values: torch.Tensor | None,
    candidate_valid: torch.Tensor,
    *,
    fill_value: float | int,
) -> torch.Tensor:
    valid_mask = candidate_valid.detach().cpu().to(dtype=torch.bool).reshape(-1)
    if values is None:
        dtype = torch.float32 if isinstance(fill_value, float) else torch.int64
        return torch.full(valid_mask.shape, fill_value, dtype=dtype)
    return _full_candidate_vector(values, candidate_valid, fill_value=fill_value, require_full_shell=True)


def _candidate_invalid_reasons(candidates: Any) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = candidates.mask_valid.detach().cpu().to(dtype=torch.bool).reshape(-1)
    bitset = torch.zeros(valid_mask.shape, dtype=torch.int64)
    primary = torch.full(valid_mask.shape, INVALID_REASON_CODES["SAMPLER_RULE_REJECTED"], dtype=torch.int64)
    bitset[valid_mask] = 1 << INVALID_REASON_CODES["VALID"]
    primary[valid_mask] = INVALID_REASON_CODES["VALID"]

    previous = torch.ones_like(valid_mask)
    for rule_name, cumulative_mask in candidates.masks.items():
        current = cumulative_mask.detach().cpu().to(dtype=torch.bool).reshape(-1)
        if current.shape != valid_mask.shape:
            continue
        failed_here = previous & (~current)
        reason_bit = _RULE_REASON_BITS.get(rule_name, INVALID_REASON_CODES["SAMPLER_RULE_REJECTED"])
        bitset[failed_here] = bitset[failed_here] | (1 << reason_bit)
        primary[failed_here] = reason_bit
        previous = current

    unresolved_invalid = (~valid_mask) & (bitset == 0)
    bitset[unresolved_invalid] = 1 << INVALID_REASON_CODES["SAMPLER_RULE_REJECTED"]

    shell = candidates.shell_poses.tensor().detach().cpu()
    nonfinite = ~torch.isfinite(shell.reshape(shell.shape[0], -1)).all(dim=1)
    bitset[nonfinite] = bitset[nonfinite] | (1 << INVALID_REASON_CODES["POSE_NONFINITE"])
    primary[nonfinite] = INVALID_REASON_CODES["POSE_NONFINITE"]

    return bitset.to(dtype=torch.int64), primary.to(dtype=torch.int64)


def _termination_reason(result: CounterfactualRolloutResult, trajectory: CounterfactualTrajectory) -> str:
    if trajectory.terminated_early:
        return "terminated_early"
    if len(trajectory.steps) >= int(result.horizon):
        return "fixed_horizon"
    return "incomplete_rollout"


def _policy_name(policy: str | CounterfactualSelectionPolicy) -> str:
    return policy.value if isinstance(policy, CounterfactualSelectionPolicy) else str(policy)


def _config_hash(config: BaseConfig) -> str:
    payload = repr(config.model_dump_jsonable()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _identity_pose() -> PoseTW:
    return PoseTW(
        torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
    )


def _dummy_camera() -> CameraTW:
    return CameraTW.from_surreal(
        width=torch.tensor([64.0]),
        height=torch.tensor([64.0]),
        type_str="Pinhole",
        params=torch.tensor([[60.0, 60.0, 32.0, 32.0]]),
        gain=torch.zeros(1),
        exposure_s=torch.zeros(1),
        valid_radius=torch.tensor([64.0]),
        T_camera_rig=PoseTW.from_matrix3x4(torch.eye(3, 4).unsqueeze(0)),
    )


def _smoke_scores(
    result: Any,
    trajectory: CounterfactualTrajectory,
    step_index: int,
) -> CounterfactualCandidateEvaluation:
    del trajectory
    valid_poses = result.poses_world_cam()
    centers = valid_poses.t.reshape(-1, 3)
    scores = torch.linspace(0.1, 0.1 * centers.shape[0], centers.shape[0], device=centers.device)
    scores = scores + float(step_index)
    return CounterfactualCandidateEvaluation(
        scores=scores,
        score_label="oracle_rri",
        metric_vectors={"rri": scores, "scene_rri": scores, "target_rri": scores},
    )


__all__ = [
    "ACTOR_VISIBLE_STEP_FIELDS",
    "INVALID_REASON_CODES",
    "INVALID_REASON_VERSION",
    "OPTIONAL_HEAVY_STEP_FIELDS",
    "ORACLE_ONLY_STEP_FIELDS",
    "RolloutLineage",
    "RolloutStepTrace",
    "RolloutTrace",
    "build_synthetic_rollout_traces",
    "read_rollout_traces",
    "smoke_main",
    "traces_from_rollout_result",
    "write_rollout_traces",
]
