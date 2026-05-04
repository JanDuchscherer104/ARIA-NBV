"""Durable trace schema for bounded counterfactual rollouts.

The trace format is intentionally smaller than a full simulator state. It stores
candidate tables, selected actions, lightweight per-step metrics, cumulative
rollout metrics, and deterministic lineage. Heavy observations such as rendered
depths or per-candidate point clouds remain optional and should be materialized
only for selected actions or retained chains.
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
    "selected_shell_index",
    "selected_pose_world_cam",
)
"""Step fields that can be exposed to an actor or sequence model."""

ORACLE_ONLY_STEP_FIELDS = (
    "candidate_scores",
    "metric_vectors",
    "selected_metrics",
)
"""Step fields that should be treated as supervision/evaluation only."""

OPTIONAL_HEAVY_STEP_FIELDS = ("selected_point_cloud_world",)
"""Optional heavy diagnostics stored only for selected or retained actions."""


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

    candidate_scores: torch.Tensor | None = None
    """Full-shell score vector; invalid or unscored candidates are ``NaN``."""

    selection_score: float = 0.0
    """Score used to choose the selected action."""

    selection_score_label: str = "score"
    """Semantic label for :attr:`selection_score`, for example ``oracle_rri``."""

    metric_vectors: dict[str, torch.Tensor] = field(default_factory=dict)
    """Full-shell metric vectors aligned with :attr:`candidate_valid`."""

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

        return cls(
            step_index=int(step.step_index),
            selected_valid_index=int(step.selected_valid_index),
            selected_shell_index=int(step.selected_shell_index),
            selected_pose_world_cam=selected_pose,
            candidate_poses_world_cam=candidate_poses,
            candidate_valid=candidate_valid,
            candidate_scores=candidate_scores,
            selection_score=float(step.selection_score),
            selection_score_label=str(step.selection_score_label),
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
        """Reconstruct a trace from :meth:`to_serializable` output."""

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
    """Read rollout traces written by :func:`write_rollout_traces`."""

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
    """Build greedy and random-valid rollout traces on a synthetic box scene.

    This helper backs the trace smoke CLI. It is not a simulator; it only verifies
    that the current rollout generator and trace writer can produce replayable
    one-scene artifacts without ASE data.
    """

    traces: list[RolloutTrace] = []
    for policy in (
        CounterfactualSelectionPolicy.FARTHEST_FROM_HISTORY,
        CounterfactualSelectionPolicy.RANDOM,
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
                random_seed=seed,
                source_cache_version="synthetic",
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


def _full_candidate_vector(values: torch.Tensor, candidate_valid: torch.Tensor) -> torch.Tensor:
    valid_values = values.detach().cpu().reshape(-1)
    valid_mask = candidate_valid.detach().cpu().to(dtype=torch.bool).reshape(-1)
    valid_count = int(valid_mask.sum().item())
    if valid_values.numel() != valid_count:
        raise ValueError(f"Expected {valid_count} valid values, got {valid_values.numel()}.")
    fill_value = float("nan") if torch.is_floating_point(valid_values) else 0
    full = torch.full(
        valid_mask.shape,
        fill_value,
        dtype=valid_values.dtype,
        device=valid_values.device,
    )
    full[valid_mask] = valid_values
    return full


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
    return CounterfactualCandidateEvaluation(scores=scores, score_label="oracle_rri", metric_vectors={"rri": scores})


__all__ = [
    "ACTOR_VISIBLE_STEP_FIELDS",
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
