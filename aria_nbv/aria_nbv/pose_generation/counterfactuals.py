"""Multi-step counterfactual pose rollout utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from pydantic import Field, field_validator

from ..data_handling import EfmSnippetView
from ..rendering.candidate_depth_renderer import CandidateDepthRendererConfig
from ..rendering.candidate_pointclouds import build_candidate_pointclouds
from ..rri_metrics.oracle_rri import OracleRRIConfig
from ..utils import BaseConfig, Console, Verbosity
from ..utils.frames import rotate_yaw_cw90
from .candidate_generation import CandidateViewGenerator, CandidateViewGeneratorConfig
from .types import CandidateSamplingResult
from .utils import ensure_unbatched_pose

if TYPE_CHECKING:
    import trimesh


def _pose_row(pose: PoseTW) -> torch.Tensor:
    return ensure_unbatched_pose(pose).tensor().reshape(1, -1)


def _pose_batch_len(poses: PoseTW) -> int:
    tensor = poses.tensor()
    return 1 if tensor.ndim == 1 else int(tensor.shape[0])


def _pose_at(poses: PoseTW, index: int) -> PoseTW:
    if _pose_batch_len(poses) == 1:
        if index != 0:
            raise IndexError(f"Pose batch has length 1, cannot index {index}.")
        return ensure_unbatched_pose(poses)
    return ensure_unbatched_pose(poses[index])


class CounterfactualSelectionPolicy(StrEnum):
    """Built-in policies used to rank valid candidates during rollout expansion."""

    FARTHEST_FROM_HISTORY = "farthest_from_history"
    FARTHEST_FROM_REFERENCE = "farthest_from_reference"
    RANDOM = "random"


@dataclass(slots=True)
class CounterfactualCandidateEvaluation:
    """Structured per-valid-candidate rollout scores and optional diagnostics."""

    scores: torch.Tensor
    score_label: str = "score"
    metric_vectors: dict[str, torch.Tensor] = field(default_factory=dict)
    candidate_point_clouds_world: torch.Tensor | None = None
    candidate_point_cloud_lengths: torch.Tensor | None = None

    def validate(
        self,
        *,
        num_valid: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "CounterfactualCandidateEvaluation":
        """Normalize score vectors and ensure they align with valid candidates."""

        scores = torch.as_tensor(self.scores, device=device, dtype=dtype).reshape(-1)
        if scores.shape[0] != num_valid:
            raise ValueError(f"Counterfactual evaluator must return {num_valid} scores, got {scores.shape[0]}.")

        metric_vectors: dict[str, torch.Tensor] = {}
        for name, values in self.metric_vectors.items():
            metric = torch.as_tensor(values, device=device, dtype=dtype).reshape(-1)
            if metric.shape[0] != num_valid:
                raise ValueError(
                    f"Counterfactual evaluator metric '{name}' must return {num_valid} values, got {metric.shape[0]}.",
                )
            metric_vectors[name] = metric

        candidate_point_clouds_world = self.candidate_point_clouds_world
        candidate_point_cloud_lengths = self.candidate_point_cloud_lengths
        if candidate_point_clouds_world is not None:
            candidate_point_clouds_world = torch.as_tensor(candidate_point_clouds_world, device=device, dtype=dtype)
            if candidate_point_clouds_world.ndim != 3 or candidate_point_clouds_world.shape[0] != num_valid:
                raise ValueError(
                    "Counterfactual evaluator candidate_point_clouds_world must have shape (num_valid, P, 3).",
                )
            if candidate_point_cloud_lengths is None:
                candidate_point_cloud_lengths = torch.full(
                    (num_valid,),
                    candidate_point_clouds_world.shape[1],
                    dtype=torch.long,
                    device=device,
                )
            else:
                candidate_point_cloud_lengths = torch.as_tensor(
                    candidate_point_cloud_lengths,
                    device=device,
                    dtype=torch.long,
                ).reshape(-1)
                if candidate_point_cloud_lengths.shape[0] != num_valid:
                    raise ValueError(
                        "Counterfactual evaluator candidate_point_cloud_lengths must align with num_valid.",
                    )
        elif candidate_point_cloud_lengths is not None:
            raise ValueError("candidate_point_cloud_lengths requires candidate_point_clouds_world.")

        return CounterfactualCandidateEvaluation(
            scores=scores,
            score_label=self.score_label,
            metric_vectors=metric_vectors,
            candidate_point_clouds_world=candidate_point_clouds_world,
            candidate_point_cloud_lengths=candidate_point_cloud_lengths,
        )

    def selected_metrics(self, valid_index: int) -> dict[str, float]:
        return {name: float(values[valid_index].item()) for name, values in self.metric_vectors.items()}

    def selected_point_cloud(self, valid_index: int) -> torch.Tensor | None:
        if self.candidate_point_clouds_world is None:
            return None
        cloud = self.candidate_point_clouds_world[valid_index]
        if self.candidate_point_cloud_lengths is None:
            return cloud
        length = int(self.candidate_point_cloud_lengths[valid_index].item())
        return cloud[:length]


@dataclass(slots=True)
class CounterfactualStepResult:
    """One selected rollout step."""

    step_index: int
    candidates: CandidateSamplingResult
    selected_valid_index: int
    selected_shell_index: int
    selection_score: float
    selection_score_label: str = "score"
    selection_scores: torch.Tensor | None = None
    selected_metrics: dict[str, float] = field(default_factory=dict)
    metric_vectors: dict[str, torch.Tensor] = field(default_factory=dict)
    selected_point_cloud_world: torch.Tensor | None = None

    @property
    def selected_pose_world(self) -> PoseTW:
        pose = _pose_at(self.candidates.poses_world_cam(), self.selected_valid_index)
        return rotate_yaw_cw90(pose)

    @property
    def selected_view(self) -> CameraTW:
        views = self.candidates.views
        if getattr(views, "ndim", 1) > 1:
            return views[self.selected_valid_index]
        return views


@dataclass(slots=True)
class CounterfactualTrajectory:
    """One rollout trajectory rooted at one initial pose."""

    root_pose_world: PoseTW
    steps: list[CounterfactualStepResult] = field(default_factory=list)
    cumulative_score: float = 0.0
    cumulative_rri: float | None = None
    terminated_early: bool = False

    def final_pose_world(self) -> PoseTW:
        if not self.steps:
            return self.root_pose_world
        return self.steps[-1].selected_pose_world

    def pose_chain_world(self) -> PoseTW:
        rows = [_pose_row(self.root_pose_world)]
        rows.extend(_pose_row(step.selected_pose_world) for step in self.steps)
        return PoseTW(torch.cat(rows, dim=0))

    def history_centers_world(self) -> torch.Tensor:
        return self.pose_chain_world().t.reshape(-1, 3)

    def reference_pose_world(self, step_index: int) -> PoseTW:
        if step_index <= 0 or not self.steps:
            return self.root_pose_world
        return self.steps[step_index - 1].selected_pose_world

    def with_appended_step(self, step: CounterfactualStepResult) -> "CounterfactualTrajectory":
        step_rri = step.selected_metrics.get("rri")
        cumulative_rri = self.cumulative_rri
        if step_rri is not None:
            cumulative_rri = float(step_rri) if cumulative_rri is None else float(cumulative_rri + step_rri)
        return CounterfactualTrajectory(
            root_pose_world=self.root_pose_world,
            steps=[*self.steps, step],
            cumulative_score=float(self.cumulative_score + step.selection_score),
            cumulative_rri=cumulative_rri,
            terminated_early=False,
        )

    def mark_terminated(self) -> "CounterfactualTrajectory":
        return replace(self, terminated_early=True)

    def accumulated_points_world(self) -> torch.Tensor:
        clouds = [step.selected_point_cloud_world for step in self.steps if step.selected_point_cloud_world is not None]
        if not clouds:
            root = ensure_unbatched_pose(self.root_pose_world)
            return torch.empty((0, 3), device=root.t.device, dtype=root.t.dtype)
        return torch.cat(clouds, dim=0)


@dataclass(slots=True)
class CounterfactualRolloutResult:
    """All trajectories produced by one rollout call."""

    root_pose_world: PoseTW
    trajectories: list[CounterfactualTrajectory]
    horizon: int
    branch_factor: int
    beam_width: int | None
    selection_policy: str | CounterfactualSelectionPolicy
    score_label: str = "score"


CounterfactualEvaluatorFn = Callable[
    [CandidateSamplingResult, CounterfactualTrajectory, int],
    CounterfactualCandidateEvaluation | torch.Tensor,
]


class CounterfactualPoseGeneratorConfig(BaseConfig):
    """Configuration for multi-step counterfactual rollout generation."""

    @property
    def target(self) -> type["CounterfactualPoseGenerator"]:
        return CounterfactualPoseGenerator

    candidate_config: CandidateViewGeneratorConfig = Field(default_factory=CandidateViewGeneratorConfig)
    horizon: int = 3
    branch_factor: int = 2
    beam_width: int | None = None
    selection_policy: CounterfactualSelectionPolicy = CounterfactualSelectionPolicy.FARTHEST_FROM_HISTORY
    min_history_distance_m: float = 0.0
    min_sibling_distance_m: float = 0.0
    seed: int | None = 0
    verbosity: Verbosity = Field(default=Verbosity.NORMAL)
    is_debug: bool = False

    _coerce_verbosity = field_validator("verbosity", mode="before")(BaseConfig._coerce_verbosity)
    _non_negative_seed = field_validator("seed")(BaseConfig._validate_non_negative_seed)

    @field_validator("horizon", "branch_factor")
    @classmethod
    def _positive_ints(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("horizon and branch_factor must be >= 1.")
        return int(value)

    @field_validator("beam_width")
    @classmethod
    def _positive_beam(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if int(value) <= 0:
            raise ValueError("beam_width must be >= 1 when provided.")
        return int(value)

    @field_validator("min_history_distance_m", "min_sibling_distance_m")
    @classmethod
    def _non_negative_distance(cls, value: float) -> float:
        value = float(value)
        if value < 0.0:
            raise ValueError("Distance thresholds must be >= 0.")
        return value


class CounterfactualOracleRriScorerConfig(BaseConfig):
    """Config-as-factory wrapper for oracle-RRI rollout scoring."""

    @property
    def target(self) -> type["CounterfactualOracleRriScorer"]:
        return CounterfactualOracleRriScorer

    depth: CandidateDepthRendererConfig = Field(default_factory=CandidateDepthRendererConfig)
    oracle: OracleRRIConfig = Field(default_factory=OracleRRIConfig)
    backprojection_stride: int = 1
    verbosity: Verbosity = Field(default=Verbosity.NORMAL)
    is_debug: bool = False

    _coerce_verbosity = field_validator("verbosity", mode="before")(BaseConfig._coerce_verbosity)

    @field_validator("backprojection_stride")
    @classmethod
    def _positive_stride(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("backprojection_stride must be >= 1.")
        return int(value)


class CounterfactualOracleRriScorer:
    """Evaluate valid candidates with oracle RRI relative to the current trajectory."""

    def __init__(self, config: CounterfactualOracleRriScorerConfig, *, sample: EfmSnippetView) -> None:
        self.config = config
        self.sample = sample
        self.console = (
            Console.with_prefix(self.__class__.__name__)
            .set_verbosity(self.config.verbosity)
            .set_debug(self.config.is_debug)
        )
        self._depth_renderer = self.config.depth.setup_target()
        self._oracle = self.config.oracle.setup_target()

    def __call__(
        self,
        candidates: CandidateSamplingResult,
        trajectory: CounterfactualTrajectory,
        step_index: int,
    ) -> CounterfactualCandidateEvaluation:
        del step_index

        if self.sample.mesh_verts is None or self.sample.mesh_faces is None:
            raise ValueError("CounterfactualOracleRriScorer requires sample.mesh_verts and sample.mesh_faces.")

        depths = self._depth_renderer.render(self.sample, candidates)
        point_clouds = build_candidate_pointclouds(
            self.sample,
            depths,
            stride=self.config.backprojection_stride,
        )

        history_points = trajectory.accumulated_points_world()
        points_t = point_clouds.semidense_points
        if history_points.numel() > 0:
            points_t = torch.cat(
                [
                    points_t,
                    history_points.to(device=points_t.device, dtype=points_t.dtype),
                ],
                dim=0,
            )

        rri = self._oracle.score(
            points_t=points_t,
            points_q=point_clouds.points,
            lengths_q=point_clouds.lengths,
            gt_verts=self.sample.mesh_verts.to(device=point_clouds.points.device, dtype=point_clouds.points.dtype),
            gt_faces=self.sample.mesh_faces.to(device=point_clouds.points.device),
            extend=point_clouds.occupancy_bounds,
        )

        return CounterfactualCandidateEvaluation(
            scores=rri.rri,
            score_label="oracle_rri",
            metric_vectors={"rri": rri.rri},
            candidate_point_clouds_world=point_clouds.points,
            candidate_point_cloud_lengths=point_clouds.lengths,
        )


class CounterfactualPoseGenerator:
    """Expand a multi-step counterfactual pose tree from the current generator."""

    def __init__(self, config: CounterfactualPoseGeneratorConfig) -> None:
        self.config = config
        self.console = (
            Console.with_prefix(self.__class__.__name__)
            .set_verbosity(self.config.verbosity)
            .set_debug(self.config.is_debug)
        )
        self._candidate_generator: CandidateViewGenerator = self.config.candidate_config.setup_target()
        self._selection_generator = torch.Generator(device="cpu")
        if self.config.seed is not None:
            self._selection_generator.manual_seed(int(self.config.seed))

    @staticmethod
    def _canonicalize_pose(reference_pose: PoseTW) -> PoseTW:
        return rotate_yaw_cw90(ensure_unbatched_pose(reference_pose))

    @staticmethod
    def _generator_input_pose(reference_pose_world: PoseTW) -> PoseTW:
        return rotate_yaw_cw90(ensure_unbatched_pose(reference_pose_world), undo=True)

    def generate_from_typed_sample(
        self,
        sample: EfmSnippetView,
        *,
        reference_pose: PoseTW | None = None,
        score_candidates: CounterfactualEvaluatorFn | None = None,
    ) -> CounterfactualRolloutResult:
        """Generate rollout trajectories directly from one typed snippet."""

        if sample.mesh is None or sample.mesh_verts is None or sample.mesh_faces is None:
            raise ValueError("Counterfactual rollouts require sample mesh, mesh_verts, and mesh_faces.")
        device = torch.device(self.config.candidate_config.device)
        cam_view = sample.get_camera(self.config.candidate_config.camera_label)
        resolved_pose = sample.trajectory.final_pose if reference_pose is None else reference_pose
        return self.generate(
            reference_pose=resolved_pose.to(device=device),
            gt_mesh=sample.mesh,
            mesh_verts=sample.mesh_verts.to(device=device),
            mesh_faces=sample.mesh_faces.to(device=device),
            camera_calib_template=cam_view.calib.to(device=device),
            occupancy_extent=sample.get_occupancy_extend().to(device=device, dtype=torch.float32),
            score_candidates=score_candidates,
        )

    def generate(
        self,
        *,
        reference_pose: PoseTW,
        gt_mesh: "trimesh.Trimesh",
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
        camera_calib_template: CameraTW,
        occupancy_extent: torch.Tensor,
        score_candidates: CounterfactualEvaluatorFn | None = None,
    ) -> CounterfactualRolloutResult:
        """Generate multi-step counterfactual rollouts from one root pose."""

        root_pose_world = self._canonicalize_pose(reference_pose)
        frontier = [CounterfactualTrajectory(root_pose_world=root_pose_world)]
        score_label = self.config.selection_policy.value

        for step_index in range(self.config.horizon):
            self.console.dbg(
                f"Expanding counterfactual rollout step {step_index + 1}/{self.config.horizon}.",
            )
            next_frontier: list[CounterfactualTrajectory] = []
            for trajectory in frontier:
                candidates = self._candidate_generator.generate(
                    reference_pose=self._generator_input_pose(trajectory.final_pose_world()),
                    gt_mesh=gt_mesh,
                    mesh_verts=mesh_verts,
                    mesh_faces=mesh_faces,
                    camera_calib_template=camera_calib_template,
                    occupancy_extent=occupancy_extent,
                )
                valid_count = int(candidates.mask_valid.sum().item())
                if valid_count <= 0:
                    next_frontier.append(trajectory.mark_terminated())
                    continue

                evaluation = self._evaluate_valid_candidates(
                    result=candidates,
                    trajectory=trajectory,
                    step_index=step_index,
                    score_candidates=score_candidates,
                )
                score_label = evaluation.score_label
                valid_indices = self._select_valid_indices(
                    scores=evaluation.scores,
                    valid_poses=candidates.poses_world_cam(),
                    trajectory=trajectory,
                )
                if not valid_indices:
                    next_frontier.append(trajectory.mark_terminated())
                    continue

                for valid_index in valid_indices:
                    shell_valid = torch.nonzero(candidates.mask_valid, as_tuple=False).reshape(-1)
                    selected_shell_index = int(shell_valid[valid_index].item())
                    step = CounterfactualStepResult(
                        step_index=step_index,
                        candidates=candidates,
                        selected_valid_index=valid_index,
                        selected_shell_index=selected_shell_index,
                        selection_score=float(evaluation.scores[valid_index].item()),
                        selection_score_label=evaluation.score_label,
                        selection_scores=evaluation.scores.detach().clone(),
                        selected_metrics=evaluation.selected_metrics(valid_index),
                        metric_vectors={
                            name: values.detach().clone() for name, values in evaluation.metric_vectors.items()
                        },
                        selected_point_cloud_world=(
                            None
                            if evaluation.selected_point_cloud(valid_index) is None
                            else evaluation.selected_point_cloud(valid_index).detach().clone()
                        ),
                    )
                    next_frontier.append(trajectory.with_appended_step(step))

            frontier = self._apply_beam_width(next_frontier)
            if not frontier:
                frontier = [CounterfactualTrajectory(root_pose_world=root_pose_world, terminated_early=True)]
                break

        return CounterfactualRolloutResult(
            root_pose_world=root_pose_world,
            trajectories=frontier,
            horizon=self.config.horizon,
            branch_factor=self.config.branch_factor,
            beam_width=self.config.beam_width,
            selection_policy=self.config.selection_policy,
            score_label=score_label,
        )

    def _evaluate_valid_candidates(
        self,
        *,
        result: CandidateSamplingResult,
        trajectory: CounterfactualTrajectory,
        step_index: int,
        score_candidates: CounterfactualEvaluatorFn | None,
    ) -> CounterfactualCandidateEvaluation:
        valid_poses = result.poses_world_cam()
        num_valid = _pose_batch_len(valid_poses)
        device = valid_poses.t.device
        dtype = valid_poses.t.dtype

        if score_candidates is not None:
            raw_eval = score_candidates(result, trajectory, step_index)
            if isinstance(raw_eval, CounterfactualCandidateEvaluation):
                return raw_eval.validate(num_valid=num_valid, device=device, dtype=dtype)
            return CounterfactualCandidateEvaluation(
                scores=torch.as_tensor(raw_eval, device=device, dtype=dtype),
                score_label="score",
            ).validate(num_valid=num_valid, device=device, dtype=dtype)

        scores = self._builtin_scores(valid_poses=valid_poses, trajectory=trajectory)
        return CounterfactualCandidateEvaluation(
            scores=scores,
            score_label=self.config.selection_policy.value,
        )

    def _builtin_scores(
        self,
        *,
        valid_poses: PoseTW,
        trajectory: CounterfactualTrajectory,
    ) -> torch.Tensor:
        centers = valid_poses.t.reshape(-1, 3)
        if self.config.selection_policy is CounterfactualSelectionPolicy.RANDOM:
            return torch.rand(centers.shape[0], generator=self._selection_generator, device="cpu").to(
                device=centers.device,
                dtype=centers.dtype,
            )

        if self.config.selection_policy is CounterfactualSelectionPolicy.FARTHEST_FROM_REFERENCE:
            reference_center = ensure_unbatched_pose(trajectory.final_pose_world()).t.reshape(1, 3)
            return torch.linalg.norm(centers - reference_center, dim=1)

        history = trajectory.history_centers_world().to(device=centers.device, dtype=centers.dtype)
        distances = torch.cdist(centers, history)
        return distances.min(dim=1).values

    def _select_valid_indices(
        self,
        *,
        scores: torch.Tensor,
        valid_poses: PoseTW,
        trajectory: CounterfactualTrajectory,
    ) -> list[int]:
        order = torch.argsort(scores, descending=True)
        centers = valid_poses.t.reshape(-1, 3)
        history = trajectory.history_centers_world().to(device=centers.device, dtype=centers.dtype)

        selected: list[int] = []
        selected_centers: list[torch.Tensor] = []
        for index_tensor in order:
            index = int(index_tensor.item())
            center = centers[index]
            if not self._passes_distance_guards(center=center, history=history, selected_centers=selected_centers):
                continue
            selected.append(index)
            selected_centers.append(center)
            if len(selected) >= self.config.branch_factor:
                break

        if not selected and order.numel() > 0:
            selected.append(int(order[0].item()))
        return selected

    def _passes_distance_guards(
        self,
        *,
        center: torch.Tensor,
        history: torch.Tensor,
        selected_centers: list[torch.Tensor],
    ) -> bool:
        if self.config.min_history_distance_m > 0.0 and history.numel() > 0:
            history_dists = torch.linalg.norm(history - center.reshape(1, 3), dim=1)
            if bool((history_dists < self.config.min_history_distance_m).any().item()):
                return False

        if self.config.min_sibling_distance_m > 0.0 and selected_centers:
            sibling = torch.stack(selected_centers, dim=0)
            sibling_dists = torch.linalg.norm(sibling - center.reshape(1, 3), dim=1)
            if bool((sibling_dists < self.config.min_sibling_distance_m).any().item()):
                return False

        return True

    def _apply_beam_width(self, trajectories: list[CounterfactualTrajectory]) -> list[CounterfactualTrajectory]:
        if self.config.beam_width is None or len(trajectories) <= self.config.beam_width:
            return trajectories
        return sorted(trajectories, key=lambda trajectory: trajectory.cumulative_score, reverse=True)[
            : self.config.beam_width
        ]


__all__ = [
    "CounterfactualCandidateEvaluation",
    "CounterfactualEvaluatorFn",
    "CounterfactualOracleRriScorer",
    "CounterfactualOracleRriScorerConfig",
    "CounterfactualPoseGenerator",
    "CounterfactualPoseGeneratorConfig",
    "CounterfactualRolloutResult",
    "CounterfactualSelectionPolicy",
    "CounterfactualStepResult",
    "CounterfactualTrajectory",
]
