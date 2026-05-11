"""Target-aware oracle RRI scoring for counterfactual rollouts.

This module scores valid candidate rows with target-specific point-mesh RRI.
The actor selects an observed/predicted target record upstream; this scorer then
uses the matched GT OBB only as an oracle/evaluation crop. Missing GT matches,
ambiguous matches, empty mesh crops, sparse current support, or unusable depth
are expected invalidity cases and surface as `TargetRriInvalidError`.

Scene RRI may be emitted as a diagnostic from the same candidate point clouds,
but it must not replace target RRI labels in thesis-core rollout stores.
"""

from __future__ import annotations

from typing import Any

import torch
from efm3d.aria.obb import ObbTW
from pydantic import Field, field_validator

# TODO(fix: TargetCandidateRow and target_gt_obb_world are not recognized as valid imports!)
from ..data_handling import EfmSnippetView, TargetCandidateRow, target_gt_obb_world
from ..rendering.candidate_depth_renderer import CandidateDepthRendererConfig
from ..rendering.candidate_pointclouds import build_candidate_pointclouds
from ..rri_metrics.oracle_rri import OracleRRIConfig
from ..utils import BaseConfig, Console, TargetConfig, Verbosity
from .counterfactuals import (
    CounterfactualCandidateEvaluation,
    CounterfactualTrajectory,
)
from .types import CandidateSamplingResult

TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1 = "gt_obb_oriented_any_vertex_v1"
"""Target crop policy: keep mesh faces with any vertex inside the matched oriented GT OBB."""

SCENE_CROP_POLICY_SNIPPET_EXTENT_V1 = "snippet_occupancy_extent_v1"
"""Scene-RRI crop policy matching the existing snippet occupancy-extent scorer."""


class TargetRriInvalidError(ValueError):
    """Expected data invalidity that prevents target-RRI labeling for a row."""


class CounterfactualTargetOracleRriScorerConfig(TargetConfig["CounterfactualTargetOracleRriScorer"]):
    """Config-as-factory wrapper for target-cropped oracle-RRI rollout scoring."""

    @property
    def target_type(self) -> type["CounterfactualTargetOracleRriScorer"]:
        return CounterfactualTargetOracleRriScorer

    depth: CandidateDepthRendererConfig = Field(default_factory=CandidateDepthRendererConfig)
    """Depth renderer used once per candidate table before target/scene scoring."""

    oracle: OracleRRIConfig = Field(default_factory=OracleRRIConfig)
    """Point-mesh oracle metric configuration shared by target and scene RRI."""

    backprojection_stride: int = Field(default=1, ge=1)
    """Pixel stride for backprojecting rendered candidate depths."""

    target_crop_margin_m: float = Field(default=0.0, ge=0.0)
    """Optional symmetric margin applied in GT-OBB local coordinates."""

    min_current_target_points: int = Field(default=1, ge=1)
    """Minimum current observed/support points inside the target crop."""

    include_scene_rri: bool = True
    """Whether to compute diagnostic scene RRI from the same point-cloud batch."""

    target_crop_policy: str = TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1
    """Explicit target mesh crop policy stored as rollout lineage."""

    scene_crop_policy: str = SCENE_CROP_POLICY_SNIPPET_EXTENT_V1
    """Diagnostic scene-RRI crop policy matching the scene-level scorer."""

    verbosity: Verbosity = Field(default=Verbosity.NORMAL)
    """Console verbosity."""

    is_debug: bool = False
    """Enable debug logging in scorer dependencies."""

    _coerce_verbosity = field_validator("verbosity", mode="before")(BaseConfig._coerce_verbosity)

    @field_validator("target_crop_policy")
    @classmethod
    def _known_target_crop_policy(cls, value: str) -> str:
        if value != TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1:
            raise ValueError(f"Unsupported target_crop_policy={value!r}.")
        return value

    @field_validator("scene_crop_policy")
    @classmethod
    def _known_scene_crop_policy(cls, value: str) -> str:
        if value != SCENE_CROP_POLICY_SNIPPET_EXTENT_V1:
            raise ValueError(f"Unsupported scene_crop_policy={value!r}.")
        return value


class CounterfactualTargetOracleRriScorer:
    """Evaluate valid candidates with target-cropped oracle RRI.

    The scorer renders candidate depth, backprojects world-frame point clouds,
    crops current and candidate points to the matched target OBB, crops the mesh
    with the configured policy, and returns target-RRI labels plus audit metrics.
    Invalid target crops abort the target row rather than producing low labels.
    """

    def __init__(
        self,
        config: CounterfactualTargetOracleRriScorerConfig,
        *,
        sample: EfmSnippetView,
        target_sample: Any,
        target_row: TargetCandidateRow,
    ) -> None:
        self.config = config
        self.sample = sample
        self.target_sample = target_sample
        self.target_row = target_row
        self.console = (
            Console.with_prefix(self.__class__.__name__)
            .set_verbosity(self.config.verbosity)
            .set_debug(self.config.is_debug)
        )
        self._depth_renderer = self.config.depth.setup_target()
        self._oracle = self.config.oracle.setup_target()
        try:
            self._target_obb_world = target_gt_obb_world(target_row, target_sample)
        except ValueError as exc:
            raise TargetRriInvalidError(str(exc)) from exc

    def __call__(
        self,
        candidates: CandidateSamplingResult,
        trajectory: CounterfactualTrajectory,
        step_index: int,
    ) -> CounterfactualCandidateEvaluation:
        del step_index

        if self.sample.mesh_verts is None or self.sample.mesh_faces is None:
            raise ValueError("CounterfactualTargetOracleRriScorer requires sample.mesh_verts and sample.mesh_faces.")

        depths = self._depth_renderer.render(self.sample, candidates)
        point_clouds = build_candidate_pointclouds(
            self.sample,
            depths,
            stride=self.config.backprojection_stride,
        )
        device = point_clouds.points.device
        dtype = point_clouds.points.dtype
        target_obb = self._target_obb_world.to(device=device, dtype=dtype)
        mesh_verts = self.sample.mesh_verts.to(device=device, dtype=dtype)
        mesh_faces = self.sample.mesh_faces.to(device=device)
        target_mesh_verts, target_mesh_faces = _crop_mesh_to_obb(
            mesh_verts,
            mesh_faces,
            target_obb,
            margin_m=self.config.target_crop_margin_m,
        )
        target_extent = _aabb_from_points(target_mesh_verts, margin_m=self.config.target_crop_margin_m)

        history_points = trajectory.accumulated_points_world()
        points_t = point_clouds.semidense_points
        if history_points.numel() > 0:
            points_t = torch.cat([points_t, history_points.to(device=device, dtype=dtype)], dim=0)

        target_points_t = _crop_points_to_obb(
            points_t,
            target_obb,
            margin_m=self.config.target_crop_margin_m,
        )
        if target_points_t.shape[0] < int(self.config.min_current_target_points):
            raise TargetRriInvalidError("Target crop contains too few current points for target-RRI evaluation.")

        target_points_q, target_lengths_q = _crop_padded_pointclouds_to_obb(
            point_clouds.points,
            point_clouds.lengths,
            target_obb,
            margin_m=self.config.target_crop_margin_m,
        )
        # TODO(fix): self._ocale must be typed correctly!
        target_rri = self._oracle.score(
            points_t=target_points_t,
            points_q=target_points_q,
            lengths_q=target_lengths_q,
            gt_verts=target_mesh_verts,
            gt_faces=target_mesh_faces,
            extend=target_extent,
        )
        metric_vectors = {
            "rri": target_rri.rri,
            "target_rri": target_rri.rri,
            "target_pm_dist_before": target_rri.pm_dist_before,
            "target_pm_dist_after": target_rri.pm_dist_after,
            "target_pm_acc_before": target_rri.pm_acc_before,
            "target_pm_comp_before": target_rri.pm_comp_before,
            "target_pm_acc_after": target_rri.pm_acc_after,
            "target_pm_comp_after": target_rri.pm_comp_after,
            "target_candidate_support": target_lengths_q.to(device=device, dtype=dtype),
            "target_current_support": torch.full_like(target_rri.rri, float(target_points_t.shape[0])),
        }

        if self.config.include_scene_rri:
            scene_rri = self._oracle.score(
                points_t=points_t,
                points_q=point_clouds.points,
                lengths_q=point_clouds.lengths,
                gt_verts=mesh_verts,
                gt_faces=mesh_faces,
                extend=point_clouds.occupancy_bounds,
            )
            metric_vectors.update(
                {
                    "scene_rri": scene_rri.rri,
                    "scene_pm_dist_before": scene_rri.pm_dist_before,
                    "scene_pm_dist_after": scene_rri.pm_dist_after,
                    "scene_pm_acc_before": scene_rri.pm_acc_before,
                    "scene_pm_comp_before": scene_rri.pm_comp_before,
                    "scene_pm_acc_after": scene_rri.pm_acc_after,
                    "scene_pm_comp_after": scene_rri.pm_comp_after,
                }
            )

        return CounterfactualCandidateEvaluation(
            scores=target_rri.rri,
            score_label="target_rri",
            metric_vectors=metric_vectors,
            candidate_point_clouds_world=point_clouds.points,
            candidate_point_cloud_lengths=point_clouds.lengths,
        )


def _crop_points_to_obb(points: torch.Tensor, obb: ObbTW, *, margin_m: float = 0.0) -> torch.Tensor:
    if points.numel() == 0:
        return points.reshape(0, 3)
    pts = points.reshape(-1, points.shape[-1])[:, :3]
    mask = _points_inside_obb_mask(pts, obb, margin_m=margin_m)
    return pts[mask]


def _crop_padded_pointclouds_to_obb(
    points: torch.Tensor,
    lengths: torch.Tensor,
    obb: ObbTW,
    *,
    margin_m: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    cropped: list[torch.Tensor] = []
    lengths_out: list[int] = []
    for row_index in range(points.shape[0]):
        length = int(lengths[row_index].detach().cpu().item())
        row = _crop_points_to_obb(points[row_index, :length, :3], obb, margin_m=margin_m)
        cropped.append(row)
        lengths_out.append(int(row.shape[0]))
    max_len = max(max(lengths_out), 1)
    output = torch.zeros((points.shape[0], max_len, 3), device=points.device, dtype=points.dtype)
    for row_index, row in enumerate(cropped):
        if row.numel() > 0:
            output[row_index, : row.shape[0], :] = row.to(device=points.device, dtype=points.dtype)
    return output, torch.tensor(lengths_out, device=points.device, dtype=torch.long)


def _crop_mesh_to_obb(
    verts: torch.Tensor,
    faces: torch.Tensor,
    obb: ObbTW,
    *,
    margin_m: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if verts.numel() == 0 or faces.numel() == 0:
        raise TargetRriInvalidError("Target oriented OBB crop requires a non-empty scene mesh.")
    vertex_inside = _points_inside_obb_mask(verts.reshape(-1, 3), obb, margin_m=margin_m)
    face_indices = faces.reshape(-1, 3).to(device=verts.device, dtype=torch.long)
    face_keep = vertex_inside[face_indices].any(dim=1)
    if not bool(face_keep.any().item()):
        raise TargetRriInvalidError("Target oriented OBB crop contains no mesh faces.")
    kept_faces = face_indices[face_keep]
    unique_vertices, inverse = torch.unique(kept_faces.reshape(-1), sorted=True, return_inverse=True)
    cropped_verts = verts[unique_vertices].reshape(-1, 3)
    cropped_faces = inverse.reshape(-1, 3).to(dtype=torch.long)
    return cropped_verts, cropped_faces


def _points_inside_obb_mask(points: torch.Tensor, obb: ObbTW, *, margin_m: float = 0.0) -> torch.Tensor:
    pts = points.reshape(-1, points.shape[-1])[:, :3]
    finite = torch.isfinite(pts).all(dim=-1)
    local = obb.T_world_object.inverse().transform(pts).reshape(-1, 3)
    lower = obb.bb3_min_object.reshape(-1, 3)[0].to(device=pts.device, dtype=pts.dtype) - float(margin_m)
    upper = obb.bb3_max_object.reshape(-1, 3)[0].to(device=pts.device, dtype=pts.dtype) + float(margin_m)
    return finite & torch.all((local >= lower) & (local <= upper), dim=-1)


def _aabb_from_points(points: torch.Tensor, *, margin_m: float = 0.0) -> torch.Tensor:
    pts = points.reshape(-1, points.shape[-1])[:, :3]
    if pts.numel() == 0:
        raise TargetRriInvalidError("Cannot build target crop extent from an empty point set.")
    lower = pts.min(dim=0).values - float(margin_m)
    upper = pts.max(dim=0).values + float(margin_m)
    return torch.stack([lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]]).to(dtype=pts.dtype)


__all__ = [
    "CounterfactualTargetOracleRriScorer",
    "CounterfactualTargetOracleRriScorerConfig",
    "SCENE_CROP_POLICY_SNIPPET_EXTENT_V1",
    "TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1",
    "TargetRriInvalidError",
]
