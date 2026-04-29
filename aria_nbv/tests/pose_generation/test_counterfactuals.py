"""Tests for multi-step counterfactual pose rollout utilities."""

# ruff: noqa: S101

from __future__ import annotations

import pytest

pytest.importorskip("efm3d")

import plotly.graph_objects as go  # type: ignore[import-untyped]
import torch
import trimesh
from efm3d.aria import CameraTW, PoseTW

from aria_nbv.pose_generation import (
    CandidateViewGeneratorConfig,
    CounterfactualCandidateEvaluation,
    CounterfactualPoseGenerator,
    CounterfactualPoseGeneratorConfig,
    CounterfactualSelectionPolicy,
    SamplingStrategy,
)
from aria_nbv.pose_generation.plotting import (
    plot_counterfactual_paths_simple,
    plot_counterfactual_step_simple,
)


def _identity_pose(device: torch.device | str = "cpu") -> PoseTW:
    return PoseTW(
        torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            device=device,
        )
    )


def _dummy_camera(device: torch.device | str = "cpu") -> CameraTW:
    return CameraTW.from_surreal(
        width=torch.tensor([64.0], device=device),
        height=torch.tensor([64.0], device=device),
        type_str="Pinhole",
        params=torch.tensor([[60.0, 60.0, 32.0, 32.0]], device=device),
        gain=torch.zeros(1, device=device),
        exposure_s=torch.zeros(1, device=device),
        valid_radius=torch.tensor([64.0], device=device),
        T_camera_rig=PoseTW.from_matrix3x4(torch.eye(3, 4, device=device).unsqueeze(0)),
    )


def _mesh_triplet(device: torch.device | str = "cpu") -> tuple[trimesh.Trimesh, torch.Tensor, torch.Tensor]:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    verts = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device)
    faces = torch.from_numpy(mesh.faces).to(dtype=torch.int64, device=device)
    return mesh, verts, faces


def _default_extent(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=torch.float32, device=device)


def _make_rollout_config(
    *,
    horizon: int = 3,
    branch_factor: int = 1,
    beam_width: int | None = None,
) -> CounterfactualPoseGeneratorConfig:
    candidate_cfg = CandidateViewGeneratorConfig(
        num_samples=16,
        min_radius=0.5,
        max_radius=0.5,
        ensure_collision_free=False,
        ensure_free_space=False,
        min_distance_to_mesh=0.0,
        sampling_strategy=SamplingStrategy.UNIFORM_SPHERE,
        view_max_azimuth_deg=0.0,
        view_max_elevation_deg=0.0,
        verbosity=0,
        seed=0,
        is_debug=True,
    )
    return CounterfactualPoseGeneratorConfig(
        candidate_config=candidate_cfg,
        horizon=horizon,
        branch_factor=branch_factor,
        beam_width=beam_width,
        selection_policy=CounterfactualSelectionPolicy.FARTHEST_FROM_HISTORY,
        verbosity=0,
    )


def _run_rollouts(
    *,
    horizon: int = 3,
    branch_factor: int = 1,
    beam_width: int | None = None,
    score_candidates=None,
):
    cfg = _make_rollout_config(horizon=horizon, branch_factor=branch_factor, beam_width=beam_width)
    generator = CounterfactualPoseGenerator(cfg)
    mesh, verts, faces = _mesh_triplet(cfg.candidate_config.device)
    return generator.generate(
        reference_pose=_identity_pose(device=cfg.candidate_config.device),
        gt_mesh=mesh,
        mesh_verts=verts,
        mesh_faces=faces,
        camera_calib_template=_dummy_camera(cfg.candidate_config.device),
        occupancy_extent=_default_extent(cfg.candidate_config.device),
        score_candidates=score_candidates,
    )


def _fake_rri_evaluator(result, trajectory, step_index):
    del trajectory

    valid_poses = result.poses_world_cam()
    centers = valid_poses.t.reshape(-1, 3)
    scores = torch.linspace(0.1, 0.1 * centers.shape[0], centers.shape[0], device=centers.device)
    scores = scores + float(step_index)
    candidate_points = centers.unsqueeze(1).repeat(1, 2, 1)
    lengths = torch.full((centers.shape[0],), 2, dtype=torch.long, device=centers.device)
    return CounterfactualCandidateEvaluation(
        scores=scores,
        score_label="oracle_rri",
        metric_vectors={"rri": scores},
        candidate_point_clouds_world=candidate_points,
        candidate_point_cloud_lengths=lengths,
    )


def test_counterfactual_rollout_greedy_length_and_step_radius() -> None:
    rollouts = _run_rollouts(horizon=3, branch_factor=1)
    assert len(rollouts.trajectories) == 1

    trajectory = rollouts.trajectories[0]
    assert len(trajectory.steps) == 3

    positions = trajectory.pose_chain_world().t
    step_lengths = torch.linalg.norm(positions[1:] - positions[:-1], dim=1)
    assert torch.allclose(step_lengths, torch.full_like(step_lengths, 0.5), atol=1e-4)


def test_counterfactual_rollout_beam_width_caps_frontier() -> None:
    rollouts = _run_rollouts(horizon=2, branch_factor=3, beam_width=2)
    assert len(rollouts.trajectories) == 2
    assert all(len(traj.steps) == 2 for traj in rollouts.trajectories)


def test_counterfactual_simple_plots_return_figures() -> None:
    rollouts = _run_rollouts(horizon=2, branch_factor=1)
    trajectory = rollouts.trajectories[0]

    path_fig = plot_counterfactual_paths_simple(rollouts)
    step_fig = plot_counterfactual_step_simple(trajectory, step_index=0)

    assert isinstance(path_fig, go.Figure)
    assert isinstance(step_fig, go.Figure)
    assert len(path_fig.data) >= 2
    assert len(step_fig.data) >= 2


def test_counterfactual_rollout_tracks_cumulative_rri_and_selected_point_clouds() -> None:
    rollouts = _run_rollouts(horizon=2, branch_factor=1, score_candidates=_fake_rri_evaluator)
    trajectory = rollouts.trajectories[0]

    expected_rri = sum(step.selected_metrics["rri"] for step in trajectory.steps)
    assert trajectory.cumulative_rri == pytest.approx(expected_rri)
    assert trajectory.cumulative_score == pytest.approx(expected_rri)
    assert trajectory.accumulated_points_world().shape == (4, 3)
    assert rollouts.score_label == "oracle_rri"


def test_counterfactual_path_plot_uses_rri_colorbar_when_available() -> None:
    rollouts = _run_rollouts(horizon=2, branch_factor=1, score_candidates=_fake_rri_evaluator)

    fig = plot_counterfactual_paths_simple(rollouts)

    assert isinstance(fig, go.Figure)
    assert any("rri=" in str(trace.name) for trace in fig.data if getattr(trace, "name", None))
    assert any(
        getattr(getattr(trace, "marker", None), "showscale", False) for trace in fig.data if hasattr(trace, "marker")
    )
