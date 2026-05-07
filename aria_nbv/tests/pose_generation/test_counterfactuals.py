"""Tests for multi-step counterfactual pose rollout utilities."""

# ruff: noqa: S101

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("efm3d")

import plotly.graph_objects as go  # type: ignore[import-untyped]
import torch
import trimesh
from efm3d.aria import CameraTW, PoseTW

from aria_nbv.pose_generation import (
    CandidateSamplingResult,
    CandidateViewGeneratorConfig,
    CounterfactualCandidateEvaluation,
    CounterfactualPoseGenerator,
    CounterfactualPoseGeneratorConfig,
    CounterfactualSelectionPolicy,
    RolloutTrace,
    SamplingStrategy,
    read_rollout_traces,
    traces_from_rollout_result,
    write_rollout_traces,
)
from aria_nbv.pose_generation.plotting import (
    plot_counterfactual_paths_simple,
    plot_counterfactual_step_simple,
)
from aria_nbv.rendering import CandidateDepthRendererConfig


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
    selection_policy: CounterfactualSelectionPolicy = CounterfactualSelectionPolicy.FARTHEST_FROM_HISTORY,
    selection_temperature: float = 1.0,
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
        selection_policy=selection_policy,
        selection_temperature=selection_temperature,
        verbosity=0,
    )


def _run_rollouts(
    *,
    horizon: int = 3,
    branch_factor: int = 1,
    beam_width: int | None = None,
    selection_policy: CounterfactualSelectionPolicy = CounterfactualSelectionPolicy.FARTHEST_FROM_HISTORY,
    selection_temperature: float = 1.0,
    score_candidates=None,
):
    cfg = _make_rollout_config(
        horizon=horizon,
        branch_factor=branch_factor,
        beam_width=beam_width,
        selection_policy=selection_policy,
        selection_temperature=selection_temperature,
    )
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


def test_temperature_softmax_masks_invalid_candidates_and_reproduces_selection() -> None:
    rollouts_a = _run_rollouts(
        horizon=1,
        branch_factor=1,
        selection_policy=CounterfactualSelectionPolicy.TEMPERATURE_SOFTMAX,
        selection_temperature=1.0,
        score_candidates=_fake_rri_evaluator,
    )
    rollouts_b = _run_rollouts(
        horizon=1,
        branch_factor=1,
        selection_policy=CounterfactualSelectionPolicy.TEMPERATURE_SOFTMAX,
        selection_temperature=1.0,
        score_candidates=_fake_rri_evaluator,
    )

    step_a = rollouts_a.trajectories[0].steps[0]
    step_b = rollouts_b.trajectories[0].steps[0]

    assert step_a.selection_policy == CounterfactualSelectionPolicy.TEMPERATURE_SOFTMAX.value
    assert step_a.selection_probabilities is not None
    assert torch.all(step_a.selection_probabilities >= 0)
    assert torch.isclose(step_a.selection_probabilities.sum(), torch.tensor(1.0))
    assert step_a.selected_valid_index == step_b.selected_valid_index
    assert step_a.selected_shell_index == step_b.selected_shell_index

    trace = traces_from_rollout_result(rollouts_a, rollout_id_prefix="temp-softmax")[0]
    trace_step = trace.steps[0]
    assert trace_step.selection_probabilities is not None
    invalid_probs = trace_step.selection_probabilities[~trace_step.candidate_valid]
    assert torch.equal(invalid_probs, torch.zeros_like(invalid_probs))
    assert torch.isclose(trace_step.selection_probabilities[trace_step.candidate_valid].sum(), torch.tensor(1.0))
    assert trace_step.selection_temperature == pytest.approx(1.0)
    assert trace_step.selected_log_probability is not None


def test_temperature_softmax_branch_factor_samples_distinct_candidates() -> None:
    rollouts = _run_rollouts(
        horizon=1,
        branch_factor=3,
        selection_policy=CounterfactualSelectionPolicy.TEMPERATURE_SOFTMAX,
        selection_temperature=1.0,
        score_candidates=_fake_rri_evaluator,
    )

    selected = [trajectory.steps[0].selected_shell_index for trajectory in rollouts.trajectories]

    assert len(selected) == 3
    assert len(set(selected)) == 3


def test_counterfactual_path_plot_uses_rri_colorbar_when_available() -> None:
    rollouts = _run_rollouts(horizon=2, branch_factor=1, score_candidates=_fake_rri_evaluator)

    fig = plot_counterfactual_paths_simple(rollouts)

    assert isinstance(fig, go.Figure)
    assert any("rri=" in str(trace.name) for trace in fig.data if getattr(trace, "name", None))
    assert any(
        getattr(getattr(trace, "marker", None), "showscale", False) for trace in fig.data if hasattr(trace, "marker")
    )


def test_candidate_sampling_roundtrip_preserves_valid_pose_order_and_cw90_display_is_read_only() -> None:
    cfg = _make_rollout_config(horizon=1, branch_factor=1)
    generator = CounterfactualPoseGenerator(cfg)
    mesh, verts, faces = _mesh_triplet(cfg.candidate_config.device)
    candidates = generator._candidate_generator.generate(  # noqa: SLF001
        reference_pose=_identity_pose(device=cfg.candidate_config.device),
        gt_mesh=mesh,
        mesh_verts=verts,
        mesh_faces=faces,
        camera_calib_template=_dummy_camera(cfg.candidate_config.device),
        occupancy_extent=_default_extent(cfg.candidate_config.device),
    )
    views_before = candidates.views.tensor().detach().clone()
    reference_before = candidates.reference_pose.tensor().detach().clone()
    poses_before = candidates.poses_world_cam().tensor().detach().clone()

    candidates.get_offsets_and_dirs_ref(display_rotate=True)

    assert torch.equal(candidates.views.tensor(), views_before)
    assert torch.equal(candidates.reference_pose.tensor(), reference_before)
    assert torch.equal(candidates.poses_world_cam().tensor(), poses_before)

    decoded = type(candidates).from_serializable(candidates.to_serializable(), device=torch.device("cpu"))

    assert torch.equal(decoded.mask_valid.cpu(), candidates.mask_valid.cpu())
    assert torch.allclose(decoded.shell_poses.tensor(), candidates.shell_poses.to(device="cpu").tensor())
    assert torch.allclose(decoded.poses_world_cam().tensor(), candidates.poses_world_cam().to(device="cpu").tensor())


def test_candidate_depth_renderer_reports_full_shell_candidate_indices() -> None:
    shell = PoseTW(
        torch.cat(
            [
                torch.eye(3, dtype=torch.float32).reshape(1, 9).repeat(4, 1),
                torch.arange(4, dtype=torch.float32).reshape(4, 1).expand(4, 3),
            ],
            dim=1,
        )
    )
    camera = _dummy_camera()
    camera_data = camera.tensor().repeat(2, 1)
    candidates = CandidateSamplingResult(
        views=CameraTW(camera_data),
        reference_pose=_identity_pose(),
        mask_valid=torch.tensor([False, True, False, True]),
        masks={},
        shell_poses=shell,
    )
    renderer = CandidateDepthRendererConfig(device="cpu", max_candidates_final=2, verbosity=0).setup_target()

    _, _, candidate_indices = renderer._select_candidate_views(candidates)  # noqa: SLF001

    assert candidate_indices.tolist() == [1, 3]


def test_candidate_depth_renderer_rejects_ambiguous_candidate_index_mapping() -> None:
    shell = PoseTW(
        torch.cat(
            [
                torch.eye(3, dtype=torch.float32).reshape(1, 9).repeat(4, 1),
                torch.arange(4, dtype=torch.float32).reshape(4, 1).expand(4, 3),
            ],
            dim=1,
        )
    )
    camera = _dummy_camera()
    candidates = CandidateSamplingResult(
        views=CameraTW(camera.tensor().repeat(3, 1)),
        reference_pose=_identity_pose(),
        mask_valid=torch.tensor([False, True, False, True]),
        masks={},
        shell_poses=shell,
    )
    renderer = CandidateDepthRendererConfig(device="cpu", max_candidates_final=3, verbosity=0).setup_target()

    with pytest.raises(ValueError, match="Candidate views cannot be mapped"):
        renderer._select_candidate_views(candidates)  # noqa: SLF001


def test_rollout_trace_maps_scores_to_full_candidate_shell_and_roundtrips(tmp_path: Path) -> None:
    rollouts = _run_rollouts(horizon=2, branch_factor=1, score_candidates=_fake_rri_evaluator)
    traces = traces_from_rollout_result(
        rollouts,
        rollout_id_prefix="test-rollout",
        scene_id="scene",
        snippet_id="snippet",
        candidate_config_hash="candidate-hash",
        oracle_config_hash="oracle-hash",
        random_seed=0,
    )

    assert len(traces) == 1
    trace = traces[0]
    assert isinstance(trace, RolloutTrace)
    assert trace.lineage.rollout_id == "test-rollout-000000"
    assert trace.lineage.candidate_config_hash == "candidate-hash"
    assert trace.termination_reason == "fixed_horizon"

    first_step = trace.steps[0]
    assert first_step.candidate_valid.shape[0] == first_step.candidate_poses_world_cam.shape[0]
    assert first_step.candidate_scores is not None
    valid_indices = torch.nonzero(first_step.candidate_valid, as_tuple=False).reshape(-1)
    expected_valid_scores = torch.linspace(
        0.1,
        0.1 * valid_indices.numel(),
        valid_indices.numel(),
        dtype=first_step.candidate_scores.dtype,
    )
    assert torch.allclose(first_step.candidate_scores[valid_indices], expected_valid_scores)
    assert torch.isnan(first_step.candidate_scores[~first_step.candidate_valid]).all()
    assert torch.isclose(
        first_step.candidate_scores[first_step.selected_shell_index],
        torch.tensor(first_step.selection_score, dtype=first_step.candidate_scores.dtype),
    )
    assert first_step.metric_vectors["rri"].shape == first_step.candidate_valid.shape
    assert torch.allclose(first_step.metric_vectors["rri"][valid_indices], expected_valid_scores)
    assert torch.isnan(first_step.metric_vectors["rri"][~first_step.candidate_valid]).all()
    assert first_step.cumulative_rri == pytest.approx(first_step.selected_metrics["rri"])

    path = write_rollout_traces(tmp_path / "rollouts.msgpack", traces)
    decoded = read_rollout_traces(path)

    assert decoded[0].lineage.scene_id == "scene"
    assert decoded[0].steps[0].selected_shell_index == first_step.selected_shell_index
    assert torch.equal(decoded[0].steps[0].candidate_valid, first_step.candidate_valid)
    assert decoded[0].steps[0].candidate_scores is not None
    assert torch.allclose(
        decoded[0].steps[0].candidate_scores,
        first_step.candidate_scores,
        equal_nan=True,
    )
