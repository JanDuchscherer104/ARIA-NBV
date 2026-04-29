"""Tests for candidate panel diagnostics."""

# ruff: noqa: S101, D103, SLF001

from types import SimpleNamespace

import torch
from efm3d.aria import CameraTW
from efm3d.aria.pose import PoseTW

from aria_nbv.app.panels import candidates as candidates_panel
from aria_nbv.pose_generation import (
    CandidateViewGeneratorConfig,
    CounterfactualPoseGeneratorConfig,
    CounterfactualRolloutResult,
    CounterfactualStepResult,
    CounterfactualTrajectory,
)
from aria_nbv.pose_generation.types import CandidateSamplingResult

ORTHO_TOL = 1e-6
ORTHO_ERR = 1e-3


def test_pose_orthonormality_stats_identity() -> None:
    pose = PoseTW.from_Rt(torch.eye(3), torch.zeros(3))
    stats = candidates_panel._pose_orthonormality_stats(pose)
    assert stats["orth_max"] < ORTHO_TOL
    assert stats["orth_mean"] < ORTHO_TOL
    assert stats["axis_norm_max"] < ORTHO_TOL
    assert abs(stats["det_mean"] - 1.0) < ORTHO_TOL


def test_pose_orthonormality_stats_detects_scale() -> None:
    r = torch.eye(3)
    r[0, 0] = 2.0
    pose = PoseTW.from_Rt(r, torch.zeros(3))
    stats = candidates_panel._pose_orthonormality_stats(pose)
    assert stats["orth_max"] > ORTHO_ERR
    assert stats["axis_norm_max"] > ORTHO_ERR
    assert stats["det_mean"] > 1.0


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


def _candidate_result_for_pose(pose: PoseTW) -> CandidateSamplingResult:
    return CandidateSamplingResult(
        views=_dummy_camera(),
        reference_pose=pose,
        mask_valid=torch.tensor([True]),
        masks={},
        shell_poses=pose,
    )


def test_counterfactual_cache_key_tracks_sample_and_config() -> None:
    sample = SimpleNamespace(scene_id="scene_a", snippet_id="snippet_1")
    cand_cfg = CandidateViewGeneratorConfig()
    cfg_a = CounterfactualPoseGeneratorConfig(candidate_config=cand_cfg, horizon=2, branch_factor=1)
    cfg_b = CounterfactualPoseGeneratorConfig(candidate_config=cand_cfg, horizon=3, branch_factor=1)

    key_a = candidates_panel._counterfactual_cache_key(sample, cand_cfg, cfg_a)
    key_a_repeat = candidates_panel._counterfactual_cache_key(sample, cand_cfg, cfg_a)
    key_b = candidates_panel._counterfactual_cache_key(sample, cand_cfg, cfg_b)

    assert key_a == key_a_repeat
    assert key_a != key_b


def test_counterfactual_trajectory_rows_capture_step_count_score_and_final_pose() -> None:
    root_pose = PoseTW.from_Rt(torch.eye(3), torch.zeros(3))
    selected_pose = PoseTW.from_Rt(torch.eye(3), torch.tensor([1.0, 2.0, 3.0]))
    step = CounterfactualStepResult(
        step_index=0,
        candidates=_candidate_result_for_pose(selected_pose),
        selected_valid_index=0,
        selected_shell_index=0,
        selection_score=0.75,
    )
    trajectory = CounterfactualTrajectory(
        root_pose_world=root_pose,
        steps=[step],
        cumulative_score=0.75,
        terminated_early=False,
    )
    rollouts = CounterfactualRolloutResult(
        root_pose_world=root_pose,
        trajectories=[trajectory],
        horizon=1,
        branch_factor=1,
        beam_width=None,
        selection_policy="farthest_from_history",
    )

    rows = candidates_panel._counterfactual_trajectory_rows(rollouts)

    assert len(rows) == 1
    assert rows[0]["steps"] == 1
    assert rows[0]["cumulative_score"] == 0.75
    assert rows[0]["final_x"] == 1.0
    assert rows[0]["final_y"] == 2.0
    assert rows[0]["final_z"] == 3.0
