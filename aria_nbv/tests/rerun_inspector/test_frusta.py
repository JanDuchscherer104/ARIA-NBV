"""Tests for frame-safe Rerun frustum geometry helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

CameraTW = pytest.importorskip("efm3d.aria.camera").CameraTW
PoseTW = pytest.importorskip("efm3d.aria.pose").PoseTW
PerspectiveCameras = pytest.importorskip("pytorch3d.renderer.cameras").PerspectiveCameras

from aria_nbv.rerun_inspector import (  # noqa: E402  # noqa: E402
    _colors,
    _frusta,
    apply_display_cw90,
    frusta_from_camera_tw,
    frusta_from_p3d_cameras,
)


def _make_camera(width: int = 4, height: int = 4, fx: float = 2.0, fy: float = 2.0) -> CameraTW:
    return CameraTW.from_surreal(
        width=torch.tensor([float(width)], dtype=torch.float32),
        height=torch.tensor([float(height)], dtype=torch.float32),
        type_str="Pinhole",
        params=torch.tensor([[fx, fy, width / 2.0, height / 2.0]], dtype=torch.float32),
        gain=torch.zeros(1),
        exposure_s=torch.zeros(1),
        valid_radius=torch.tensor([float(max(width, height) * 2)], dtype=torch.float32),
        T_camera_rig=PoseTW.from_matrix3x4(torch.eye(3, 4, dtype=torch.float32).unsqueeze(0)),
    )


def _pose_from_rt(rotation: torch.Tensor, translation: torch.Tensor) -> PoseTW:
    return PoseTW.from_Rt(rotation.unsqueeze(0), translation.reshape(1, 3))


def _p3d_from_pose(poses_world_cam: PoseTW, *, width: int = 4, height: int = 4) -> PerspectiveCameras:
    poses_cam_world = poses_world_cam.inverse()
    count = int(poses_cam_world.shape[0])
    return PerspectiveCameras(
        R=poses_cam_world.R.transpose(-1, -2).contiguous(),
        T=poses_cam_world.t,
        focal_length=torch.full((count, 2), 2.0, dtype=torch.float32),
        principal_point=torch.full((count, 2), 2.0, dtype=torch.float32),
        image_size=torch.tensor([[float(height), float(width)]], dtype=torch.float32).expand(count, 2).clone(),
        in_ndc=False,
    )


def test_camera_tw_frustum_endpoints_for_known_pose() -> None:
    pose = _pose_from_rt(torch.eye(3, dtype=torch.float32), torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    result = frusta_from_camera_tw(pose, _make_camera(), depth_m=2.0)

    assert result.labels == ["candidate_id=0"]
    assert len(result.line_strips) == 1
    assert result.line_strips[0].shape == (12, 3)
    np.testing.assert_allclose(result.centers_world, np.array([[1.0, 2.0, 3.0]]), atol=1e-6)
    np.testing.assert_allclose(
        result.corners_world[0],
        np.array(
            [
                [-1.0, 0.0, 5.0],
                [3.0, 0.0, 5.0],
                [3.0, 4.0, 5.0],
                [-1.0, 4.0, 5.0],
            ]
        ),
        atol=1e-6,
    )


def test_p3d_fallback_produces_finite_world_frame_strips() -> None:
    angle = torch.tensor(np.pi / 2.0, dtype=torch.float32)
    c = torch.cos(angle)
    s = torch.sin(angle)
    rotation = torch.tensor(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=torch.float32,
    )
    translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    pose = _pose_from_rt(rotation, translation)
    cameras = _p3d_from_pose(pose)

    result = frusta_from_p3d_cameras(pose, cameras, depth_m=2.0)

    assert len(result.line_strips) == 1
    assert np.isfinite(result.line_strips[0]).all()
    np.testing.assert_allclose(result.centers_world[0], translation.numpy(), atol=1e-6)

    plane_center = result.corners_world[0].mean(axis=0)
    forward_world = rotation[:, 2].numpy()
    np.testing.assert_allclose(plane_center - result.centers_world[0], forward_world * 2.0, atol=1e-5)


def test_cw90_is_display_only_and_copied() -> None:
    pose = _pose_from_rt(torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32))
    base = frusta_from_camera_tw(pose, _make_camera(), depth_m=2.0)
    base_strip_before = base.line_strips[0].copy()
    base_corners_before = base.corners_world.copy()

    display = frusta_from_camera_tw(pose, _make_camera(), depth_m=2.0, display_cw90=True)
    applied = apply_display_cw90(base, pose)

    np.testing.assert_allclose(base.line_strips[0], base_strip_before, atol=0.0)
    np.testing.assert_allclose(base.corners_world, base_corners_before, atol=0.0)
    assert not np.shares_memory(applied.line_strips[0], base.line_strips[0])
    assert not np.shares_memory(applied.corners_world, base.corners_world)
    np.testing.assert_allclose(applied.corners_world, display.corners_world, atol=1e-6)

    np.testing.assert_allclose(
        base.corners_world[0],
        np.array(
            [
                [-2.0, -2.0, 2.0],
                [2.0, -2.0, 2.0],
                [2.0, 2.0, 2.0],
                [-2.0, 2.0, 2.0],
            ]
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        display.corners_world[0],
        np.array(
            [
                [2.0, -2.0, 2.0],
                [2.0, 2.0, 2.0],
                [-2.0, 2.0, 2.0],
                [-2.0, -2.0, 2.0],
            ]
        ),
        atol=1e-6,
    )


def test_batched_helper_returns_stable_labels() -> None:
    rotations = torch.eye(3, dtype=torch.float32).expand(2, 3, 3).clone()
    translations = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    poses = PoseTW.from_Rt(rotations, translations)

    result = frusta_from_camera_tw(
        poses,
        _make_camera(),
        candidate_ids=[42, 43],
        ranks=[1, 0],
        oracle_rri=[0.125, -0.5],
        validity=[True, False],
    )

    assert result.labels == [
        "candidate_id=42 | rank=1 | oracle_rri=0.1250 | validity=valid",
        "candidate_id=43 | rank=0 | oracle_rri=-0.5000 | validity=invalid",
    ]
    assert len(result.line_strips) == 2
    np.testing.assert_allclose(result.centers_world, translations.numpy(), atol=1e-6)


def test_rerun_inspector_slice_has_no_data_handling_imports() -> None:
    source = Path(_frusta.__file__).read_text(encoding="utf-8") + Path(_colors.__file__).read_text(encoding="utf-8")
    assert "data_handling" not in source
