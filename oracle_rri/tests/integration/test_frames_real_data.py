import pytest
import torch

from oracle_rri.data import AseEfmDatasetConfig
from oracle_rri.utils.frames import view_axes_from_points, world_from_rig_camera_pose


def _first_sample():
    """Load a single real sample if available, otherwise skip the test."""
    cfg = AseEfmDatasetConfig(
        load_meshes=False,
        mesh_simplify_ratio=None,
        batch_size=1,
        snippet_length_s=0.5,
        is_debug=True,
        verbose=False,
    )
    try:
        dataset = cfg.setup_target()
        return next(iter(dataset))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"ASE dataset not available locally: {exc}")


def test_view_axes_from_points_luf_basis():
    """Synthetic sanity: view_axes returns LUF axes (left, up, forward)."""
    cam_pos = torch.tensor([[0.0, 0.0, 0.0]])
    look_at = torch.tensor([[0.0, 0.0, 1.0]])
    axes = view_axes_from_points(cam_pos, look_at)  # (1,3,3)
    left, up, fwd = axes[0, :, 0], axes[0, :, 1], axes[0, :, 2]
    assert torch.allclose(left, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(up, torch.tensor([0.0, 1.0, 0.0]), atol=1e-4)
    assert torch.allclose(fwd, torch.tensor([0.0, 0.0, 1.0]), atol=1e-4)
    # right-handed: left x up == forward
    cross = torch.cross(left, up, dim=0)
    assert torch.allclose(cross, fwd, atol=1e-4)


def test_world_from_rig_camera_pose_roundtrip_real():
    """Real data: rig→camera→rig roundtrip stays consistent."""
    sample = _first_sample()
    t_world_rig = sample.trajectory.t_world_rig[0]
    cam = sample.camera_rgb.calib
    t_world_cam = world_from_rig_camera_pose(t_world_rig, cam, frame_idx=0)
    recomposed = t_world_cam @ cam.T_camera_rig[0]
    assert torch.allclose(recomposed.matrix3x4, t_world_rig.matrix3x4, atol=1e-5)


def test_pose_transform_roundtrip_real_point():
    """Transform a point cam->world->cam and check roundtrip with real poses."""
    sample = _first_sample()
    t_world_rig = sample.trajectory.t_world_rig[0]
    cam = sample.camera_rgb.calib
    t_world_cam = world_from_rig_camera_pose(t_world_rig, cam, frame_idx=0)
    p_cam = torch.tensor([0.0, 0.0, 1.0])
    p_world = t_world_cam.transform(p_cam)
    p_cam_back = t_world_cam.inverse().transform(p_world)
    assert torch.allclose(p_cam_back, p_cam, atol=1e-6)
