import torch
import trimesh

from oracle_rri.views.candidate_rendering import (
    CandidatePointCloudGenerator,
    CandidatePointCloudGeneratorConfig,
)


def _make_camera(w: int = 32, h: int = 32, device: str = "cpu"):
    """Minimal pinhole-ish camera for tests (approx Aria scale)."""

    from efm3d.aria import CameraTW, PoseTW

    params = torch.tensor([[700.0, 700.0, w / 2.0, h / 2.0]], device=device, dtype=torch.float32)
    width = torch.tensor([float(w)], device=device)
    height = torch.tensor([float(h)], device=device)
    return CameraTW.from_surreal(
        width=width,
        height=height,
        type_str="Pinhole",
        params=params,
        gain=torch.zeros(1, device=device),
        exposure_s=torch.zeros(1, device=device),
        valid_radius=torch.tensor([max(w, h)], device=device),
        T_camera_rig=PoseTW.from_matrix3x4(torch.eye(3, 4, device=device)),
    )


def _make_pose(device: str = "cpu"):
    from efm3d.aria import PoseTW

    return PoseTW.from_matrix3x4(torch.eye(3, 4, device=device))


def _make_box_mesh(z_center: float = 2.0, extent: float = 0.5) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(extent, extent, extent))
    mesh.apply_translation((0, 0, z_center))
    return mesh


def test_depth_hits_center_and_has_misses():
    device = "cpu"
    cam = _make_camera(32, 32, device=device)
    pose = _make_pose(device=device)
    mesh = _make_box_mesh(z_center=2.0, extent=0.5)

    cfg = CandidatePointCloudGeneratorConfig(max_depth=20.0, device=device, verbose=False)
    renderer = cfg.setup_target()

    depth = renderer.render_depth(pose_world_cam=pose, mesh=mesh, camera=cam)

    assert depth.shape == (32, 32)
    center_depth = depth[16, 16].item()
    assert 1.0 < center_depth < 3.0  # front face near 1.75
    assert torch.isfinite(depth).all()


def test_pointcloud_matches_valid_rays():
    device = "cpu"
    cam = _make_camera(16, 16, device=device)
    pose = _make_pose(device=device)
    mesh = _make_box_mesh(z_center=2.5, extent=0.4)

    cfg = CandidatePointCloudGeneratorConfig(max_depth=10.0, device=device, verbose=False)
    renderer = cfg.setup_target()

    depth, pts_world = renderer.render_point_cloud(pose_world_cam=pose, mesh=mesh, camera=cam)

    valid = (depth < cfg.max_depth).view(-1)
    assert pts_world.shape[0] == valid.sum().item()
    # points should cluster near z ~ front face distance
    mean_z = pts_world[:, 2].mean().item()
    assert 1.5 < mean_z < 4.0
