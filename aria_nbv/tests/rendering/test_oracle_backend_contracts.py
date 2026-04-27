import pytest
import torch
import trimesh
from efm3d.aria import CameraTW, PoseTW

from aria_nbv.pose_generation.types import CandidateSamplingResult
from aria_nbv.rendering import CandidateDepthRendererConfig
from aria_nbv.utils.pytorch3d_compat import is_pytorch3d_available


def _require_mojo_backend() -> None:
    from aria_nbv.rendering.mojo_backend import is_mojo_available

    if not is_mojo_available():
        pytest.skip("Mojo rendering backend not available locally.")


def _make_camera(width: int = 32, height: int = 32) -> CameraTW:
    return CameraTW.from_surreal(
        width=torch.tensor([float(width)], dtype=torch.float32),
        height=torch.tensor([float(height)], dtype=torch.float32),
        type_str="Pinhole",
        params=torch.tensor([[40.0, 40.0, width / 2.0, height / 2.0]], dtype=torch.float32),
        gain=torch.zeros(1),
        exposure_s=torch.zeros(1),
        valid_radius=torch.tensor([float(max(width, height))], dtype=torch.float32),
        T_camera_rig=PoseTW.from_matrix3x4(torch.eye(3, 4).unsqueeze(0)),
    )


def _make_pose(z: float = 0.0) -> PoseTW:
    return PoseTW.from_Rt(torch.eye(3).unsqueeze(0), torch.tensor([[0.0, 0.0, z]], dtype=torch.float32))


def _make_mesh(z_center: float = 2.0, extent: float = 4.0) -> trimesh.Trimesh:
    verts = torch.tensor(
        [
            [-extent, -extent, z_center],
            [extent, -extent, z_center],
            [extent, extent, z_center],
            [-extent, extent, z_center],
        ],
        dtype=torch.float32,
    ).numpy()
    faces = torch.tensor([[0, 2, 1], [0, 3, 2]], dtype=torch.int64).numpy()
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _make_candidates(num: int = 1, z: float = 1.0) -> CandidateSamplingResult:
    cam_single = _make_camera()
    width = cam_single.size[0].item()
    height = cam_single.size[1].item()
    f_vals = cam_single.f.reshape(-1, 2)[0]
    c_vals = cam_single.c.reshape(-1, 2)[0]
    device = f_vals.device
    t_ref_cam = PoseTW.from_Rt(
        torch.eye(3, device=device).unsqueeze(0).repeat(num, 1, 1),
        torch.tensor([[0.0, 0.0, z]], device=device).repeat(num, 1),
    )
    cams = CameraTW.from_parameters(
        width=torch.full((num, 1), float(width), device=device),
        height=torch.full((num, 1), float(height), device=device),
        fx=torch.full((num, 1), f_vals[0].item(), device=device),
        fy=torch.full((num, 1), f_vals[1].item(), device=device),
        cx=torch.full((num, 1), c_vals[0].item(), device=device),
        cy=torch.full((num, 1), c_vals[1].item(), device=device),
        gain=torch.zeros((num, 1), device=device),
        exposure_s=torch.zeros((num, 1), device=device),
        valid_radiusx=torch.full((num, 1), max(width, height), device=device),
        valid_radiusy=torch.full((num, 1), max(width, height), device=device),
        T_camera_rig=t_ref_cam,
        dist_params=cam_single.dist.expand(num, -1),
    )
    return CandidateSamplingResult(
        views=cams,
        reference_pose=_make_pose(),
        mask_valid=torch.ones(num, dtype=torch.bool),
        masks={},
        shell_poses=t_ref_cam,
    )


def test_depth_renderer_backend_contract() -> None:
    from aria_nbv.rendering.candidate_depth_renderer import DepthRendererBackend

    assert DepthRendererBackend.PYTORCH3D.value == "pytorch3d"
    assert DepthRendererBackend.MOJO.value == "mojo"


def test_pointcloud_backend_contract() -> None:
    from aria_nbv.rendering.candidate_pointclouds import CandidatePointCloudBuilderConfig, PointCloudBackend

    cfg = CandidatePointCloudBuilderConfig()
    assert PointCloudBackend.PYTORCH3D.value == "pytorch3d"
    assert PointCloudBackend.MOJO.value == "mojo"
    assert cfg.backend == PointCloudBackend.PYTORCH3D
    assert cfg.backprojection_stride == 1


def test_candidate_depth_renderer_config_defaults_to_pytorch3d() -> None:
    from aria_nbv.rendering.candidate_depth_renderer import DepthRendererBackend

    cfg = CandidateDepthRendererConfig()
    assert cfg.backend == DepthRendererBackend.PYTORCH3D
    assert cfg.pytorch3d is not None
    assert cfg.mojo is not None


def test_candidate_depth_renderer_legacy_renderer_alias_does_not_recurse() -> None:
    from aria_nbv.rendering import Pytorch3DDepthRendererConfig

    renderer_cfg = Pytorch3DDepthRendererConfig(device="cpu", is_debug=True)
    cfg = CandidateDepthRendererConfig(renderer=renderer_cfg)

    assert cfg.renderer is None
    assert cfg.pytorch3d is renderer_cfg
    assert cfg.pytorch3d.device == torch.device("cpu")
    cfg.max_candidates_final = 4
    assert cfg.max_candidates_final == 4


def test_candidate_depths_allow_optional_backend_state() -> None:
    from aria_nbv.rendering.candidate_depth_renderer import CandidateDepths

    batch = CandidateDepths(
        depths=torch.ones((1, 4, 4), dtype=torch.float32),
        depths_valid_mask=torch.ones((1, 4, 4), dtype=torch.bool),
        poses=_make_pose(),
        reference_pose=_make_pose(),
        candidate_indices=torch.tensor([0], dtype=torch.long),
        camera=_make_camera(),
        p3d_cameras=None,
    )
    assert batch.p3d_cameras is None


def test_mojo_depth_renderer_matches_pytorch3d_on_plane() -> None:
    _require_mojo_backend()
    if not is_pytorch3d_available():
        pytest.skip("PyTorch3D is required for Mojo/PyTorch3D renderer parity.")

    from aria_nbv.data_handling.efm_views import EfmSnippetView
    from aria_nbv.rendering.candidate_depth_renderer import DepthRendererBackend

    mesh = _make_mesh()
    mesh_verts = torch.from_numpy(mesh.vertices).to(dtype=torch.float32)
    mesh_faces = torch.from_numpy(mesh.faces).to(dtype=torch.int64)
    camera = _make_camera()
    sample = EfmSnippetView(
        efm={
            "rgb/img": torch.zeros(1, 3, 32, 32),
            "rgb/calib": camera,
            "pose/t_world_rig": _make_pose(),
            "points/p3s_world": torch.zeros(1, 1, 3),
            "points/dist_std": torch.zeros(1, 1),
            "points/inv_dist_std": torch.zeros(1, 1),
            "points/time_ns": torch.zeros(1, dtype=torch.int64),
            "frame_id": torch.zeros(1, dtype=torch.int64),
            "img/time_ns": torch.zeros(1, dtype=torch.int64),
            "pose/time_ns": torch.zeros(1, dtype=torch.int64),
            "points/vol_min": torch.tensor([-1.0, -1.0, -1.0]),
            "points/vol_max": torch.tensor([1.0, 1.0, 1.0]),
            "pose/gravity_in_world": torch.tensor([0.0, 0.0, -9.81]),
        },
        scene_id="render_scene",
        snippet_id="render_snippet",
        mesh=mesh,
        mesh_verts=mesh_verts,
        mesh_faces=mesh_faces,
    )
    candidates = _make_candidates(num=1, z=1.0)

    baseline_cfg = CandidateDepthRendererConfig()
    mojo_cfg = baseline_cfg.model_copy(update={"backend": DepthRendererBackend.MOJO})
    baseline = baseline_cfg.setup_target().render(sample=sample, candidates=candidates)
    mojo = mojo_cfg.setup_target().render(sample=sample, candidates=candidates)

    assert torch.equal(baseline.depths_valid_mask, mojo.depths_valid_mask)
    assert torch.allclose(baseline.depths, mojo.depths, atol=1e-4, rtol=1e-4)
    assert torch.equal(baseline.candidate_indices, mojo.candidate_indices)
