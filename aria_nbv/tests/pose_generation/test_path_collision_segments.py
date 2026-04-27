import torch
import trimesh
from efm3d.aria import CameraTW, PoseTW

from aria_nbv.pose_generation import CandidateViewGeneratorConfig, CollisionBackend
from aria_nbv.pose_generation.candidate_generation_rules import PathCollisionRule
from aria_nbv.pose_generation.types import CandidateContext


def _identity_pose(device: torch.device | str = "cpu") -> PoseTW:
    return PoseTW.from_matrix3x4(torch.eye(3, 4, device=device).unsqueeze(0))


def _dummy_camera(device: torch.device | str = "cpu") -> CameraTW:
    return CameraTW.from_surreal(
        width=torch.tensor([64.0], device=device),
        height=torch.tensor([64.0], device=device),
        type_str="Pinhole",
        params=torch.tensor([[60.0, 60.0, 32.0, 32.0]], device=device),
        gain=torch.zeros(1, device=device),
        exposure_s=torch.zeros(1, device=device),
        valid_radius=torch.tensor([64.0], device=device),
        T_camera_rig=_identity_pose(device=device),
    )


def _default_extent(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([-20.0, 20.0, -20.0, 20.0, -20.0, 20.0], dtype=torch.float32, device=device)


def _context(points: torch.Tensor, mesh: trimesh.Trimesh, cfg: CandidateViewGeneratorConfig) -> CandidateContext:
    poses = PoseTW.from_Rt(torch.eye(3, device=points.device).repeat(points.shape[0], 1, 1), points)
    return CandidateContext(
        cfg=cfg,
        reference_pose=_identity_pose(device=points.device),
        sampling_pose=_identity_pose(device=points.device),
        gt_mesh=mesh,
        mesh_verts=torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=points.device),
        mesh_faces=torch.from_numpy(mesh.faces).to(dtype=torch.int64, device=points.device),
        occupancy_extent=_default_extent(points.device),
        camera_calib_template=_dummy_camera(points.device),
        shell_poses=poses,
        centers_world=points,
        shell_offsets_ref=points,
        mask_valid=torch.ones(points.shape[0], dtype=torch.bool, device=points.device),
    )


def test_path_collision_rule_respects_segment_length_for_trimesh_backend() -> None:
    wall = trimesh.creation.box(extents=(0.1, 4.0, 4.0))
    wall.apply_translation([10.0, 0.0, 0.0])
    cfg = CandidateViewGeneratorConfig(
        ensure_collision_free=True,
        ensure_free_space=False,
        min_distance_to_mesh=0.0,
        collision_backend=CollisionBackend.TRIMESH,
        verbosity=0,
        is_debug=False,
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0],   # finite segment ends before the wall
            [12.0, 0.0, 0.0],  # finite segment crosses the wall
        ],
        dtype=torch.float32,
    )
    ctx = _context(targets, wall, cfg)

    PathCollisionRule(cfg)(ctx)

    assert torch.equal(ctx.mask_valid.cpu(), torch.tensor([True, False]))
