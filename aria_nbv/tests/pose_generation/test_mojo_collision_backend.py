import pytest
import torch
import trimesh
from efm3d.aria import CameraTW, PoseTW

from aria_nbv.pose_generation import CandidateViewGeneratorConfig, CollisionBackend
from aria_nbv.pose_generation.candidate_generation_rules import MinDistanceToMeshRule, PathCollisionRule
from aria_nbv.pose_generation.types import CandidateContext


def _require_mojo_backend() -> None:
    from aria_nbv.pose_generation.mojo_backend import is_mojo_available

    if not is_mojo_available():
        pytest.skip("Mojo collision backend not available locally.")


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
    return torch.tensor([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=torch.float32, device=device)


def _mesh_triplet(device: torch.device | str = "cpu") -> tuple[trimesh.Trimesh, torch.Tensor, torch.Tensor]:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    verts = torch.from_numpy(mesh.vertices).to(dtype=torch.float32, device=device)
    faces = torch.from_numpy(mesh.faces).to(dtype=torch.int64, device=device)
    return mesh, verts, faces


def _context_from_points(
    *,
    cfg: CandidateViewGeneratorConfig,
    mesh: trimesh.Trimesh,
    verts: torch.Tensor,
    faces: torch.Tensor,
    points: torch.Tensor,
) -> CandidateContext:
    poses = PoseTW.from_Rt(torch.eye(3, device=points.device).repeat(points.shape[0], 1, 1), points)
    return CandidateContext(
        cfg=cfg,
        reference_pose=_identity_pose(device=points.device),
        sampling_pose=_identity_pose(device=points.device),
        gt_mesh=mesh,
        mesh_verts=verts,
        mesh_faces=faces,
        occupancy_extent=_default_extent(points.device),
        camera_calib_template=_dummy_camera(points.device),
        shell_poses=poses,
        centers_world=points,
        shell_offsets_ref=points,
        mask_valid=torch.ones(points.shape[0], dtype=torch.bool, device=points.device),
    )


def test_collision_backend_mojo_enum() -> None:
    assert CollisionBackend.MOJO.value == "mojo"


def test_min_distance_rule_mojo_matches_trimesh() -> None:
    _require_mojo_backend()

    mesh, verts, faces = _mesh_triplet()
    points = torch.tensor(
        [
            [0.8, 0.0, 0.0],
            [0.0, 0.0, 1.2],
            [0.6, 0.6, 0.6],
        ],
        dtype=torch.float32,
    )
    cfg_trimesh = CandidateViewGeneratorConfig(
        min_distance_to_mesh=0.25,
        ensure_collision_free=False,
        ensure_free_space=False,
        collision_backend=CollisionBackend.TRIMESH,
        collect_debug_stats=True,
        verbosity=0,
        is_debug=False,
    )
    cfg_mojo = cfg_trimesh.model_copy(update={"collision_backend": CollisionBackend.MOJO})

    ctx_trimesh = _context_from_points(cfg=cfg_trimesh, mesh=mesh, verts=verts, faces=faces, points=points)
    ctx_mojo = _context_from_points(cfg=cfg_mojo, mesh=mesh, verts=verts, faces=faces, points=points)

    MinDistanceToMeshRule(cfg_trimesh)(ctx_trimesh)
    MinDistanceToMeshRule(cfg_mojo)(ctx_mojo)

    assert torch.equal(ctx_trimesh.mask_valid, ctx_mojo.mask_valid)
    assert "min_distance_to_mesh" in ctx_trimesh.debug
    assert "min_distance_to_mesh" in ctx_mojo.debug
    assert torch.allclose(
        ctx_trimesh.debug["min_distance_to_mesh"],
        ctx_mojo.debug["min_distance_to_mesh"],
        atol=1e-4,
        rtol=1e-4,
    )


def test_path_collision_rule_mojo_matches_trimesh() -> None:
    _require_mojo_backend()

    mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    mesh.apply_translation([0.5, 0.0, 0.0])
    verts = torch.from_numpy(mesh.vertices).to(dtype=torch.float32)
    faces = torch.from_numpy(mesh.faces).to(dtype=torch.int64)
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

    cfg_trimesh = CandidateViewGeneratorConfig(
        min_distance_to_mesh=0.0,
        ensure_collision_free=True,
        ensure_free_space=False,
        collision_backend=CollisionBackend.TRIMESH,
        collect_debug_stats=True,
        verbosity=0,
        is_debug=False,
    )
    cfg_mojo = cfg_trimesh.model_copy(update={"collision_backend": CollisionBackend.MOJO})

    ctx_trimesh = _context_from_points(cfg=cfg_trimesh, mesh=mesh, verts=verts, faces=faces, points=points)
    ctx_mojo = _context_from_points(cfg=cfg_mojo, mesh=mesh, verts=verts, faces=faces, points=points)

    PathCollisionRule(cfg_trimesh)(ctx_trimesh)
    PathCollisionRule(cfg_mojo)(ctx_mojo)

    assert torch.equal(ctx_trimesh.mask_valid, ctx_mojo.mask_valid)
    assert "path_collision_mask" in ctx_trimesh.debug
    assert "path_collision_mask" in ctx_mojo.debug
    assert torch.equal(ctx_trimesh.debug["path_collision_mask"], ctx_mojo.debug["path_collision_mask"])


def test_mojo_collision_runs_in_subprocess_in_unsupported_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    from aria_nbv.pose_generation import candidate_generation_rules as rules_mod

    monkeypatch.setattr(rules_mod, "is_mojo_thread_context_supported", lambda: False)

    mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    mesh.apply_translation([0.5, 0.0, 0.0])
    verts = torch.from_numpy(mesh.vertices).to(dtype=torch.float32)
    faces = torch.from_numpy(mesh.faces).to(dtype=torch.int64)
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

    cfg_mojo = CandidateViewGeneratorConfig(
        min_distance_to_mesh=0.0,
        ensure_collision_free=True,
        ensure_free_space=False,
        collision_backend=CollisionBackend.MOJO,
        collect_debug_stats=True,
        verbosity=0,
        is_debug=False,
    )
    ctx = _context_from_points(cfg=cfg_mojo, mesh=mesh, verts=verts, faces=faces, points=points)

    PathCollisionRule(cfg_mojo)(ctx)

    assert torch.equal(ctx.mask_valid, torch.tensor([False, True]))
    assert "path_collision_mask" in ctx.debug
