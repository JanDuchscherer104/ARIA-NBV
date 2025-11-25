import math

import numpy as np
import pytest

pytest.importorskip("efm3d")

import torch
import trimesh
from efm3d.aria import PoseTW
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from oracle_rri.data import AseEfmDatasetConfig
from oracle_rri.pose_generation import (
    CandidateViewGenerator,
    CandidateViewGeneratorConfig,
    CollisionBackend,
    SamplingStrategy,
)
from oracle_rri.pose_generation.candidate_generation_rules import (
    FreeSpaceRule,
    MinDistanceToMeshRule,
    PathCollisionRule,
    ShellSamplingRule,
)
from oracle_rri.pose_generation.types import CandidateContext


def _identity_pose(batch: int = 1, device: torch.device | str = "cpu") -> PoseTW:
    """Convenience: identity PoseTW repeated `batch` times."""

    data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device)
    if batch > 1:
        data = data.repeat(batch, 1)
    return PoseTW(data)


# --------------------------------------------------------------------------- unit tests (rule logic)
def test_shell_sampling_rule_orients_toward_last_pose():
    """Shell sampling should place cameras on a shell and orient them to look at the last pose."""

    cfg = CandidateViewGeneratorConfig(
        num_samples=8,
        min_radius=0.5,
        max_radius=0.8,
        min_distance_to_mesh=0.0,
        ensure_collision_free=False,
        ensure_free_space=False,
        sampling_strategy=SamplingStrategy.SHELL_UNIFORM,
        verbose=False,
        is_debug=True,  # force CPU
    )
    rule = ShellSamplingRule(cfg)
    last_pose = _identity_pose()
    ctx: CandidateContext = {
        "last_pose": last_pose,
        "gt_mesh": None,
        "occupancy_extent": None,
        "device": torch.device("cpu"),
        "poses": torch.empty(cfg.num_samples, 12),
        "mask": torch.ones(cfg.num_samples, dtype=torch.bool),
    }

    out = rule(ctx)
    poses: PoseTW = out["poses"]
    translations = poses.t  # (N,3)
    rotations = poses.R  # (N,3,3)

    # All candidates should lie within the spherical shell radii.
    radii = torch.linalg.norm(translations, dim=1)
    assert torch.all(radii >= cfg.min_radius - 1e-4)
    assert torch.all(radii <= cfg.max_radius + 1e-4)

    # Forward axis (z) should point back to the last pose (origin).
    forward = rotations[:, :, 2]  # (N,3)
    to_origin = torch.nn.functional.normalize(-translations, dim=1)
    cos_sim = (forward * to_origin).sum(dim=1)
    assert torch.allclose(cos_sim, torch.ones_like(cos_sim), atol=1e-3)

    # Rotation matrices should be proper (determinant ~ +1).
    dets = torch.linalg.det(rotations)
    assert torch.allclose(dets, torch.ones_like(dets), atol=1e-3)


def test_shell_sampling_uniform_area():
    """Elevation sampling for SHELL_UNIFORM should be area-uniform on the cap."""

    cfg = CandidateViewGeneratorConfig(
        num_samples=4096,
        min_radius=1.0,
        max_radius=1.0,
        min_elev_deg=-30,
        max_elev_deg=30,
        sampling_strategy=SamplingStrategy.SHELL_UNIFORM,
        ensure_collision_free=False,
        ensure_free_space=False,
        verbose=False,
        is_debug=True,
    )
    rule = ShellSamplingRule(cfg)
    dev = torch.device("cpu")
    az = torch.rand(cfg.num_samples, device=dev) * 2 * torch.pi
    dirs = rule._sample_directions(cfg.num_samples, az, dev)
    elev = torch.asin(dirs[:, 1])  # y = sin(elev)

    sin_min = math.sin(math.radians(cfg.min_elev_deg))
    sin_max = math.sin(math.radians(cfg.max_elev_deg))
    # For uniform sin(elev), expected mean is midpoint of interval
    expected_mean = 0.5 * (sin_min + sin_max)
    empirical = torch.sin(elev).mean().item()
    assert abs(empirical - expected_mean) < 0.02  # loose tolerance for MC noise


def test_min_distance_rule_rejects_near_mesh(monkeypatch):
    """Candidates inside or too near the mesh should be culled.

    We mock `ProximityQuery.signed_distance` to avoid optional rtree dependency
    and to make the expected sign/scale deterministic.
    """

    mesh = trimesh.creation.box(extents=(0.4, 0.4, 0.4))  # centered at origin
    cfg = CandidateViewGeneratorConfig(
        min_distance_to_mesh=0.25,  # larger than cube half-extent (0.2) so centre is rejected
        min_radius=0.1,
        max_radius=0.1,
        ensure_collision_free=False,
        ensure_free_space=False,
        verbose=False,
        is_debug=True,
    )
    rule = MinDistanceToMeshRule(cfg)

    # Mock distances: candidate 0 at distance 0.1, candidate 1 at distance 0.5
    monkeypatch.setattr(
        trimesh.proximity.ProximityQuery,
        "signed_distance",
        lambda self, pts: np.array([0.1, 0.5]),
        raising=False,
    )

    # Two candidates: one inside the cube, one outside.
    poses = PoseTW.from_Rt(
        torch.eye(3).repeat(2, 1, 1),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    ctx: CandidateContext = {
        "last_pose": _identity_pose(),
        "gt_mesh": mesh,
        "occupancy_extent": None,
        "device": torch.device("cpu"),
        "poses": poses,
        "mask": torch.ones(2, dtype=torch.bool),
    }

    out = rule(ctx)
    # Expect: first candidate (inside cube) rejected, second kept.
    assert torch.equal(out["mask"], torch.tensor([False, True]))


def test_path_collision_rule_blocks_intersecting_ray():
    """A candidate whose straight-line path crosses the mesh should be rejected."""

    mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    mesh.apply_translation([0.5, 0.0, 0.0])  # box in the path from origin to (1,0,0)

    cfg = CandidateViewGeneratorConfig(
        ensure_free_space=False,
        min_distance_to_mesh=0.0,
        collision_backend=CollisionBackend.TRIMESH,
        verbose=False,
        is_debug=True,
    )
    rule = PathCollisionRule(cfg)

    poses = PoseTW.from_Rt(
        torch.eye(3).repeat(2, 1, 1),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    ctx: CandidateContext = {
        "last_pose": _identity_pose(),
        "gt_mesh": mesh,
        "occupancy_extent": None,
        "device": torch.device("cpu"),
        "poses": poses,
        "mask": torch.ones(2, dtype=torch.bool),
    }

    out = rule(ctx)
    # First ray hits the box; second (to y-axis) is free.
    assert torch.equal(out["mask"], torch.tensor([False, True]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for P3D backend")
def test_path_collision_rule_gpu_matches_cpu():
    """GPU P3D backend should match CPU TRIMESH behaviour on a simple scene."""

    mesh = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    mesh.apply_translation([0.5, 0.0, 0.0])

    poses = PoseTW.from_Rt(
        torch.eye(3).repeat(2, 1, 1),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    last_pose_cpu = _identity_pose()
    last_pose_gpu = _identity_pose(device="cuda")

    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).long().cuda()
    mesh_p3d = Meshes(verts=[verts], faces=[faces])
    mesh_samples = sample_points_from_meshes(mesh_p3d, 4000)

    cfg_gpu = CandidateViewGeneratorConfig(
        ensure_free_space=False,
        min_distance_to_mesh=0.0,
        collision_backend=CollisionBackend.P3D,
        ray_subsample=16,
        step_clearance=0.02,
        verbose=False,
        is_debug=True,
    )
    cfg_cpu = CandidateViewGeneratorConfig(
        ensure_free_space=False,
        min_distance_to_mesh=0.0,
        collision_backend=CollisionBackend.TRIMESH,
        verbose=False,
        is_debug=True,
    )
    ctx_gpu: CandidateContext = {
        "last_pose": last_pose_gpu,
        "gt_mesh": mesh,
        "mesh_p3d": mesh_p3d,
        "mesh_samples": mesh_samples,
        "mesh_verts": verts,
        "mesh_faces": faces,
        "occupancy_extent": None,
        "device": torch.device("cuda"),
        "poses": poses.to("cuda"),
        "mask": torch.ones(2, dtype=torch.bool, device="cuda"),
    }
    ctx_cpu: CandidateContext = {
        "last_pose": last_pose_cpu,
        "gt_mesh": mesh,
        "mesh_p3d": None,
        "mesh_samples": None,
        "mesh_verts": None,
        "mesh_faces": None,
        "occupancy_extent": None,
        "device": torch.device("cpu"),
        "poses": poses,
        "mask": torch.ones(2, dtype=torch.bool),
    }

    mask_cpu = PathCollisionRule(cfg_cpu)(ctx_cpu)["mask"]
    mask_gpu = PathCollisionRule(cfg_gpu)(ctx_gpu)["mask"].cpu()
    assert torch.equal(mask_cpu, mask_gpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for P3D backend")
def test_min_distance_rule_gpu_p3d():
    """P3D backend for min-distance should reject near-mesh candidates."""

    mesh = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    cfg_gpu = CandidateViewGeneratorConfig(
        min_distance_to_mesh=0.25,
        min_radius=0.1,
        max_radius=0.1,
        ensure_collision_free=False,
        ensure_free_space=False,
        collision_backend=CollisionBackend.P3D,
        mesh_samples=3000,
        verbose=False,
        is_debug=True,
    )
    rule_gpu = MinDistanceToMeshRule(cfg_gpu)

    poses = PoseTW.from_Rt(
        torch.eye(3).repeat(2, 1, 1),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )

    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).long().cuda()
    mesh_p3d = Meshes(verts=[verts], faces=[faces])
    mesh_samples = sample_points_from_meshes(mesh_p3d, cfg_gpu.mesh_samples)

    ctx_gpu: CandidateContext = {
        "last_pose": _identity_pose(device="cuda"),
        "gt_mesh": mesh,
        "mesh_p3d": mesh_p3d,
        "mesh_samples": mesh_samples,
        "mesh_verts": verts,
        "mesh_faces": faces,
        "occupancy_extent": None,
        "device": torch.device("cuda"),
        "poses": poses.to("cuda"),
        "mask": torch.ones(2, dtype=torch.bool, device="cuda"),
    }

    out = rule_gpu(ctx_gpu)
    assert torch.equal(out["mask"].cpu(), torch.tensor([False, True]))


def test_free_space_rule_bounds_candidates():
    """AABB extent should clip candidates outside the allowed volume."""

    cfg = CandidateViewGeneratorConfig(
        ensure_collision_free=False,
        min_distance_to_mesh=0.0,
        ensure_free_space=True,
        verbose=False,
        is_debug=True,
    )
    rule = FreeSpaceRule(cfg)
    extent = torch.tensor([-1.0, 1.0, -1.0, 1.0, -0.5, 0.5])  # xmin,xmax,ymin,ymax,zmin,zmax
    poses = PoseTW.from_Rt(
        torch.eye(3).repeat(3, 1, 1),
        torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.6]]),
    )
    ctx: CandidateContext = {
        "last_pose": _identity_pose(),
        "gt_mesh": None,
        "occupancy_extent": extent,
        "device": torch.device("cpu"),
        "poses": poses,
        "mask": torch.ones(3, dtype=torch.bool),
    }

    out = rule(ctx)
    # Only the first pose lies inside the AABB.
    assert torch.equal(out["mask"], torch.tensor([True, False, False]))


# --------------------------------------------------------------------------- integration tests (generator + optional real data)
def test_candidate_generator_pipeline_synthetic():
    """End-to-end generation on synthetic scene without mesh/free-space constraints."""

    cfg = CandidateViewGeneratorConfig(
        num_samples=16,
        min_radius=0.4,
        max_radius=0.6,
        ensure_collision_free=False,
        ensure_free_space=False,
        min_distance_to_mesh=0.0,
        verbose=False,
        is_debug=True,
    )
    gen = CandidateViewGenerator(cfg)
    result = gen.generate(last_pose=_identity_pose())

    assert result["poses"]._data.shape[0] == cfg.num_samples
    assert result["mask_valid"].shape[0] == cfg.num_samples
    # All candidates should pass because all rules are disabled.
    assert torch.all(result["mask_valid"])


@pytest.mark.slow
def test_candidate_generator_with_real_dataset_sample():
    """Integration with ASEDataset if shards/meshes are available locally.

    Skips gracefully when dataset shards or meshes are not present.
    """

    try:
        dataset = AseEfmDatasetConfig(load_meshes=True, verbose=False).setup_target()
        sample = next(iter(dataset))
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"ASE dataset not available locally: {exc}")

    cfg = CandidateViewGeneratorConfig(
        num_samples=32,
        min_radius=0.5,
        max_radius=1.0,
        ensure_collision_free=False,  # relax pruning to ensure some survive
        ensure_free_space=False,
        min_distance_to_mesh=0.0,
        sampling_strategy=SamplingStrategy.SHELL_UNIFORM,
        verbose=False,
        is_debug=True,
    )
    gen = CandidateViewGenerator(cfg)
    result = gen.generate_from_typed_sample(sample)

    # At least one valid candidate should remain after pruning.
    assert result["poses"]._data.shape[0] == cfg.num_samples
    assert result["mask_valid"].any()


@pytest.mark.slow
def test_candidate_generator_real_orientation_and_extent():
    """Real-data check: candidates look back at last_pose and lie inside inferred extent."""

    try:
        dataset = AseEfmDatasetConfig(load_meshes=True, verbose=False, require_mesh=False).setup_target()
        sample = next(iter(dataset))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"ASE dataset not available locally: {exc}")

    cfg = CandidateViewGeneratorConfig(
        num_samples=16,
        min_radius=0.5,
        max_radius=1.0,
        ensure_collision_free=False,
        ensure_free_space=True,
        min_distance_to_mesh=0.0,
        verbose=False,
        is_debug=True,
    )
    gen = CandidateViewGenerator(cfg)
    result = gen.generate_from_typed_sample(sample)

    poses = result["poses"]
    mask = result["mask_valid"]
    assert mask.any(), "All candidates were filtered out unexpectedly."

    # Check forward axis points toward last_pose for valid candidates.
    pos_world = poses.t[mask]  # (M,3)
    r_mat = poses.R[mask]  # (M,3,3)
    forward = r_mat[:, :, 2]
    to_last = torch.nn.functional.normalize(sample.trajectory.final_pose.t - pos_world, dim=1)
    cos_sim = (forward * to_last).sum(dim=1)
    assert torch.all(cos_sim > 0.9)

    # Check positions fall inside the occupancy extent used (if any).
    extent = None
    mesh = sample.mesh
    if mesh is not None:
        bounds = torch.from_numpy(mesh.bounds).float()
        vmin, vmax = bounds[0], bounds[1]
        extent = torch.stack([vmin[0], vmax[0], vmin[1], vmax[1], vmin[2], vmax[2]])
    if extent is None:
        extent = gen._occupancy_extent_from_sample(sample)

    if extent is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = extent
        inside = (
            (pos_world[:, 0] >= xmin)
            & (pos_world[:, 0] <= xmax)
            & (pos_world[:, 1] >= ymin)
            & (pos_world[:, 1] <= ymax)
            & (pos_world[:, 2] >= zmin)
            & (pos_world[:, 2] <= zmax)
        )
        assert torch.all(inside), "Some candidates fall outside inferred extent."
