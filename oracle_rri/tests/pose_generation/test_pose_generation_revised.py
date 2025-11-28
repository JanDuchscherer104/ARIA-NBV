import math

import pytest

pytest.importorskip("efm3d")

import torch
import trimesh
from efm3d.aria import PoseTW

from oracle_rri.pose_generation import (
    CandidateViewGenerator,
    CandidateViewGeneratorConfig,
    SamplingStrategy,
)
from oracle_rri.pose_generation.reference_power_spherical_distributions import (
    HypersphericalUniform,
    PowerSpherical,
)
from oracle_rri.utils.frames import view_axes_from_poses, world_up_tensor


def _id_pose(device: str | torch.device = "cpu") -> PoseTW:
    return PoseTW.from_Rt(torch.eye(3, device=device).unsqueeze(0), torch.zeros(1, 3, device=device))


def _yaw_angles(directions: torch.Tensor, forward: torch.Tensor, world_up: torch.Tensor) -> torch.Tensor:
    dirs_h = directions - (directions @ world_up)[:, None] * world_up[None, :]
    fwd_h = forward - (forward @ world_up) * world_up
    dirs_h = dirs_h / dirs_h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    fwd_h = fwd_h / fwd_h.norm().clamp_min(1e-6)
    cross = torch.cross(fwd_h.expand_as(dirs_h), dirs_h, dim=-1)
    sin_yaw = (cross * world_up).sum(dim=-1)
    cos_yaw = (dirs_h * fwd_h).sum(dim=-1)
    return torch.atan2(sin_yaw, cos_yaw)


def test_view_axes_from_poses_zero_roll_and_forward():
    last = _id_pose()
    targets = PoseTW.from_Rt(torch.eye(3).repeat(2, 1, 1), torch.tensor([[0.5, 0.0, 1.0], [0.0, 0.5, 1.0]]))

    poses = view_axes_from_poses(last, targets, look_away=True)

    dirs = targets.t - last.t
    dirs = torch.nn.functional.normalize(dirs, dim=1)
    forward = poses.R[:, :, 2]
    cos_sim = (forward * dirs).sum(dim=1)
    assert torch.allclose(cos_sim, torch.ones_like(cos_sim), atol=1e-5)

    world_up = world_up_tensor(device=dirs.device, dtype=dirs.dtype)
    x_axis = poses.R[:, :, 0]
    roll = (x_axis * world_up).sum(dim=1).abs()
    assert torch.all(roll < 1e-6)


def test_delta_azimuth_zero_yields_planar_yaw():
    torch.manual_seed(0)
    cfg = CandidateViewGeneratorConfig(
        num_samples=64,
        min_radius=0.5,
        max_radius=0.5,
        min_elev_deg=-5,
        max_elev_deg=5,
        delta_azimuth_deg=0.0,
        ensure_collision_free=False,
        ensure_free_space=False,
        min_distance_to_mesh=0.0,
        sampling_strategy=SamplingStrategy.SHELL_UNIFORM,
        oversample_factor=4.0,
        max_resamples=5,
        is_debug=True,
    )
    gen = CandidateViewGenerator(cfg)
    res = gen.generate(last_pose=_id_pose())

    dirs_world = torch.nn.functional.normalize(res.shell_poses.t, dim=1)
    world_up = world_up_tensor(device=dirs_world.device, dtype=dirs_world.dtype)
    forward = torch.tensor([0.0, 0.0, 1.0], device=dirs_world.device, dtype=dirs_world.dtype)
    yaw = _yaw_angles(dirs_world, forward, world_up)
    assert torch.all(yaw.abs() < math.radians(1.0))


def test_power_spherical_biases_forward_vs_uniform():
    torch.manual_seed(1)
    base_cfg = {
        "num_samples": 2000,
        "min_radius": 1.0,
        "max_radius": 1.0,
        "min_elev_deg": -45,
        "max_elev_deg": 45,
        "ensure_collision_free": False,
        "ensure_free_space": False,
        "min_distance_to_mesh": 0.0,
        "oversample_factor": 2.0,
        "max_resamples": 3,
        "delta_azimuth_deg": 360.0,
        "is_debug": True,
    }

    cfg_uniform = CandidateViewGeneratorConfig(sampling_strategy=SamplingStrategy.SHELL_UNIFORM, **base_cfg)
    cfg_power = CandidateViewGeneratorConfig(sampling_strategy=SamplingStrategy.FORWARD_GAUSSIAN, kappa=8.0, **base_cfg)

    mean_dot = []
    for cfg in (cfg_uniform, cfg_power):
        gen = CandidateViewGenerator(cfg)
        res = gen.generate(last_pose=_id_pose())
        dirs_world = torch.nn.functional.normalize(res.shell_poses.t, dim=1)
        forward = torch.tensor([0.0, 0.0, 1.0], device=dirs_world.device)
        mean_dot.append((dirs_world @ forward).mean().item())

    assert mean_dot[1] > mean_dot[0] + 0.15


def test_min_distance_rule_emits_debug_and_masks():
    mesh = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    cfg = CandidateViewGeneratorConfig(
        num_samples=16,
        min_radius=0.6,
        max_radius=0.6,
        ensure_collision_free=False,
        ensure_free_space=False,
        min_distance_to_mesh=0.3,
        collect_rule_masks=True,
        collect_debug_stats=True,
        is_debug=True,
    )
    gen = CandidateViewGenerator(cfg)
    res = gen.generate(last_pose=_id_pose(), gt_mesh=mesh)

    assert "MinDistanceToMeshRule" in res.masks
    assert "min_distance_to_mesh" in res.extras
    # At least one candidate should be rejected as too close to the mesh center.
    assert (~res.mask_valid).any()


def test_power_spherical_wrapper_device_dtype():
    mu = torch.tensor([0.0, 0.0, 1.0])
    dist = PowerSpherical(mu=mu, kappa=5.0, device="cpu", dtype=torch.float64)
    sample = dist.sample((2,))
    assert sample.device.type == "cpu"
    assert sample.dtype == torch.float64

    uni = HypersphericalUniform(dim=3, device="cpu")
    s = uni.sample((1,))
    assert s.shape == (1, 3)
