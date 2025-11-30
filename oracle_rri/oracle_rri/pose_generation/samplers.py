"""Direction and position sampling utilities for candidate generation."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Protocol

import torch
from power_spherical import HypersphericalUniform, PowerSpherical  # type: ignore[import-untyped]

from ..utils.frames import world_up_tensor
from .geometry import DEVICE_FWD
from .types import SamplingStrategy

if TYPE_CHECKING:
    from efm3d.aria.pose import PoseTW

    from .candidate_generation import CandidateViewGeneratorConfig


class DirectionSampler(Protocol):
    """Abstract base class for unit direction sampling in rig (LUF) frame."""

    def sample(self, cfg: CandidateViewGeneratorConfig, num: int, device: torch.device) -> torch.Tensor: ...


class UniformDirectionSampler(DirectionSampler):
    """Uniformly sample directions on S² in rig (LUF) frame."""

    def sample(self, cfg: CandidateViewGeneratorConfig, num: int, device: torch.device) -> torch.Tensor:
        dist = HypersphericalUniform(dim=3, device=device)
        dirs = dist.sample((num,))
        return dirs / dirs.norm(dim=-1, keepdim=True)


class ForwardPowerSphericalSampler(DirectionSampler):
    """Forward-biased PowerSpherical sampler in rig (LUF) frame."""

    def sample(self, cfg: CandidateViewGeneratorConfig, num: int, device: torch.device) -> torch.Tensor:
        mu = torch.tensor(DEVICE_FWD, device=device)
        dist = PowerSpherical(mu, torch.tensor(cfg.kappa, device=device))
        dirs = dist.sample((num,))
        return dirs / dirs.norm(dim=-1, keepdim=True)


_DIRECTION_SAMPLERS = {
    SamplingStrategy.UNIFORM_SPHERE: UniformDirectionSampler(),
    SamplingStrategy.FORWARD_POWERSPHERICAL: ForwardPowerSphericalSampler(),
}


class PositionSampler:
    """Sample candidate centers around a reference pose."""

    def __init__(self, cfg: CandidateViewGeneratorConfig):
        self.cfg = cfg

    def _filter_directions_world(self, dirs_world: torch.Tensor, last_forward_world: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        device = dirs_world.device
        wup = world_up_tensor(device=device, dtype=dirs_world.dtype)

        dot_up = (dirs_world * wup).sum(dim=-1)
        elev = torch.asin(dot_up.clamp(-1.0, 1.0))

        min_elev = torch.tensor(cfg.min_elev_rad, device=device, dtype=dirs_world.dtype)
        max_elev = torch.tensor(cfg.max_elev_rad, device=device, dtype=dirs_world.dtype)
        mask_elev = (elev >= min_elev) & (elev <= max_elev)

        def _project_horizontal(v: torch.Tensor) -> torch.Tensor:
            dot = (v * wup).sum(dim=-1, keepdim=True)
            v_h = v - dot * wup
            return v_h / v_h.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        dirs_h = _project_horizontal(dirs_world)
        fwd_h = _project_horizontal(last_forward_world.view(1, 3)).expand_as(dirs_h)

        cross = torch.cross(fwd_h, dirs_h, dim=-1)
        sin_yaw = (cross * wup).sum(dim=-1)
        cos_yaw = (dirs_h * fwd_h).sum(dim=-1)
        yaw = torch.atan2(sin_yaw, cos_yaw)

        if cfg.delta_azimuth_deg >= 360.0 - 1e-1:
            mask_yaw = torch.ones_like(mask_elev)
        else:
            half_delta = 0.5 * torch.tensor(cfg.delta_azimuth_rad, device=device, dtype=dirs_world.dtype)
            mask_yaw = (yaw >= -half_delta) & (yaw <= half_delta)

        return mask_elev & mask_yaw

    def sample(self, reference_pose: PoseTW) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw candidate centers and offsets in reference frame.

        Args:
            reference_pose: ``PoseTW`` reference2world pose used as sampling origin.

        Returns:
            Tuple of ``(centers_world, offsets_ref)`` where both are ``Tensor[N, 3]`` with
            ``N = cfg.num_samples``. Offsets are in the reference frame before rotation into world.
        """
        cfg = self.cfg
        device = cfg.device
        sampler = _DIRECTION_SAMPLERS[cfg.sampling_strategy]

        reference_pose_dev = reference_pose
        fwd_world = reference_pose_dev.rotate(torch.tensor(DEVICE_FWD, device=device).view(1, 3))[0]

        n_draw = ceil(cfg.num_samples * cfg.oversample_factor)
        dirs_rig = sampler.sample(cfg, n_draw, device=device)
        dirs_world = reference_pose_dev.rotate(dirs_rig)
        mask = self._filter_directions_world(dirs_world, fwd_world).view(-1)

        if not mask.any():
            raise RuntimeError("Directional sampling failed; relax elevation/azimuth constraints.")

        dirs_world = dirs_world[mask]
        offsets_rig = dirs_rig[mask]

        if dirs_world.shape[0] < cfg.num_samples:
            deficit = cfg.num_samples - dirs_world.shape[0]
            dirs_world = torch.cat([dirs_world, dirs_world[:deficit]], dim=0)
            offsets_rig = torch.cat([offsets_rig, offsets_rig[:deficit]], dim=0)

        dirs_world = dirs_world[: cfg.num_samples]
        offsets_rig = offsets_rig[: cfg.num_samples]

        radii = torch.empty(dirs_world.shape[0], device=device, dtype=dirs_world.dtype).uniform_(
            cfg.min_radius, cfg.max_radius
        )
        offsets_rig = offsets_rig * radii[:, None]
        centers_world = reference_pose_dev.transform(offsets_rig)
        return centers_world, offsets_rig


__all__ = [
    "DirectionSampler",
    "UniformDirectionSampler",
    "ForwardPowerSphericalSampler",
    "PositionSampler",
    "_DIRECTION_SAMPLERS",
]
