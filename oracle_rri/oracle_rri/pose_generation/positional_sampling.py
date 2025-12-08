"""Direction and position sampling utilities for candidate generation."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import torch
from power_spherical import HypersphericalUniform, PowerSpherical  # type: ignore[import-untyped]

from ..data.plotting import rotate_yaw_cw90
from .geometry import DEVICE_FWD
from .types import SamplingStrategy

if TYPE_CHECKING:
    from efm3d.aria.pose import PoseTW

    from .candidate_generation import CandidateViewGeneratorConfig


class PositionSampler:
    """Sample candidate centers around a reference pose."""

    def __init__(self, cfg: CandidateViewGeneratorConfig):
        self.cfg = cfg

    @staticmethod
    def _angles_from_dirs_rig(dirs_rig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (azimuth, elevation) for rig-frame unit vectors.

        Azimuth is measured around the rig up-axis (+Y) with 0 aligned to +Z (forward).
        Elevation is measured from the horizontal plane; +90° looks straight up, -90° down.
        """

        az = torch.atan2(dirs_rig[:, 0], dirs_rig[:, 2])  # x vs z in LUF convention
        elev = torch.atan2(dirs_rig[:, 1], torch.linalg.norm(dirs_rig[:, (0, 2)], dim=-1) + 1e-8)
        return az, elev

    def _scale_into_caps(self, dirs_rig: torch.Tensor) -> torch.Tensor:
        """Uniformly scale directions into az/elev caps without rejection.

        Strategy:
        - Azimuth: linear scale from [-pi, pi] to [-delta/2, delta/2] (wrap-friendly).
        - Elevation: linearly map y=sin(elev) from [-1, 1] to [sin(min), sin(max)];
          rescale xz-plane to keep unit norm. This preserves azimuth and avoids pile-up.
        """

        cfg = self.cfg
        device = dirs_rig.device
        dtype = dirs_rig.dtype

        x, y, z = dirs_rig.unbind(dim=-1)

        # Azimuth scaling (around +Y). Keep distribution uniform over the target band.
        if cfg.delta_azimuth_deg < 360.0 - 1e-3:
            az_raw = torch.atan2(x, z)  # [-pi, pi]
            scale_az = torch.tensor(cfg.delta_azimuth_rad, device=device, dtype=dtype) / (2 * torch.pi)
            az_scaled = az_raw * scale_az  # now in [-delta/2, delta/2]
            x = torch.sin(az_scaled)
            z = torch.cos(az_scaled)

        # Elevation scaling via y = sin(elev) interval mapping.
        y_min = torch.sin(torch.tensor(cfg.min_elev_rad, device=device, dtype=dtype))
        y_max = torch.sin(torch.tensor(cfg.max_elev_rad, device=device, dtype=dtype))
        # map [-1,1] -> [y_min, y_max]
        y_scaled = y_min + (y + 1.0) * 0.5 * (y_max - y_min)
        xz_norm = torch.linalg.norm(torch.stack([x, z], dim=-1), dim=-1).clamp_min(1e-8)
        xz_scale = torch.sqrt(torch.clamp(1.0 - y_scaled**2, min=0.0)) / xz_norm
        x = x * xz_scale
        z = z * xz_scale
        y = y_scaled

        dirs = torch.stack([x, y, z], dim=-1)
        return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def sample(self, reference_pose: PoseTW) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw candidate centers and offsets in reference frame.

        Args:
            reference_pose: ``PoseTW`` reference2world pose used as sampling origin.

        Returns:
            Tuple of ``(centers_world, offsets_ref)`` where both are ``Tensor[N, 3]`` with
            ``N = cfg.num_samples * cfg.oversample_factor``. Offsets are in the reference frame before rotation into world.
        """
        reference_pose_cor = rotate_yaw_cw90(reference_pose)

        n_draw = ceil(self.cfg.num_samples * self.cfg.oversample_factor)

        match self.cfg.sampling_strategy:
            case SamplingStrategy.UNIFORM_SPHERE:
                dirs = HypersphericalUniform(dim=3, device=self.cfg.device).sample((n_draw,))
            case SamplingStrategy.FORWARD_POWERSPHERICAL:
                mu = torch.tensor(DEVICE_FWD, device=self.cfg.device)
                dirs = PowerSpherical(mu, torch.tensor(self.cfg.kappa, device=self.cfg.device)).sample((n_draw,))
        dirs_rig = dirs / dirs.norm(dim=-1, keepdim=True)

        # Work entirely in reference (rig) frame for angle limits.
        dirs_rig = self._scale_into_caps(dirs_rig)

        dirs_world = reference_pose.rotate(dirs_rig)
        offsets_rig = dirs_rig

        radii = torch.empty(dirs_world.shape[0], device=self.cfg.device, dtype=dirs_world.dtype).uniform_(
            self.cfg.min_radius, self.cfg.max_radius
        )
        offsets_rig = offsets_rig * radii[:, None]
        centers_world = reference_pose_cor.transform(offsets_rig)
        return centers_world, offsets_rig


__all__ = [
    "PositionSampler",
]
