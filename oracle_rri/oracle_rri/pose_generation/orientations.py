"""Orientation builder for candidate poses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from efm3d.aria.pose import PoseTW
from power_spherical import HypersphericalUniform, PowerSpherical

from ..utils.frames import view_axes_from_poses, world_up_tensor
from .geometry import DEVICE_FWD
from .types import SamplingStrategy, ViewDirectionMode

if TYPE_CHECKING:
    from .candidate_generation import CandidateViewGeneratorConfig


class OrientationBuilder:
    """Construct candidate camera orientations from centers and view settings."""

    def __init__(self, cfg: CandidateViewGeneratorConfig):
        self.cfg = cfg

    def _sample_view_dirs_cam(self, num: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample camera-forward directions in the base camera frame.

        Args:
            num: Number of directions to draw.
            device: Torch device for samples.
            dtype: Torch dtype for samples.

        Returns:
            `Tensor[num, 3]` unit directions in camera frame respecting optional caps and concentration.
        """
        cfg = self.cfg
        strat = cfg.view_sampling_strategy
        if strat is None:
            v = torch.tensor(DEVICE_FWD, device=device, dtype=dtype)
            return v.view(1, 3).expand(num, 3)

        if strat == SamplingStrategy.UNIFORM_SPHERE:
            dist = HypersphericalUniform(dim=3, device=device, dtype=dtype)
        elif strat == SamplingStrategy.FORWARD_POWERSPHERICAL:
            mu = torch.tensor(DEVICE_FWD, device=device, dtype=dtype)
            scale = torch.tensor(cfg.view_kappa, device=device, dtype=dtype)
            dist = PowerSpherical(mu, scale)
        else:
            raise ValueError(f"Unsupported view_sampling_strategy: {strat}")

        dirs = dist.rsample((num,))
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if cfg.view_max_angle_deg > 0.0:
            mu = torch.tensor(DEVICE_FWD, device=device, dtype=dtype)
            cos_max = torch.cos(
                torch.tensor(torch.deg2rad(torch.tensor(cfg.view_max_angle_deg)), device=device, dtype=dtype)
            )
            mask = (dirs * mu).sum(dim=-1) < cos_max
            tries = 0
            while mask.any() and tries < 8:
                tries += 1
                resample_n = int(mask.sum().item())
                new_dirs = dist.rsample((resample_n,))
                new_dirs = new_dirs / new_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                dirs[mask] = new_dirs
                mask = (dirs * mu).sum(dim=-1) < cos_max

        return dirs

    def build(self, reference_pose: PoseTW, centers_world: torch.Tensor) -> PoseTW:
        """Construct cam2world candidate poses for given centers.

        Args:
            reference_pose:
                reference2world :class:`PoseTW` used as origin for radial modes and as source of the rig2world
                rotation.
            centers_world:
                `Tensor['N, 3']` candidate camera centers in the world frame.

        Returns:
            :class:`PoseTW` instance encoding cam2world SE(3) transforms for all candidates (length `N`).

        This method takes a reference2world rig pose and world-space candidate centers and returns per-candidate
        :class:`PoseTW` cam2world transforms. It proceeds in two stages:

        1. Construct base poses according to :class:`ViewDirectionMode`:

           * :data:`ViewDirectionMode.FORWARD_RIG`:
             reuse the rig rotation for all candidates and place cameras at `centers_world`.
           * :data:`ViewDirectionMode.RADIAL_AWAY` / :data:`RADIAL_TOWARDS`:
             call :func:`view_axes_from_poses` so that camera optical axes point away from or towards the reference
             pose along the center-reference line, keeping x horizontal.
           * :data:`ViewDirectionMode.TARGET_POINT`:
             build look-at frames for the configured target point, using the world up vector to stabilise roll.

        2. Apply local view jitter:

           * sample camera-frame forward axes via :meth:`_sample_view_dirs_cam`,
           * build orthonormal camera bases from these directions,
           * optionally apply roll jitter around the forward axis, and
           * compose the resulting rotations as right-multiplicative deltas with the base cam2world poses.
        """
        cfg = self.cfg
        device = centers_world.device
        dtype = centers_world.dtype
        n = centers_world.shape[0]

        reference_pose_dev = reference_pose

        if cfg.view_direction_mode is ViewDirectionMode.FORWARD_RIG:
            r_last = reference_pose_dev.R
            if r_last.ndim == 3:
                r_last = r_last[0]
            r_base = r_last.unsqueeze(0).expand(n, 3, 3)
            base_poses = PoseTW.from_Rt(r_base, centers_world)

        elif cfg.view_direction_mode in (ViewDirectionMode.RADIAL_AWAY, ViewDirectionMode.RADIAL_TOWARDS):
            eye = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)
            centers_pose = PoseTW.from_Rt(eye, centers_world)
            base_poses = view_axes_from_poses(
                from_pose=reference_pose_dev,
                to_pose=centers_pose,
                look_away=(cfg.view_direction_mode is ViewDirectionMode.RADIAL_AWAY),
            )

        elif cfg.view_direction_mode is ViewDirectionMode.TARGET_POINT:
            if cfg.view_target_point_world is None:
                raise ValueError("TARGET_POINT mode requires `view_target_point_world` to be set.")
            target = cfg.view_target_point_world.to(device=device, dtype=dtype).view(1, 3)
            wup = world_up_tensor(device=device, dtype=dtype)
            v = target - centers_world
            z_world = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            dot_up = (z_world * wup.view(1, 3)).sum(dim=-1, keepdim=True)
            y_world = wup.view(1, 3) - dot_up * z_world
            y_world = y_world / y_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            x_world = torch.cross(y_world, z_world, dim=-1)
            r_base = torch.stack([x_world, y_world, z_world], dim=-1)
            base_poses = PoseTW.from_Rt(r_base, centers_world)

        else:
            raise ValueError(f"Unsupported view_direction_mode: {cfg.view_direction_mode}")

        if cfg.view_sampling_strategy is None and cfg.view_roll_jitter_deg == 0.0:
            return base_poses

        dirs_cam = self._sample_view_dirs_cam(n, device=device, dtype=dtype)
        z_new = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        up_cam = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).expand_as(z_new)
        x_new = torch.cross(up_cam, z_new, dim=-1)
        x_new = x_new / x_new.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        y_new = torch.cross(z_new, x_new, dim=-1)
        y_new = y_new / y_new.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        r_delta = torch.stack([x_new, y_new, z_new], dim=-1)

        if cfg.view_roll_jitter_deg > 0.0:
            roll = (2.0 * torch.rand(n, device=device, dtype=dtype) - 1.0) * torch.deg2rad(
                torch.tensor(cfg.view_roll_jitter_deg, device=device, dtype=dtype)
            )
            cr, sr = torch.cos(roll), torch.sin(roll)
            r_roll = torch.zeros(n, 3, 3, device=device, dtype=dtype)
            r_roll[:, 0, 0] = cr
            r_roll[:, 0, 1] = -sr
            r_roll[:, 1, 0] = sr
            r_roll[:, 1, 1] = cr
            r_roll[:, 2, 2] = 1.0
            r_delta = torch.matmul(r_delta, r_roll)

        delta_poses = PoseTW.from_Rt(r_delta, torch.zeros_like(centers_world))
        return base_poses.compose(delta_poses)


__all__ = ["OrientationBuilder"]
