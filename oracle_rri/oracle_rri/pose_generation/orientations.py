"""Orientation builder for candidate poses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from efm3d.aria.pose import PoseTW
from power_spherical import HypersphericalUniform, PowerSpherical  # type: ignore[import-untyped]

from ..utils import Console
from ..utils.frames import view_axes_from_poses, world_up_tensor
from .geometry import DEVICE_FWD
from .types import SamplingStrategy, ViewDirectionMode

if TYPE_CHECKING:
    from .candidate_generation import CandidateViewGeneratorConfig


# TODO: if cfg.
class OrientationBuilder:
    """Construct candidate camera orientations from centers and view settings."""

    def __init__(self, cfg: CandidateViewGeneratorConfig):
        self.cfg = cfg
        self.console = Console.with_prefix(self.__class__.__name__).set_verbose(cfg.verbosity)

    def _sample_view_dirs_cam(self, num: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample camera-forward directions in the base camera frame without rejection.

        Sampling rules (priority order):
        1) If both az/el caps are zero and ``view_sampling_strategy`` is ``None`` → deterministic forward.
        2) If any cap is set → box-uniform in yaw/pitch: yaw ~ U(-az, az), pitch ~ U(-el, el).
           This is cheap, unbiased inside the box, and matches the intent of "small jitter".
        3) Else if a sampling_strategy is set → draw from the chosen distribution (legacy path).
        """

        cfg = self.cfg
        az_limit = torch.deg2rad(torch.tensor(cfg.view_max_azimuth_deg, device=device, dtype=dtype))
        el_limit = torch.deg2rad(torch.tensor(cfg.view_max_elevation_deg, device=device, dtype=dtype))

        # 1) No jitter requested.
        if az_limit <= 0 and el_limit <= 0 and cfg.view_sampling_strategy is None:
            v = torch.tensor(DEVICE_FWD, device=device, dtype=dtype)
            return v.view(1, 3).expand(num, 3)

        # 2) Box-uniform jitter in yaw/pitch (preferred).
        if az_limit > 0 or el_limit > 0:
            self.console.dbg(
                f"Sampling view deltas with daz={cfg.view_max_azimuth_deg}°, del={cfg.view_max_elevation_deg}°"
            )
            yaw = (torch.rand(num, device=device, dtype=dtype) * 2.0 - 1.0) * az_limit
            pitch = (torch.rand(num, device=device, dtype=dtype) * 2.0 - 1.0) * el_limit
            cos_pitch = torch.cos(pitch)
            dirs = torch.stack(
                [
                    cos_pitch * torch.sin(yaw),
                    torch.sin(pitch),
                    cos_pitch * torch.cos(yaw),
                ],
                dim=-1,
            )
            return _normalise(dirs)

        # 3) Legacy distributions when no caps are provided.
        strat = cfg.view_sampling_strategy
        if strat == SamplingStrategy.UNIFORM_SPHERE:
            dist = HypersphericalUniform(dim=3, device=device, dtype=dtype)
        elif strat == SamplingStrategy.FORWARD_POWERSPHERICAL:
            mu = torch.tensor(DEVICE_FWD, device=device, dtype=dtype)
            scale = torch.tensor(cfg.view_kappa, device=device, dtype=dtype)
            dist = PowerSpherical(loc=mu, scale=scale)
        else:
            v = torch.tensor(DEVICE_FWD, device=device, dtype=dtype)
            return v.view(1, 3).expand(num, 3)

        return _normalise(dist.rsample((num,)))

    def build(self, reference_pose: PoseTW, centers_world: torch.Tensor) -> tuple[PoseTW, PoseTW | None]:
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

        match cfg.view_direction_mode:
            case ViewDirectionMode.FORWARD_RIG:
                r_last = reference_pose_dev.R
                if r_last.ndim == 3:
                    r_last = r_last[0]
                r_base = r_last.unsqueeze(0).expand(n, 3, 3)
                base_poses = PoseTW.from_Rt(r_base, centers_world)
            case ViewDirectionMode.RADIAL_AWAY | ViewDirectionMode.RADIAL_TOWARDS:
                eye = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)
                centers_pose = PoseTW.from_Rt(eye, centers_world)
                base_poses = view_axes_from_poses(
                    from_pose=reference_pose_dev,
                    to_pose=centers_pose,
                    look_away=(cfg.view_direction_mode is ViewDirectionMode.RADIAL_AWAY),
                )

            case ViewDirectionMode.TARGET_POINT:
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

        if (
            cfg.view_sampling_strategy is None
            and cfg.view_roll_jitter_deg == 0.0
            and cfg.view_max_azimuth_deg == 0.0
            and cfg.view_max_elevation_deg == 0.0
        ):
            return base_poses, None

        dirs_cam = self._sample_view_dirs_cam(n, device=device, dtype=dtype)

        z_new = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # Derive yaw/pitch from the jittered forward and build roll-free rotation.
        yaw = torch.atan2(z_new[:, 0], z_new[:, 2])  # around +Y (up)
        pitch = torch.asin(z_new[:, 1].clamp(-1.0, 1.0))  # around +X (left)

        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)

        ry = torch.zeros(n, 3, 3, device=device, dtype=dtype)
        ry[:, 0, 0] = cy
        ry[:, 0, 2] = sy
        ry[:, 1, 1] = 1.0
        ry[:, 2, 0] = -sy
        ry[:, 2, 2] = cy

        rx = torch.zeros_like(ry)
        rx[:, 0, 0] = 1.0
        rx[:, 1, 1] = cp
        rx[:, 1, 2] = sp
        rx[:, 2, 1] = -sp
        rx[:, 2, 2] = cp

        r_delta = torch.matmul(ry, rx)

        if cfg.view_roll_jitter_deg > 0.0:
            # Jitter is applied as a rotation matrix about the forward axis so the basis stays orthonormal. Adding Gaussian noise to direction vectors would skew/scale them unless you re‑orthogonalise anyway (which is what this code guarantees).
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
        return base_poses.compose(delta_poses), delta_poses


def _normalise(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(v.dtype).eps)


__all__ = ["OrientationBuilder"]
