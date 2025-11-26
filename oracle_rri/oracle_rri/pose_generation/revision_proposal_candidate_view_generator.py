"""
Simplified candidate pose generator for next‑best‑view (NBV) evaluation.

This module contains a minimal implementation of a candidate pose generator
for the oracle_rri project.  It removes the heavy rule‑based sampling logic
of the original `CandidateViewGenerator` and focuses on clear, composable
components:

1. **Directional sampling:** Uniform or forward‑biased directions are
   drawn from hyperspherical distributions.  Uniform sampling uses the
   `HypersphericalUniform` distribution, which samples standard normals
   and normalises them to unit length【730068789940973†L65-L70】.  Forward
   biasing uses the Power Spherical distribution, which concentrates
   samples around a mean direction while remaining differentiable【344824407890469†L44-L53】.
2. **Position sampling:** Radii are sampled uniformly between
   `min_radius` and `max_radius` and combined with the directions to
   produce offsets in the last‑pose rig frame.
3. **Orientation:** Each candidate is oriented using a look‑at
   construction with a fixed world up vector.  The camera looks back
   towards the last pose; the up vector aligns the x‑axis horizontally so
   there is no roll【193155017766346†L985-L1018】.
4. **Rule filtering:** Optional filters such as minimum distance to the
   mesh or path collision checks can be added in a single place without
   mutating shared state.

This design emphasises transparency and testability.  Each step can be
verified independently with unit tests or visualisations, and the overall
workflow avoids implicit side effects.
"""

from dataclasses import dataclass

import torch

try:
    # External distributions for directional sampling.  These are optional
    # dependencies; if not available, users should install the `power_spherical`
    # package from https://github.com/nicola-decao/power_spherical
    from power_spherical import HypersphericalUniform, PowerSpherical
except ImportError as e:  # pragma: no cover
    raise ImportError("Missing `power_spherical` dependency.  Install with `pip install power_spherical`.") from e


@dataclass
class SimpleCandidateViewGeneratorConfig:
    """Configuration for :class:`SimpleCandidateViewGenerator`.

    Attributes
    ----------
    num_samples : int
        Number of candidate poses to draw per call to :meth:`generate`.
    min_radius, max_radius : float
        Inner and outer radii of the spherical shell (metres).  Centres
        are sampled uniformly in this range.
    min_elev_deg, max_elev_deg : float
        Minimum and maximum elevations (degrees) defining a spherical cap
        around the rig forward axis.  Elevation is measured in the rig
        frame with 0° at the horizontal plane and positive values
        pointing up.
    azimuth_full_circle : bool
        If ``True``, azimuth is sampled over a full 360° range; otherwise
        it is restricted to a half circle in front of the rig.
    sampling_strategy : str
        Either ``"uniform"`` for area‑uniform sampling or ``"forward"`` for
        forward‑biased sampling.  Forward sampling uses the Power Spherical
        distribution.
    kappa : float
        Concentration parameter for forward sampling.  Larger values
        concentrate samples more tightly around the mean direction; ``0``
        degenerates to a uniform distribution【344824407890469†L44-L53】.
    world_up : torch.Tensor
        A 3‑vector specifying the gravity or up direction in the world
        frame.  This is used to construct camera orientations with zero
        roll.
    min_distance_to_mesh : Optional[float]
        If not ``None``, candidate camera centres closer than this
        clearance to the ground‑truth mesh are discarded.  Distance
        computation must be implemented externally.
    ensure_collision_free : bool
        When ``True``, candidate centres whose straight‑line path from
        the last pose intersects the mesh are discarded.  Collision
        detection must be implemented externally.
    ensure_free_space : bool
        When ``True``, candidate centres outside a predefined occupancy
        extent are discarded.  Extent checks must be implemented
        externally.
    """

    num_samples: int = 20
    min_radius: float = 1.0
    max_radius: float = 3.0
    min_elev_deg: float = -20.0
    max_elev_deg: float = 20.0
    azimuth_full_circle: bool = True
    sampling_strategy: str = "uniform"
    kappa: float = 4.0
    world_up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])
    min_distance_to_mesh: float | None = None
    ensure_collision_free: bool = False
    ensure_free_space: bool = False


class SimpleCandidateViewGenerator:
    """Generate candidate camera poses around the latest pose.

    This class implements a simplified next‑best‑view proposal mechanism:

    1. Directions on the unit sphere are sampled either uniformly or
       forward‑biased.  Uniform sampling uses `HypersphericalUniform`
       which draws standard normals and normalises【730068789940973†L65-L70】.  Forward
       sampling uses `PowerSpherical` with a mean pointing along the rig
       forward axis.
    2. Radii are sampled uniformly between ``min_radius`` and ``max_radius``.
    3. Offsets in the rig frame are combined with the last pose to
       produce camera centres in world coordinates.
    4. Each candidate is oriented to look back at the last pose with a
       fixed ``world_up`` vector, guaranteeing zero roll【193155017766346†L985-L1018】.
    5. Optional rule filters can discard invalid candidates based on
       mesh proximity or path collisions.  The implementation of these
       checks is left to the caller.

    The generator returns a list of :class:`PoseTW` objects and a
    boolean mask indicating valid candidates.  It does not mutate
    internal state, making it easy to test and extend.
    """

    def __init__(self, cfg: SimpleCandidateViewGeneratorConfig):
        self.cfg = cfg

    def _sample_directions(self, num: int, device: torch.device) -> torch.Tensor:
        """Draw unit directions in the rig frame.

        Directions are sampled either area‑uniformly or using the
        Power Spherical distribution.  Elevation and azimuth bounds are
        enforced by simple rejection sampling.

        Parameters
        ----------
        num : int
            Number of directions to draw.
        device : torch.device
            Device on which to allocate the returned tensor.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_valid, 3)`` containing unit directions in the
            rig frame.  The number of returned samples may be less than
            ``num`` if some were rejected.
        """
        cfg = self.cfg
        if cfg.sampling_strategy.lower() == "uniform":
            base = HypersphericalUniform(dim=3, device=device)
            dirs = base.rsample((num,))
        elif cfg.sampling_strategy.lower() == "forward":
            # mean direction in rig frame is +Y (forward)
            mean = torch.tensor([0.0, 1.0, 0.0], device=device)
            dist = PowerSpherical(
                loc=mean,
                scale=torch.tensor(cfg.kappa, device=device, dtype=torch.float32),
            )
            dirs = dist.rsample((num,))
        else:
            raise ValueError(f"Unknown sampling strategy: {cfg.sampling_strategy}")

        # Enforce elevation and azimuth bounds via rejection sampling.  Elevation
        # in the rig convention is asin(y), where y is the second component.
        min_elev = torch.tensor(cfg.min_elev_deg * torch.pi / 180, device=device)
        max_elev = torch.tensor(cfg.max_elev_deg * torch.pi / 180, device=device)
        elev = torch.asin(dirs[:, 1])
        # For azimuth, compute atan2(z, x).  In the rig frame, the x‑axis is
        # horizontal (left/right) and the z‑axis is depth.  We wrap into
        # [−π, π] and then restrict if not sampling the full circle.
        azim = torch.atan2(dirs[:, 2], dirs[:, 0])
        mask = (elev >= min_elev) & (elev <= max_elev)
        if not cfg.azimuth_full_circle:
            # Keep directions within ±90° about the forward axis (y)
            mask &= (azim >= -0.5 * torch.pi) & (azim <= 0.5 * torch.pi)
        return dirs[mask]

    def _sample_positions(self, last_pose, num: int) -> torch.Tensor:
        """Sample candidate camera centres in the world frame.

        Parameters
        ----------
        last_pose : PoseTW
            The most recent camera pose (world ← rig).  Assumed to provide
            attributes ``R`` (3×3 rotation) and ``t`` (3‑vector translation).
        num : int
            Number of positions to sample *before* rejection.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_valid, 3)`` containing camera centres in
            world coordinates.
        """
        device = last_pose.R.device
        dirs = self._sample_directions(num, device)
        if dirs.numel() == 0:
            return dirs  # empty
        # Radii uniformly in [min_radius, max_radius]
        cfg = self.cfg
        u = torch.rand(len(dirs), device=device)
        r = cfg.min_radius + u * (cfg.max_radius - cfg.min_radius)
        offsets_rig = dirs * r[:, None]
        # Transform offsets into world frame: X_world = R_world_rig @ X_rig + t_world_rig
        centers_world = (last_pose.R @ offsets_rig.t()).t() + last_pose.t
        return centers_world

    def _look_at(self, cam_pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute rotation matrices so cameras look at ``target`` with fixed up.

        Parameters
        ----------
        cam_pos : torch.Tensor
            (N, 3) world‑frame camera centres.
        target : torch.Tensor
            (N, 3) world‑frame points that cameras should look at (last
            pose centre).  If a single 3‑vector is provided, it is
            broadcast across the batch.

        Returns
        -------
        torch.Tensor
            Rotation matrices of shape (N, 3, 3) that map vectors from the
            camera to the world frame.  The construction uses the
            `world_up` vector to fix the roll, similar to
            ``look_at_view_transform`` in PyTorch3D【193155017766346†L985-L1018】.
        """
        world_up = self.cfg.world_up.to(device=cam_pos.device, dtype=cam_pos.dtype)
        # Ensure batch dimensions match
        if target.dim() == 1:
            target = target.expand_as(cam_pos)
        forward = target - cam_pos  # look back towards last pose
        forward = forward / (forward.norm(dim=-1, keepdim=True) + 1e-8)
        # Right is up × forward
        right = torch.cross(world_up.expand_as(forward), forward, dim=-1)
        right = right / (right.norm(dim=-1, keepdim=True) + 1e-8)
        # Recompute up to ensure orthogonality
        up = torch.cross(forward, right, dim=-1)
        R = torch.stack((right, up, forward), dim=-1)
        return R

    def generate(self, last_pose, mesh=None, occupancy=None):
        """Produce a batch of candidate poses around the last pose.

        Parameters
        ----------
        last_pose : PoseTW
            The most recent camera pose (world ← rig).  Must provide
            attributes ``R`` and ``t``.
        mesh : optional
            Triangle mesh used for distance/collision checks.  If
            provided, it must implement ``signed_distance(points)`` and
            optionally a collision routine.  These methods are *not*
            implemented here.
        occupancy : optional
            Axis‑aligned bounding box (AABB) specifying free space.  If
            provided, it should be a 6‑tuple (xmin, xmax, ymin, ymax,
            zmin, zmax).

        Returns
        -------
        poses : list
            A list of `PoseTW` instances for valid candidates.
        mask_valid : torch.Tensor
            Boolean mask of shape (n_candidates,) indicating which
            candidates passed all filters.
        """
        # Draw more samples than needed to account for rejections.
        positions = self._sample_positions(last_pose, self.cfg.num_samples * 2)
        if positions.numel() == 0:
            return [], torch.tensor([], dtype=torch.bool)
        # Orient cameras using the look‑at construction.
        target = last_pose.t.unsqueeze(0).expand(len(positions), -1)
        R = self._look_at(positions, target)
        # Construct PoseTW objects.  We delay import to avoid circular deps.
        from oracle_rri.utils.frames import PoseTW  # imported here to avoid heavy import cost

        poses_tw = PoseTW.from_R_t(R, positions)
        # Build a validity mask and apply filters
        mask_valid = torch.ones(len(poses_tw), dtype=torch.bool, device=positions.device)
        cfg = self.cfg
        if cfg.min_distance_to_mesh is not None and mesh is not None:
            # Compute signed distances and require clearance > threshold
            dists = mesh.signed_distance(positions)
            mask_valid &= dists > cfg.min_distance_to_mesh
        if cfg.ensure_collision_free and mesh is not None:
            # Placeholder for path collision check: users should implement
            # their own ray tracing here.
            pass
        if cfg.ensure_free_space and occupancy is not None:
            x, y, z = positions.unbind(-1)
            xmin, xmax, ymin, ymax, zmin, zmax = occupancy
            mask_valid &= (x >= xmin) & (x <= xmax)
            mask_valid &= (y >= ymin) & (y <= ymax)
            mask_valid &= (z >= zmin) & (z <= zmax)
        # Subselect the first num_samples valid poses
        valid_indices = torch.where(mask_valid)[0]
        if len(valid_indices) > self.cfg.num_samples:
            valid_indices = valid_indices[: self.cfg.num_samples]
        poses_out = [poses_tw[i] for i in valid_indices.tolist()]
        mask_out = mask_valid[valid_indices]
        return poses_out, mask_out
