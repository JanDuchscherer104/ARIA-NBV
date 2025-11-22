"""Depth and point-cloud rendering for candidate viewpoints.

- Input: candidate `PoseTW` (``T_world_camera``) and GT mesh (world frame).
- Output: per‑pixel depth map (metres) in the **candidate camera frame** plus
  optional point cloud in world coordinates.

Uses `efm3d.utils.ray.ray_grid` + `transform_rays` to generate/render rays; ray–
mesh intersection defaults to trimesh (PyEmbree if available). This keeps the
intersection backend swappable when we wire a fully GPU path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import trimesh
from efm3d.aria import CameraTW, PoseTW
from efm3d.utils.ray import ray_grid, transform_rays
from pydantic import Field

from ..utils import BaseConfig, Console

#   TODO: Add pytest coverage for: (a) synthetic cube mesh hit/miss depth map, (b) point count vs. valid-ray mask.
#   TODO: If you want true GPU intersections, swap mesh.ray with a CUDA path (e.g., PyTorch3D’s ray_triangle).


class IntersectionBackend(str, Enum):
    """Supported ray-mesh backends."""

    PYEMBREE = "pyembree"
    TRIMESH = "trimesh"


class CandidatePointCloudGeneratorConfig(BaseConfig["CandidatePointCloudGenerator"]):
    """Config-as-factory for candidate depth/point rendering."""

    target: type["CandidatePointCloudGenerator"] = Field(
        default_factory=lambda: CandidatePointCloudGenerator,
        exclude=True,
    )

    backend: IntersectionBackend = Field(
        default=IntersectionBackend.PYEMBREE,
        description="Ray-mesh backend (falls back to trimesh if unavailable).",
    )
    device: str = Field(default="cuda", description="Device for ray generation (rays/depth tensors).")
    max_depth: float = Field(
        default=20.0,
        description="Clamp depth (m) for rays with no intersection; acts as far plane.",
    )
    verbose: bool = Field(default=False)
    is_debug: bool = Field(default=False, description="Enable debug logging for ray stats.")


@dataclass(slots=True)
class CandidatePointCloudGenerator:
    """Render depth for candidate poses against a GT mesh."""

    config: CandidatePointCloudGeneratorConfig
    console: Console | None = None
    _pyembree_available: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "console",
            Console.with_prefix(self.__class__.__name__)
            .set_verbose(self.config.verbose)
            .set_debug(self.config.is_debug),
        )
        object.__setattr__(self, "_pyembree_available", hasattr(trimesh.ray, "ray_pyembree"))

    def render_depth(
        self,
        pose_world_cam: PoseTW,
        mesh: trimesh.Trimesh,
        camera: CameraTW,
    ) -> torch.Tensor:
        """Render a single depth map for candidate camera.

        Args:
            pose_world_cam: ``T_world_camera`` pose of the candidate.
            mesh: GT mesh in world frame.
            camera: Camera intrinsics (RGB camera recommended).

        Returns:
            Tensor[H, W] float32 depth in metres in the candidate camera frame.
            Missing rays are filled with ``max_depth``.
        """

        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        rays_cam, valid = ray_grid(camera)  # (H,W,6) or (B,H,W,6)
        if rays_cam.ndim == 4:
            rays_cam = rays_cam[0]
            valid = valid[0]
        rays_cam = rays_cam.to(device)
        valid = valid.to(device)

        rays_world = transform_rays(rays_cam, pose_world_cam.to(device))
        rays_world = torch.where(
            valid.unsqueeze(-1),
            rays_world,
            torch.zeros_like(rays_world),
        )
        origins = rays_world[..., :3].reshape(-1, 3)
        directions = rays_world[..., 3:].reshape(-1, 3)
        valid_flat = valid.view(-1)
        origins_np = origins[valid_flat].cpu().numpy()
        directions_np = directions[valid_flat].cpu().numpy()

        # Ray-mesh intersection
        if self.config.backend == IntersectionBackend.PYEMBREE and self._pyembree_available:
            ray_engine = mesh.ray
        else:
            # trimesh fallback (still uses rtree/numba if installed)
            ray_engine = mesh.ray

        try:
            locations, index_ray, _ = ray_engine.intersects_location(
                ray_origins=origins_np,
                ray_directions=directions_np,
                multiple_hits=False,
            )
        except BaseException as exc:  # pragma: no cover - backend-specific errors
            self.console.warn(f"Ray intersection failed, falling back to no-hits. Details: {exc}")
            locations = np.empty((0, 3), dtype=np.float32)
            index_ray = np.empty((0,), dtype=np.int64)

        h, w = rays_cam.shape[:2]
        depth = torch.full((h, w), self.config.max_depth, device=device, dtype=torch.float32)
        if len(locations) > 0:
            loc_t = torch.from_numpy(locations).to(device=device, dtype=torch.float32)
            orig_t = torch.from_numpy(origins_np[index_ray]).to(device=device, dtype=torch.float32)
            ray_depth = torch.linalg.norm(loc_t - orig_t, dim=1)
            update_idx = torch.from_numpy(index_ray).to(device)
            valid_indices = torch.nonzero(valid_flat, as_tuple=False).squeeze(-1)
            depth.view(-1)[valid_indices[update_idx]] = ray_depth
        hit_ratio = (depth < self.config.max_depth).float().mean().item()
        if self.console.is_debug:
            self.console.dbg_summary(
                "depth_render_stats",
                {
                    "hit_ratio": hit_ratio,
                    "n_rays": int(valid_flat.numel()),
                    "n_valid": int(valid_flat.sum().item()),
                    "mesh_faces": mesh.faces.shape[0] if hasattr(mesh, "faces") else None,
                },
            )

        return depth

    def render_point_cloud(
        self,
        pose_world_cam: PoseTW,
        mesh: trimesh.Trimesh,
        camera: CameraTW,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render depth then unproject to world-frame point cloud.

        Returns:
            depth (H,W) in metres, points_world (N,3) float32.
        """

        depth = self.render_depth(pose_world_cam=pose_world_cam, mesh=mesh, camera=camera)

        # Rays in camera frame
        device = depth.device
        rays_cam, valid = ray_grid(camera)
        if rays_cam.ndim == 4:
            rays_cam = rays_cam[0]
            valid = valid[0]
        rays_cam = rays_cam.to(device)
        valid = valid.to(device)
        dirs = rays_cam[..., 3:].reshape(-1, 3)
        depth_flat = depth.view(-1)
        valid_mask = valid.view(-1) & (depth_flat < self.config.max_depth)
        pts_cam = dirs[valid_mask] * depth_flat[valid_mask].unsqueeze(1)
        pts_world = pose_world_cam.to(device).transform(pts_cam)
        return depth, pts_world


__all__ = [
    "CandidatePointCloudGenerator",
    "CandidatePointCloudGeneratorConfig",
    "IntersectionBackend",
]
