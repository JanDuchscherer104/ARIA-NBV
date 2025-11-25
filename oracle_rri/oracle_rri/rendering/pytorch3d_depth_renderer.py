"""PyTorch3D depth renderer used for oracle RRI simulations.

This module contains a configurable renderer that turns candidate poses
(:class:`efm3d.aria.PoseTW`) plus an ASE mesh into per-pixel depth maps.
It follows the project's config-as-factory pattern and keeps the public
API torch-native so it can slot into the NBV training loop without
intermediate numpy copies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import trimesh  # type: ignore[import-untyped]
from pydantic import AliasChoices, Field, field_validator
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings  # type: ignore[import-untyped]
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]
from pytorch3d.structures import Meshes  # type: ignore[import-untyped]
from torch import Tensor
from trimesh import Trimesh

from ..utils import BaseConfig, Console, Verbosity, select_device

if TYPE_CHECKING:
    from efm3d.aria import CameraTW, PoseTW
    from trimesh import Trimesh


class Pytorch3DDepthRendererConfig(BaseConfig["Pytorch3DDepthRenderer"]):
    """Configuration for :class:`Pytorch3DDepthRenderer`."""

    target: type["Pytorch3DDepthRenderer"] = Field(  # type: ignore[assignment]
        default_factory=lambda: Pytorch3DDepthRenderer,
        exclude=True,
    )
    """Factory target for :meth:`BaseConfig.setup_target` (auto-populated)."""

    device: str = "cuda"
    """Torch device to run rasterisation on (falls back to CPU if unavailable)."""

    znear: float = 0.01
    """Near clipping plane (metres)."""

    zfar: float = 20.0
    """Far clipping plane (metres); also used to fill miss pixels."""

    faces_per_pixel: int = 1
    """Number of faces stored per pixel; 1 = closest triangle only."""

    cull_backfaces: bool = True
    """If ``True`` drop triangles with normals pointing away from the camera.

    NBV cameras are typically *inside* closed meshes (rooms); backface culling
    would therefore remove the interior walls. Default is ``False`` so both
    sides are rendered.
    """

    blur_radius: float = 0.0
    """Soft rasterizer blur radius; keep ``0`` for hard z-buffer."""

    bin_size: int | None = None
    """Rasterisation bin size; ``None`` lets PyTorch3D pick heuristics."""

    max_faces_per_bin: int | None = None
    """Performance knob mirroring ``RasterizationSettings.max_faces_per_bin``."""

    two_sided: bool = True
    """If ``True`` duplicate faces with reversed winding so interior walls remain visible."""

    add_proxy_walls: bool = False
    """If ``True`` add a thin bounding box shell when large wall areas are missing."""

    proxy_wall_area_threshold: float = 0.2
    """If existing faces cover < this fraction of the expected wall area, add proxy walls."""

    proxy_eps: float = 0.05
    """Epsilon (m) for detecting faces near scene bounds."""

    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    """Computation dtype; use float16/bfloat16 on CUDA for speed (may reduce z precision)."""

    verbosity: Verbosity = Field(
        default=Verbosity.NORMAL,
        validation_alias=AliasChoices("verbosity", "verbose"),
        description="Verbosity level for logging (0=quiet, 1=normal, 2=verbose).",
    )
    """Enable :class:`Console` logging."""

    is_debug: bool = False
    """Force CPU rendering and enable debug logging."""

    @field_validator("verbosity", mode="before")
    @classmethod
    def _coerce_verbosity(cls, value: Any) -> Verbosity:
        return Verbosity.from_any(value)


class Pytorch3DDepthRenderer:
    """Depth rendering backend based on PyTorch3D."""

    def __init__(self, config: Pytorch3DDepthRendererConfig) -> None:
        self.config = config
        self.console = (
            Console.with_prefix(self.__class__.__name__)
            .set_verbosity(self.config.verbosity)
            .set_debug(self.config.is_debug)
        )
        preferred = "cpu" if self.config.is_debug else self.config.device
        self._device = select_device(preferred, component=self.__class__.__name__)

    @property
    def device(self) -> torch.device:
        """Torch device used for rendering."""

        return self._device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render_depth(
        self,
        pose_world_cam: PoseTW,
        mesh: Trimesh,
        camera: CameraTW,
        *,
        frame_index: int | None = None,
        occupancy_extent: torch.Tensor | None = None,
    ) -> Tensor:
        """Render a depth map for a single candidate pose.

        Args:
            pose_world_cam: ``T_world_camera`` pose to render from.
            mesh: Trimesh instance in world coordinates.
            camera: Camera intrinsics (``CameraTW``) describing the RGB or SLAM stream.
            frame_index: Optional frame index for the calib tensor. Defaults to the last frame.

        Returns:
            Tensor[H, W] float32 depth values filled with ``config.zfar`` where no hit occurs.
        """

        batched = self.render_batch(
            poses=pose_world_cam,
            mesh=mesh,
            camera=camera,
            frame_index=frame_index,
            occupancy_extent=occupancy_extent,
        )
        return batched[0]

    def render_batch(
        self,
        poses: PoseTW,
        mesh: Trimesh,
        camera: CameraTW,
        *,
        frame_index: int | None = None,
        occupancy_extent: torch.Tensor | None = None,
    ) -> Tensor:
        """Render depth for a batch of poses."""

        cam_single = self._slice_camera(camera, frame_index)
        width, height, fx, fy = self._camera_parameters(cam_single)
        rotations, translations = self._pose_to_r_t(poses)
        rotations = rotations.to(self.device)
        translations = translations.to(self.device)
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.config.dtype]

        mesh_struct = self._mesh_to_struct(
            mesh,
            batch_size=rotations.shape[0],
            occupancy_extent=occupancy_extent,
        )
        if dtype != torch.float32:
            mesh_struct = mesh_struct.to(dtype=dtype)
        focal, principal = self._perspective_params(camera, width, height)
        cameras = PerspectiveCameras(
            device=self.device,
            R=rotations,
            T=translations,
            focal_length=focal,
            principal_point=principal,
            image_size=torch.tensor([[height, width]], device=self.device),
            in_ndc=True,
        )
        raster_settings = RasterizationSettings(
            image_size=(int(height), int(width)),
            blur_radius=self.config.blur_radius,
            faces_per_pixel=self.config.faces_per_pixel,
            cull_backfaces=self.config.cull_backfaces,
            bin_size=self.config.bin_size,
            max_faces_per_bin=self.config.max_faces_per_bin,
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        autocast_enable = dtype != torch.float32 and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=autocast_enable):
            fragments = rasterizer(mesh_struct)
            depth = fragments.zbuf[..., 0]
        far = torch.full_like(depth, self.config.zfar)
        depth = torch.where(torch.isfinite(depth), depth, far)
        depth = torch.where(depth <= 0, far, depth)
        hit_mask = depth < self.config.zfar
        hit_ratio = float(hit_mask.float().mean().item())
        if self.console.is_debug:
            self.console.dbg_summary(
                "depth_stats",
                {
                    "hits_ratio": hit_ratio,
                    "min": float(depth.min().item()),
                    "max": float(depth.max().item()),
                    "zfar": float(self.config.zfar),
                    "cull_backfaces": self.config.cull_backfaces,
                },
            )
        if hit_ratio == 0.0 and self.config.cull_backfaces:
            self.console.warn(
                "No depth hits recorded (all pixels at zfar). Consider disabling backface culling "
                "for interior viewpoints."
            )
        return depth

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _slice_camera(self, camera: CameraTW, frame_index: int | None) -> CameraTW:
        """Return a single-frame ``CameraTW`` entry."""

        camera = camera.to(self.device)
        data = camera.tensor()
        if data.ndim == 1:
            return camera
        idx = data.shape[0] - 1 if frame_index is None else frame_index
        idx = max(-data.shape[0], min(data.shape[0] - 1, idx))
        return camera[idx]

    def _camera_parameters(self, camera: CameraTW) -> tuple[float, float, float, float]:
        """Extract width, height, fx, fy as floats."""

        size = camera.size.squeeze(0)
        focals = camera.f.squeeze(0)
        width = float(size[0].item())
        height = float(size[1].item())
        fx = float(focals[0].item())
        fy = float(focals[1].item())
        return width, height, fx, fy

    def _perspective_params(
        self,
        camera: CameraTW,
        width: float,
        height: float,
    ) -> tuple[Tensor, Tensor]:
        """Convert CameraTW intrinsics to PyTorch3D-normalised parameters."""

        focal_vals = camera.f.reshape(-1, 2)
        center_vals = camera.c.reshape(-1, 2)
        fx = float(focal_vals[0, 0].item())
        fy = float(focal_vals[0, 1].item())
        cx = float(center_vals[0, 0].item())
        cy = float(center_vals[0, 1].item())
        fx_ndc = 2.0 * fx / width
        fy_ndc = 2.0 * fy / height
        px_ndc = -2.0 * (cx - width / 2.0) / width
        py_ndc = 2.0 * (cy - height / 2.0) / height
        focal = torch.tensor([[fx_ndc, fy_ndc]], device=self.device)
        principal = torch.tensor([[px_ndc, py_ndc]], device=self.device)
        return focal, principal

    def _mesh_to_struct(
        self,
        mesh: Trimesh,
        batch_size: int,
        occupancy_extent: torch.Tensor | None = None,
    ) -> Meshes:
        """Convert ``trimesh`` mesh to a PyTorch3D ``Meshes`` batch."""

        mesh_use = self._maybe_with_proxy_walls(mesh, occupancy_extent=occupancy_extent)
        # mesh_use = mesh
        verts = torch.as_tensor(mesh_use.vertices, dtype=torch.float32, device=self.device)
        faces = torch.as_tensor(mesh_use.faces, dtype=torch.int64, device=self.device)
        if self.config.two_sided:
            faces_rev = faces[:, [0, 2, 1]]
            faces = torch.cat([faces, faces_rev], dim=0)
        mesh_struct = Meshes(verts=[verts], faces=[faces]).extend(batch_size)
        return mesh_struct

    # ------------------------------------------------------------------
    # Proxy walls helper
    # ------------------------------------------------------------------
    def _maybe_with_proxy_walls(
        self,
        mesh: Trimesh,
        occupancy_extent: torch.Tensor | None = None,
    ) -> Trimesh:
        """Append a thin bounding-box shell when major walls are missing."""

        if not self.config.add_proxy_walls:
            return mesh

        mesh_vmin, mesh_vmax = mesh.bounds
        vmin = mesh_vmin
        vmax = mesh_vmax
        if occupancy_extent is not None and occupancy_extent.numel() == 6:
            extent_cpu = occupancy_extent.detach().cpu().float()
            xmin, xmax, ymin, ymax, zmin, zmax = extent_cpu.tolist()
            occ_vmin = np.array([xmin, ymin, zmin], dtype=np.float32)
            occ_vmax = np.array([xmax, ymax, zmax], dtype=np.float32)
            vmin = np.minimum(mesh_vmin, occ_vmin)
            vmax = np.maximum(mesh_vmax, occ_vmax)
        extents = vmax - vmin
        desired_area = {
            "xmin": extents[1] * extents[2],
            "xmax": extents[1] * extents[2],
            "ymin": extents[0] * extents[2],
            "ymax": extents[0] * extents[2],
            "zmin": extents[0] * extents[1],
            "zmax": extents[0] * extents[1],
        }
        centers = mesh.triangles_center
        areas = mesh.area_faces
        eps = self.config.proxy_eps
        plane_masks = {
            "xmin": centers[:, 0] <= vmin[0] + eps,
            "xmax": centers[:, 0] >= vmax[0] - eps,
            "ymin": centers[:, 1] <= vmin[1] + eps,
            "ymax": centers[:, 1] >= vmax[1] - eps,
            "zmin": centers[:, 2] <= vmin[2] + eps,
            "zmax": centers[:, 2] >= vmax[2] - eps,
        }
        coverage = {k: areas[m].sum() for k, m in plane_masks.items()}
        missing = [
            k
            for k, cov in coverage.items()
            if desired_area[k] > 0 and cov < desired_area[k] * self.config.proxy_wall_area_threshold
        ]
        if not missing:
            return mesh

        # Build a box shell at bounds; keep only faces for missing planes to avoid doubling others.
        box = trimesh.creation.box(
            extents=extents,
            transform=trimesh.transformations.translation_matrix(vmin + extents / 2.0),
        )
        mask_keep = []
        for n in box.face_normals:
            if (
                (np.allclose(n, [1, 0, 0]) and "xmax" in missing)
                or (np.allclose(n, [-1, 0, 0]) and "xmin" in missing)
                or (np.allclose(n, [0, 1, 0]) and "ymax" in missing)
                or (np.allclose(n, [0, -1, 0]) and "ymin" in missing)
                or (np.allclose(n, [0, 0, 1]) and "zmax" in missing)
                or (np.allclose(n, [0, 0, -1]) and "zmin" in missing)
            ):
                mask_keep.append(True)
            else:
                mask_keep.append(False)
        mask_keep = np.array(mask_keep, dtype=bool)
        if not mask_keep.any():
            return mesh
        proxy = Trimesh(vertices=box.vertices, faces=box.faces[mask_keep], process=False)
        merged = trimesh.util.concatenate([mesh, proxy])
        if self.console.is_debug:
            self.console.dbg(
                f"Added proxy walls for planes {missing}; proxy faces={proxy.faces.shape[0]}, total={merged.faces.shape[0]}"
            )
        return merged

    def _pose_to_r_t(self, poses: PoseTW) -> tuple[Tensor, Tensor]:
        """Convert ``T_world_camera`` into PyTorch3D view matrices."""

        poses = poses.to(self.device)
        rot_wc = poses.R
        t_wc = poses.t
        rot_cw = rot_wc.transpose(-1, -2)
        t_cw = -(rot_cw @ t_wc.unsqueeze(-1)).squeeze(-1)
        if rot_cw.ndim == 2:
            rot_cw = rot_cw.unsqueeze(0)
            t_cw = t_cw.unsqueeze(0)
        return rot_cw, t_cw
