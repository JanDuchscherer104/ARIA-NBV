"""PyTorch3D depth renderer used for oracle RRI simulations.

This module contains a configurable renderer that turns candidate poses
(:class:`efm3d.aria.PoseTW`) plus an ASE mesh into per-pixel depth maps.
It follows the project's config-as-factory pattern and keeps the public
API torch-native so it can slot into the NBV training loop without
intermediate numpy copies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pydantic import Field
from pytorch3d.renderer import FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from torch import Tensor

from ..utils import BaseConfig, Console

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
    """If ``True`` drop triangles with normals pointing away from the camera."""

    blur_radius: float = 0.0
    """Soft rasterizer blur radius; keep ``0`` for hard z-buffer."""

    bin_size: int | None = None
    """Rasterisation bin size; ``None`` lets PyTorch3D pick heuristics."""

    max_faces_per_bin: int | None = None
    """Performance knob mirroring ``RasterizationSettings.max_faces_per_bin``."""

    verbose: bool = False
    """Enable :class:`Console` logging."""

    is_debug: bool = False
    """Force CPU rendering and enable debug logging."""


class Pytorch3DDepthRenderer:
    """Depth rendering backend based on PyTorch3D."""

    def __init__(self, config: Pytorch3DDepthRendererConfig) -> None:
        self.config = config
        self.console = (
            Console.with_prefix(self.__class__.__name__)
            .set_verbose(self.config.verbose)
            .set_debug(self.config.is_debug)
        )
        requested = torch.device(self.config.device)
        if self.config.is_debug or (requested.type == "cuda" and not torch.cuda.is_available()):
            if requested.type == "cuda":
                self.console.warn("CUDA unavailable; falling back to CPU for PyTorch3D rendering.")
            self._device = torch.device("cpu")
        else:
            self._device = requested

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
        )
        return batched[0]

    def render_batch(
        self,
        poses: PoseTW,
        mesh: Trimesh,
        camera: CameraTW,
        *,
        frame_index: int | None = None,
    ) -> Tensor:
        """Render depth for a batch of poses."""

        cam_single = self._slice_camera(camera, frame_index)
        width, height, fx, fy = self._camera_parameters(cam_single)
        rotations, translations = self._pose_to_r_t(poses)
        rotations = rotations.to(self.device)
        translations = translations.to(self.device)

        mesh_struct = self._mesh_to_struct(mesh, batch_size=rotations.shape[0])
        cameras = self._build_cameras(
            rotations=rotations,
            translations=translations,
            height=height,
            width=width,
            fy=fy,
            aspect_ratio=width / height,
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
        fragments = rasterizer(mesh_struct)
        depth = fragments.zbuf[..., 0]
        depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, self.config.zfar))
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

    def _build_cameras(
        self,
        *,
        rotations: Tensor,
        translations: Tensor,
        height: float,
        width: float,
        fy: float,
        aspect_ratio: float,
    ) -> FoVPerspectiveCameras:
        """Create PyTorch3D cameras for the batch."""

        fov_y = 2.0 * float(torch.atan(torch.tensor(0.5 * height / fy)).item()) * 180.0 / torch.pi
        return FoVPerspectiveCameras(
            device=self.device,
            R=rotations,
            T=translations,
            znear=self.config.znear,
            zfar=self.config.zfar,
            fov=fov_y,
            aspect_ratio=aspect_ratio,
        )

    def _mesh_to_struct(self, mesh: Trimesh, batch_size: int) -> Meshes:
        """Convert ``trimesh`` mesh to a PyTorch3D ``Meshes`` batch."""

        verts = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device=self.device)
        mesh_struct = Meshes(verts=[verts], faces=[faces]).extend(batch_size)
        return mesh_struct

    def _pose_to_r_t(self, poses: PoseTW) -> tuple[Tensor, Tensor]:
        """Convert ``T_world_camera`` into PyTorch3D view matrices."""

        pose_tensor = poses.to(self.device).matrix
        rot_wc = pose_tensor[..., :3, :3]
        t_wc = pose_tensor[..., :3, 3]
        rot_cw = rot_wc.transpose(-1, -2)
        t_cw = -(rot_cw @ t_wc.unsqueeze(-1)).squeeze(-1)
        if rot_cw.ndim == 2:
            rot_cw = rot_cw.unsqueeze(0)
            t_cw = t_cw.unsqueeze(0)
        return rot_cw, t_cw
