"""PyTorch3D depth renderer used for oracle RRI simulations.

This module contains a configurable renderer that turns candidate poses
(:class:`efm3d.aria.PoseTW`) plus an ASE mesh into per-pixel depth maps.
It follows the project's config-as-factory pattern and keeps the public
API torch-native so it can slot into the NBV training loop without
intermediate numpy copies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import AliasChoices, Field, field_validator
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings  # type: ignore[import-untyped]
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]
from pytorch3d.structures import Meshes  # type: ignore[import-untyped]
from torch import Tensor
from trimesh import Trimesh  # type: ignore[import-untyped]

from ..data.mesh_cache import get_pytorch3d_mesh
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

    zfar: float = 20.0
    """Far clipping plane (metres); also used to fill miss pixels."""

    faces_per_pixel: int = 1
    """Number of faces stored per pixel; 1 = closest triangle only."""

    cull_backfaces: bool = False
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

    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    """Computation dtype; use float16/bfloat16 on CUDA for speed (may reduce z precision)."""

    verbosity: Verbosity = Field(
        default=Verbosity.VERBOSE,
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
        mesh: Trimesh | tuple[torch.Tensor, torch.Tensor],
        camera: CameraTW,
        *,
        frame_index: int | None = None,
        mesh_cache_key: str | None = None,
    ) -> Tensor:
        """Render a single depth map."""

        batched = self.render(
            poses=pose_world_cam,
            mesh=mesh,
            camera=camera,
            frame_index=frame_index,
            mesh_cache_key=mesh_cache_key,
        )
        return batched[0]

    def render(
        self,
        poses: PoseTW,
        mesh: Trimesh | tuple[torch.Tensor, torch.Tensor],
        camera: CameraTW,
        *,
        frame_index: int | None = None,
        mesh_cache_key: str | None = None,
    ) -> Tensor:
        """Render depth for a batch of poses."""

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.config.dtype]

        cam_single = self._slice_camera(camera, frame_index)
        width, height, focal_length, principal_point, image_size = self._camera_intrinsics(
            cam_single, dtype=dtype, batch_size=poses.shape[0]
        )
        self.console.plog(
            {
                "image_size": image_size,
                "focal_length": focal_length,
                "principal_point": principal_point,
                "width": width,
                "height": height,
            }
        )

        poses_cw = poses.inverse().to(self.device)
        self.console.dbg_summary("poses_cw", poses_cw)
        rotations, translations = poses_cw.R, poses_cw.t
        batch_size = rotations.shape[0]
        if focal_length.shape[0] == 1 and batch_size > 1:
            focal_length = focal_length.expand(batch_size, -1)
            principal_point = principal_point.expand(batch_size, -1)
            image_size = image_size.expand(batch_size, -1)

        # Build (and cache) PyTorch3D mesh structure
        if isinstance(mesh, tuple):
            verts_t, faces_t = mesh
        else:
            verts_t = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=self.device)
            faces_t = torch.as_tensor(mesh.faces, dtype=torch.int64, device=self.device)
        mesh_struct_single = get_pytorch3d_mesh(
            verts_t,
            faces_t,
            cache_key=f"{mesh_cache_key or 'p3d_mesh_cache'}_{self.device.type}",
            device=self.device,
        )
        if self.console.is_debug and rotations.shape[0] > 0:
            verts_world = mesh_struct_single.verts_packed()
            verts_cam = verts_world @ rotations[0].transpose(-1, -2) + translations[0]
            min_z = float(verts_cam[:, 2].min().item())
            self.console.dbg_summary("verts_cam_min_z", {"min_z": min_z})
        mesh_struct = mesh_struct_single.extend(rotations.shape[0])
        if dtype != torch.float32:
            mesh_struct = mesh_struct.to(dtype=dtype)

        cameras = PerspectiveCameras(
            device=self.device,
            R=rotations,
            T=translations,
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=image_size,
            in_ndc=False,
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
        # with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=autocast_enable):
        #     fragments = rasterizer(mesh_struct)
        #     depth = fragments.zbuf[..., 0]
        # far = torch.full_like(depth, self.config.zfar)

        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=autocast_enable):
            fragments = rasterizer(mesh_struct)
            # <Latest change: not sure if correct!>, trying to fix `Metric depth bug: fragments.zbuf is a post‑projection z-buffer, not guaranteed to be Euclidean meters. We currently treat it as metric depth. Fix by reconstructing the hit point with barycentrics and re‑projecting to camera z`
            pix_to_face = fragments.pix_to_face[..., 0]  # (B, H, W)
            bary = fragments.bary_coords[..., 0, :]  # (B, H, W, 3)

        # Prepare metric depth buffer.
        depth = torch.full_like(pix_to_face, self.config.zfar, dtype=dtype, device=self.device)
        far = depth.clone()

        # Mapping from batch index to face offsets produced by Meshes.extend.
        face_offset = mesh_struct.mesh_to_faces_packed_first_idx()  # (B + 1,)
        verts_single = mesh_struct_single.verts_packed().to(dtype=dtype, device=self.device)  # (Fv, 3)
        faces_single = mesh_struct_single.faces_packed()  # (Ff, 3)

        for b in range(rotations.shape[0]):
            mask = pix_to_face[b] >= 0
            if not mask.any():
                continue
            fids = pix_to_face[b][mask] - face_offset[b]
            tri = faces_single[fids]
            v0 = verts_single[tri[:, 0]]
            v1 = verts_single[tri[:, 1]]
            v2 = verts_single[tri[:, 2]]
            bcoords = bary[b][mask]
            pts_world = bcoords[:, 0:1] * v0 + bcoords[:, 1:2] * v1 + bcoords[:, 2:3] * v2
            R_cw = rotations[b]
            t_cw = translations[b]
            pts_cam = pts_world @ R_cw.transpose(-1, -2) + t_cw  # world → cam
            depth_b = pts_cam[:, 2]
            depth[b][mask] = depth_b
        # </Latest change: not sure if correct!>

        depth = torch.where(torch.isfinite(depth), depth, far)
        depth = torch.where(depth <= 0, far, depth)
        hit_mask = depth < self.config.zfar
        hit_ratio = float(hit_mask.float().mean().item())

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
        """Return a per-candidate or single-frame ``CameraTW`` entry."""

        camera = camera.to(self.device)
        data = camera.tensor()
        if data.ndim == 1:
            return camera
        if frame_index is None:
            # Already per-candidate intrinsics.
            return camera
        idx = data.shape[0] - 1 if frame_index is None else frame_index
        idx = max(-data.shape[0], min(data.shape[0] - 1, idx))
        return camera[idx]

    def _camera_intrinsics(
        self,
        camera: CameraTW,
        *,
        dtype: torch.dtype,
        batch_size: int,
    ) -> tuple[int, int, Tensor, Tensor, Tensor]:
        """Return intrinsics ready for ``PerspectiveCameras``.

        Args:
            camera: Single-frame ``CameraTW`` on the target device.
            dtype: Torch dtype to use for focal/principal/image_size tensors.

        Returns:
            Tuple of ``(width_px, height_px, focal_length, principal_point, image_size)`` where
            the last three items are shaped ``(B, 2)`` on ``self.device`` with the provided dtype.
        """

        size_all = camera.size.reshape(-1, 2).to(device=self.device, dtype=torch.float32)
        if size_all.shape[0] not in (1, batch_size):
            raise ValueError(f"Camera size batch {size_all.shape[0]} does not match poses batch {batch_size}")
        size_base = size_all[0]
        if not torch.allclose(size_all, size_base):
            raise ValueError("Per-candidate varying image sizes are not supported.")
        width = int(size_base[0].item())
        height = int(size_base[1].item())

        focal_all = camera.f.reshape(-1, 2).to(device=self.device, dtype=dtype)
        principal_all = camera.c.reshape(-1, 2).to(device=self.device, dtype=dtype)

        if focal_all.shape[0] == 1 and batch_size > 1:
            focal_all = focal_all.expand(batch_size, -1)
            principal_all = principal_all.expand(batch_size, -1)
            size_all = size_all.expand(batch_size, -1)
        elif focal_all.shape[0] != batch_size:
            raise ValueError(f"Camera focal batch {focal_all.shape[0]} does not match poses batch {batch_size}")

        image_size = torch.stack((size_all[:, 1], size_all[:, 0]), dim=-1).to(dtype=dtype)

        return width, height, focal_all, principal_all, image_size

    def _mesh_to_struct(
        self,
        mesh: Trimesh,
        batch_size: int,
        *,
        mesh_cache_key: str | None = None,
    ) -> Meshes:
        """Convert ``trimesh`` mesh to a cached PyTorch3D ``Meshes`` batch."""

        verts = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device=self.device)
        mesh_struct = get_pytorch3d_mesh(
            verts,
            faces,
            cache_key=mesh_cache_key or "p3d_mesh_cache",
            device=self.device,
        )
        return mesh_struct.extend(batch_size)
