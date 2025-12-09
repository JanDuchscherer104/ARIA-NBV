"""PyTorch3D depth renderer used for oracle RRI simulations.

This module contains a configurable renderer that turns candidate poses
(:class:`efm3d.aria.PoseTW`) plus an ASE mesh into per-pixel depth maps.
It follows the project's config-as-factory pattern and keeps the public
API torch-native so it can slot into the NBV training loop without
intermediate numpy copies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from pydantic import Field
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings  # type: ignore[import-untyped]
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]
from pytorch3d.structures import Meshes  # type: ignore[import-untyped]
from torch import Tensor
from trimesh import Trimesh  # type: ignore[import-untyped]

from ..data.mesh_cache import get_pytorch3d_mesh
from ..utils import BaseConfig, Console, Verbosity

if TYPE_CHECKING:
    from efm3d.aria import CameraTW, PoseTW
    from trimesh import Trimesh


class Pytorch3DDepthRendererConfig(BaseConfig["Pytorch3DDepthRenderer"]):
    """Configuration for :class:`Pytorch3DDepthRenderer`."""

    target: type["Pytorch3DDepthRenderer"] = Field(  # type: ignore[assignment]
        default_factory=lambda: Pytorch3DDepthRenderer,
        exclude=True,
    )

    device: str = "cuda"
    """Torch device to run rasterisation on (falls back to CPU if unavailable)."""

    zfar: float = 20.0
    """Far clipping plane (metres); also used to fill miss pixels."""

    znear: float = 1e-3
    """Near clipping plane (metres); triangles closer than this are clipped."""

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

    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    """Computation dtype; use float16/bfloat16 on CUDA for speed"""

    verbosity: Verbosity = Field(
        default=Verbosity.VERBOSE,
        description="Verbosity level for logging (0=quiet, 1=normal, 2=verbose).",
    )
    """Enable :class:`Console` logging."""


class Pytorch3DDepthRenderer:
    """Depth rendering backend based on PyTorch3D."""

    def __init__(self, config: Pytorch3DDepthRendererConfig) -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbosity(self.config.verbosity)

        self.device = self._resolve_device(self.config.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(value: str | torch.device) -> torch.device:
        if isinstance(value, torch.device):
            return value
        if value is None or str(value).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(value)

    def render(
        self,
        poses: PoseTW,
        mesh: Trimesh | tuple[torch.Tensor, torch.Tensor],
        camera: CameraTW,
        *,
        frame_index: int | None = None,
        mesh_cache_key: str | None = None,
    ) -> tuple[Tensor, Tensor, PerspectiveCameras]:
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
                "image_size": image_size[0, :],
                "focal_length": focal_length[0, :],
                "principal_point": principal_point[0, :],
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

        mesh_struct = mesh_struct_single.extend(rotations.shape[0])

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
            faces_per_pixel=1,  # Closest simplex only
            cull_backfaces=False,
            bin_size=self.config.bin_size,
            max_faces_per_bin=self.config.max_faces_per_bin,
            cull_to_frustum=False,
            z_clip_value=self.config.znear,
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        autocast_enable = dtype != torch.float32 and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=autocast_enable):
            fragments = rasterizer.forward(mesh_struct)

            depth = fragments.zbuf.squeeze(-1)  # (B, H, W)
            pix_to_face = fragments.pix_to_face.squeeze(-1)  # (B, H, W)

        return (
            depth,
            pix_to_face,
            cameras,
        )

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
