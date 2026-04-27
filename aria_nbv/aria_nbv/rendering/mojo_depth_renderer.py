"""Mojo-backed depth renderer for oracle candidate depths."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import torch
from pydantic import Field, field_validator

from ..utils import BaseConfig, Console, Verbosity
from ..utils.mojo_subprocess import run_mojo_subprocess
from .mojo_backend import is_mojo_thread_context_supported, render_depth_map_mojo

if TYPE_CHECKING:
    from efm3d.aria import CameraTW, PoseTW


class MojoDepthRendererConfig(BaseConfig):
    """Configuration for :class:`MojoDepthRenderer`."""

    @property
    def target(self) -> type["MojoDepthRenderer"]:
        return MojoDepthRenderer

    device: Annotated[torch.device, Field(default="auto")]
    """Torch device used for returned tensors."""

    zfar: float = 20.0
    """Far clipping plane and fill value for miss pixels."""

    znear: float = 1e-3
    """Near clipping plane in metres."""

    workers: int | None = None
    """Optional CPU worker override for the Mojo kernels."""

    verbosity: Verbosity = Field(default=Verbosity.VERBOSE)
    """Verbosity level for logging."""

    is_debug: bool = False
    """Enable detailed debug logging."""

    _resolve_device = field_validator("device", mode="before")(BaseConfig._resolve_device)


class MojoDepthRenderer:
    """Closest-hit depth renderer backed by a Python-importable Mojo module."""

    def __init__(self, config: MojoDepthRendererConfig) -> None:
        self.config = config
        self.console = (
            Console.with_prefix(self.__class__.__name__)
            .set_verbosity(self.config.verbosity)
            .set_debug(self.config.is_debug)
        )
        self.device = self.config.device

    def render_depths(
        self,
        *,
        poses: PoseTW,
        mesh: tuple[torch.Tensor, torch.Tensor],
        camera: CameraTW,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render depth and validity masks for a batch of candidate poses."""

        if not is_mojo_thread_context_supported():
            self.console.warn(
                "Mojo depth kernels are not safe in this worker thread; rendering depths in a subprocess.",
            )
            payload = {
                "poses": poses.tensor().detach().to(device="cpu"),
                "camera": camera.tensor().detach().to(device="cpu"),
                "verts": mesh[0].detach().to(device="cpu"),
                "faces": mesh[1].detach().to(device="cpu"),
                "znear": float(self.config.znear),
                "zfar": float(self.config.zfar),
                "workers": self.config.workers,
            }
            result = run_mojo_subprocess("render_depths", payload)
            return (
                result["depths"].to(device=self.device, dtype=torch.float32),
                result["depths_valid_mask"].to(device=self.device),
            )

        verts, faces = mesh
        pose_tensor = poses.tensor()
        num_poses = 1 if pose_tensor.ndim == 1 else int(pose_tensor.shape[0])

        depth_maps: list[torch.Tensor] = []
        hit_masks: list[torch.Tensor] = []
        for idx in range(num_poses):
            pose_i = poses if num_poses == 1 else poses[idx]
            camera_i = camera if camera.tensor().ndim == 1 else camera[idx]
            width, height, fx, fy, cx, cy = self._camera_parameters(camera_i)
            triangles_cam = self._triangles_in_camera_frame(verts=verts, faces=faces, pose_world_cam=pose_i)
            depth_i, hit_i = render_depth_map_mojo(
                triangles_cam,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                znear=float(self.config.znear),
                zfar=float(self.config.zfar),
                device=self.device,
                workers=self.config.workers,
            )
            depth_maps.append(depth_i)
            hit_masks.append(hit_i)

        return torch.stack(depth_maps, dim=0), torch.stack(hit_masks, dim=0)

    def _camera_parameters(self, camera: CameraTW) -> tuple[int, int, float, float, float, float]:
        size = camera.size.reshape(-1, 2).to(device=self.device, dtype=torch.float32)[0]
        focals = camera.f.reshape(-1, 2).to(device=self.device, dtype=torch.float32)[0]
        centers = camera.c.reshape(-1, 2).to(device=self.device, dtype=torch.float32)[0]
        return (
            int(size[0].item()),
            int(size[1].item()),
            float(focals[0].item()),
            float(focals[1].item()),
            float(centers[0].item()),
            float(centers[1].item()),
        )

    def _triangles_in_camera_frame(
        self,
        *,
        verts: torch.Tensor,
        faces: torch.Tensor,
        pose_world_cam: PoseTW,
    ) -> torch.Tensor:
        pose_cam_world = pose_world_cam.inverse()
        matrix = pose_cam_world.matrix
        if matrix.ndim == 3:
            matrix = matrix[0]
        rotation = matrix[:3, :3].to(device=verts.device, dtype=verts.dtype)
        translation = matrix[:3, 3].to(device=verts.device, dtype=verts.dtype)
        verts_cam = verts @ rotation.T + translation
        return verts_cam[faces]


__all__ = ["MojoDepthRenderer", "MojoDepthRendererConfig"]
