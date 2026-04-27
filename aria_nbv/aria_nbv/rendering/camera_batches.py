"""Backend-agnostic camera batch abstractions for oracle rendering paths."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import torch
from efm3d.aria import CameraTW, PoseTW

from ..utils.pytorch3d_compat import PerspectiveCameras, import_perspective_cameras


class CameraBatchBackend(StrEnum):
    """Supported runtime camera batch representations."""

    PYTORCH3D = "pytorch3d"
    NATIVE = "native"


@dataclass(slots=True)
class NativeCameraBatch:
    """MPS-compatible camera batch based on ``CameraTW`` and ``PoseTW``."""

    camera_tw: CameraTW
    """Per-candidate or shared camera intrinsics."""

    pose_world_cam: PoseTW
    """World←camera poses aligned with the batch."""

    backend: CameraBatchBackend = CameraBatchBackend.NATIVE

    def to(self, device: torch.device | str) -> "NativeCameraBatch":
        """Move the native batch to the target device."""

        return NativeCameraBatch(
            camera_tw=self.camera_tw.to(device=device),
            pose_world_cam=self.pose_world_cam.to(device=device),
        )

    @property
    def image_size(self) -> torch.Tensor:
        """Return per-camera image sizes shaped like ``(N,2)``."""

        return self.camera_tw.size.reshape(-1, 2)

    @property
    def batch_size(self) -> int:
        """Return the number of cameras represented by this batch."""

        pose_tensor = self.pose_world_cam.tensor()
        if pose_tensor.ndim == 1:
            return 1
        return int(pose_tensor.shape[0])

    def select(self, index: int) -> "NativeCameraBatch":
        """Return one camera entry by index."""

        if self.camera_tw.tensor().ndim == 1:
            camera_i = self.camera_tw
        else:
            camera_i = self.camera_tw[index]
        pose_tensor = self.pose_world_cam.tensor()
        pose_i = self.pose_world_cam if pose_tensor.ndim == 1 else self.pose_world_cam[index]
        return NativeCameraBatch(camera_tw=camera_i, pose_world_cam=pose_i)


CameraBatchLike: TypeAlias = PerspectiveCameras | NativeCameraBatch


def is_pytorch3d_camera_batch(value: object) -> bool:
    """Return ``True`` when ``value`` is a ``PerspectiveCameras`` batch."""

    perspective_cameras = import_perspective_cameras()
    return isinstance(value, perspective_cameras)


def is_native_camera_batch(value: object) -> bool:
    """Return ``True`` when ``value`` is a native camera batch."""

    return isinstance(value, NativeCameraBatch)


def camera_batch_to(batch: CameraBatchLike, device: torch.device | str) -> CameraBatchLike:
    """Move a camera batch to the target device."""

    if is_native_camera_batch(batch):
        return batch.to(device=device)
    if is_pytorch3d_camera_batch(batch):
        return batch.to(device=device)
    raise TypeError(f"Unsupported camera batch type: {type(batch).__name__}")


def camera_batch_image_size(batch: CameraBatchLike) -> torch.Tensor:
    """Return image sizes for either camera representation."""

    if is_native_camera_batch(batch):
        return batch.image_size
    if is_pytorch3d_camera_batch(batch):
        image_size = batch.image_size
        if image_size is None:
            raise RuntimeError("PerspectiveCameras.image_size is required.")
        return image_size
    raise TypeError(f"Unsupported camera batch type: {type(batch).__name__}")


def camera_batch_size(batch: CameraBatchLike) -> int:
    """Return the number of cameras in the batch."""

    if is_native_camera_batch(batch):
        return batch.batch_size
    if is_pytorch3d_camera_batch(batch):
        return int(batch.R.shape[0])
    raise TypeError(f"Unsupported camera batch type: {type(batch).__name__}")


def camera_batch_select(batch: CameraBatchLike, index: int) -> CameraBatchLike:
    """Select one camera entry by index."""

    if is_native_camera_batch(batch):
        return batch.select(index)
    if is_pytorch3d_camera_batch(batch):
        return batch[index]
    raise TypeError(f"Unsupported camera batch type: {type(batch).__name__}")


def require_pytorch3d_camera_batch(batch: CameraBatchLike) -> PerspectiveCameras:
    """Return the PyTorch3D batch or raise for native batches."""

    if is_pytorch3d_camera_batch(batch):
        return batch
    raise TypeError("This code path requires a PyTorch3D PerspectiveCameras batch.")


__all__ = [
    "CameraBatchBackend",
    "CameraBatchLike",
    "NativeCameraBatch",
    "camera_batch_image_size",
    "camera_batch_select",
    "camera_batch_size",
    "camera_batch_to",
    "is_native_camera_batch",
    "is_pytorch3d_camera_batch",
    "require_pytorch3d_camera_batch",
]
