"""Typed tensor, pose, camera, and image conversion helpers for Rerun logging."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from efm3d.aria.pose import PoseTW
from efm3d.aria.tensor_wrapper import TensorWrapper
from torch import Tensor

if TYPE_CHECKING:
    from efm3d.aria.camera import CameraTW
    from numpy.typing import DTypeLike, NDArray
    from pytorch3d.renderer.cameras import PerspectiveCameras


def to_numpy(value: object, *, dtype: DTypeLike = np.float32) -> NDArray[Any]:
    """Convert tensors and tensor-wrapper payloads to NumPy arrays."""

    if isinstance(value, TensorWrapper):
        value = value._data
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def deterministic_downsample(points: object, *, max_points: int, seed: int | None) -> NDArray[Any]:
    """Return a deterministic subset of ``points`` with shape ``(N, 3)``."""

    arr = to_numpy(points).reshape(-1, 3)
    if max_points <= 0:
        return arr[:0]
    if arr.shape[0] <= max_points:
        return arr
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(arr.shape[0], size=max_points, replace=False))
    return arr[indices]


def pose_rt(poses: PoseTW, indices: Sequence[int] | None = None) -> tuple[NDArray[Any], NDArray[Any]]:
    """Extract ``R`` and ``t`` from a PoseTW-like batch."""

    r = to_numpy(poses.R).reshape(-1, 3, 3)
    t = to_numpy(poses.t).reshape(-1, 3)
    if indices is not None:
        idx = np.asarray(indices, dtype=np.int64)
        r = r[idx]
        t = t[idx]
    return r, t


def p3d_param_at(values: Tensor, index: int) -> NDArray[Any]:
    """Return one PyTorch3D camera parameter row as ``float32``."""

    arr = to_numpy(values).reshape(-1, values.shape[-1])
    if arr.shape[0] == 0:
        raise ValueError("PyTorch3D camera parameter batch is empty.")
    row = arr[0] if arr.shape[0] == 1 else arr[min(max(index, 0), arr.shape[0] - 1)]
    return np.asarray(row, dtype=np.float32)


def p3d_pinhole_kwargs(cameras: PerspectiveCameras, index: int) -> dict[str, list[float]]:
    """Return Rerun ``Pinhole`` kwargs from a PyTorch3D camera entry.

    PyTorch3D stores ``image_size`` as ``(height, width)``; Rerun expects
    ``resolution`` as ``[width, height]``.
    """

    image_size = p3d_param_at(cameras.image_size, index)
    focal = p3d_param_at(cameras.focal_length, index)
    principal = p3d_param_at(cameras.principal_point, index)
    height, width = float(image_size[0]), float(image_size[1])
    return {
        "resolution": [width, height],
        "focal_length": [float(focal[0]), float(focal[1])],
        "principal_point": [float(principal[0]), float(principal[1])],
    }


def camera_tw_pinhole_kwargs(camera: CameraTW) -> dict[str, list[float]]:
    """Return Rerun ``Pinhole`` kwargs from one EFM ``CameraTW`` entry."""

    size = to_numpy(camera.size).reshape(-1, 2)[0]
    focal = to_numpy(camera.f).reshape(-1, 2)[0]
    principal = to_numpy(camera.c).reshape(-1, 2)[0]
    return {
        "resolution": [float(size[0]), float(size[1])],
        "focal_length": [float(focal[0]), float(focal[1])],
        "principal_point": [float(principal[0]), float(principal[1])],
    }


def candidate_centers_world(poses_world_cam: PoseTW, indices: Sequence[int]) -> NDArray[Any]:
    """Return candidate camera centers from PoseTW translations."""

    _, centers = pose_rt(poses_world_cam, indices)
    return centers


def subset_poses(poses_world_cam: PoseTW, indices: Sequence[int]) -> PoseTW:
    """Return a PoseTW-like subset without importing data-handling internals."""

    data = poses_world_cam._data
    if data is None:
        raise ValueError("PoseTW payload is empty; cannot subset candidate poses.")
    data = data.reshape(-1, 12)
    index = torch.as_tensor(list(indices), device=data.device, dtype=torch.long)
    return cast("PoseTW", PoseTW(data.index_select(0, index)))


def image_hwc(tensor: object, index: int) -> NDArray[Any]:
    """Convert a CHW image tensor in [0,1] or [0,255] to HWC uint8."""

    arr = to_numpy(tensor)
    if arr.ndim == 4:
        arr = arr[index]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0 if float(np.nanmax(arr)) <= 1.0 else 255.0)
        if float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)
    return arr


def depth_hw(tensor: object, index: int) -> NDArray[Any]:
    """Return one depth frame as ``float32`` with shape ``(H, W)``."""

    arr = to_numpy(tensor)
    if arr.ndim == 4:
        arr = arr[index, 0]
    elif arr.ndim == 3:
        arr = arr[index]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
    return np.asarray(arr, dtype=np.float32)


def display_rot90_cw(array: NDArray[Any]) -> NDArray[Any]:
    """Apply ARIA's display-only 90 degree clockwise image convention."""

    return np.ascontiguousarray(np.rot90(array, k=-1))


__all__ = [
    "camera_tw_pinhole_kwargs",
    "candidate_centers_world",
    "depth_hw",
    "deterministic_downsample",
    "display_rot90_cw",
    "image_hwc",
    "p3d_param_at",
    "p3d_pinhole_kwargs",
    "pose_rt",
    "subset_poses",
    "to_numpy",
]
