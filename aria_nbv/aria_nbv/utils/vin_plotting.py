"""Shared VIN plotting helpers.

This module centralizes camera, pose, voxel, and lightweight model-summary
helpers that were previously embedded inside `aria_nbv.vin.plotting`. Keeping
them here lets figure-builder modules stay focused on composing figures while
the lower-level geometry and formatting utilities live in one reusable owner.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.pose import PoseTW
from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]

from ..data.plotting import SnippetPlotBuilder
from .frames import rotate_yaw_cw90

Tensor = torch.Tensor

BBOX_EDGE_IDX = np.array(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ],
    dtype=np.int64,
)
"""Canonical bounding-box edge connectivity for 8-corner voxel boxes."""


def parameter_distribution(
    model: torch.nn.Module,
    *,
    trainable_only: bool = True,
) -> pd.DataFrame:
    """Aggregate parameter counts by top-level module name.

    Args:
        model: Module whose parameters should be summarized.
        trainable_only: Whether to exclude frozen parameters.

    Returns:
        Dataframe with one row per top-level module and a `num_params` column.
    """

    rows: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        module = name.split(".", 1)[0]
        rows.append({"module": module, "num_params": int(param.numel())})
    if not rows:
        return pd.DataFrame(columns=["module", "num_params"])
    df = pd.DataFrame(rows)
    df = df.groupby("module", as_index=False)["num_params"].sum()
    return df.sort_values("num_params", ascending=False)


def voxel_corners(extent: np.ndarray) -> np.ndarray:
    """Return the 8 axis-aligned box corners for a voxel extent."""

    x_min, x_max, y_min, y_max, z_min, z_max = extent.tolist()
    return np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        dtype=float,
    )


def pose_first_batch(pose: PoseTW) -> PoseTW:
    """Select the first batch element while preserving `PoseTW` shape semantics."""

    if pose.ndim == 1:
        return PoseTW(pose._data.unsqueeze(0))
    if pose.ndim == 2 and pose.shape[0] > 1:
        return PoseTW(pose._data[:1])
    return pose


def as_pose_tw(pose: PoseTW | torch.Tensor) -> PoseTW:
    """Convert a tensor pose representation into `PoseTW` if needed."""

    if isinstance(pose, PoseTW):
        return pose
    if torch.is_tensor(pose):
        data = pose
        if data.shape[-1] == 12:
            data = data.view(*data.shape[:-1], 3, 4)
        return PoseTW.from_matrix3x4(data)
    raise TypeError(f"Unsupported pose type: {type(pose)!s}")


def as_pose_batch(pose: PoseTW | torch.Tensor) -> PoseTW:
    """Ensure poses are batched as `(B, ..., 12)` for plotting helpers."""

    pose_tw = as_pose_tw(pose)
    if pose_tw.ndim == 1:
        return PoseTW(pose_tw._data.unsqueeze(0))
    if pose_tw.ndim == 2 and pose_tw.shape[-1] == 12:
        return PoseTW(pose_tw._data.unsqueeze(0))
    return pose_tw


def broadcast_pose_batch(pose: PoseTW, *, batch_size: int, name: str) -> PoseTW:
    """Broadcast a single-pose batch to `batch_size` if needed."""

    if pose.ndim != 2:
        raise ValueError(f"{name} must have shape (B, 12), got ndim={pose.ndim}.")
    if pose.shape[0] == 1 and batch_size > 1:
        return PoseTW(pose._data.expand(batch_size, 12))
    if pose.shape[0] != batch_size:
        raise ValueError(f"{name} must have batch size 1 or match candidates.")
    return pose


def centers_rig_from_poses(
    reference_pose_world_rig: PoseTW,
    candidate_poses_world_cam: PoseTW,
) -> torch.Tensor:
    """Compute candidate centers in the reference rig frame."""

    pose_world_cam = as_pose_batch(candidate_poses_world_cam)
    pose_world_rig = as_pose_batch(reference_pose_world_rig)
    pose_world_rig = broadcast_pose_batch(
        pose_world_rig,
        batch_size=int(pose_world_cam.shape[0]),
        name="reference_pose_world_rig",
    )
    pose_rig_cam = pose_world_rig.inverse()[:, None] @ pose_world_cam
    return pose_rig_cam.t


def candidate_valid_fraction(debug: Any) -> torch.Tensor:
    """Return per-candidate validity fractions across VIN variants."""

    token_valid = getattr(debug, "token_valid", None)
    if isinstance(token_valid, torch.Tensor):
        return token_valid.float().mean(dim=-1)

    voxel_valid_frac = getattr(debug, "voxel_valid_frac", None)
    if isinstance(voxel_valid_frac, torch.Tensor):
        return voxel_valid_frac.float()

    candidate_valid = getattr(debug, "candidate_valid", None)
    if isinstance(candidate_valid, torch.Tensor):
        return candidate_valid.float()

    centers = debug.candidate_center_rig_m
    return torch.ones(centers.shape[:-1], dtype=torch.float32, device=centers.device)


def rotate_points_yaw_cw90(
    points: np.ndarray | torch.Tensor,
    *,
    pose_world_frame: PoseTW | None = None,
    undo: bool = False,
) -> np.ndarray | torch.Tensor:
    """Rotate world points by the UI roll used in VIN display plots."""

    if isinstance(points, np.ndarray):
        if points.size == 0:
            return points
        pts = torch.as_tensor(points, dtype=torch.float32)
        to_numpy = True
    else:
        pts = points
        to_numpy = False
    if pts.numel() == 0:
        return points

    angle = -np.pi / 2 if undo else np.pi / 2
    c, s = float(np.cos(angle)), float(np.sin(angle))
    r_roll = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        device=pts.device,
        dtype=pts.dtype,
    )
    if pose_world_frame is None:
        rot = PoseTW.from_Rt(
            r_roll,
            torch.zeros(3, device=pts.device, dtype=pts.dtype),
        )
        rotated = rot.transform(pts)
    else:
        pose_rot = rotate_yaw_cw90(pose_world_frame, undo=undo)
        pts_frame = pose_world_frame.inverse().transform(pts)
        rotated = pose_rot.transform(pts_frame)
    return rotated.detach().cpu().numpy() if to_numpy else rotated


def collect_backbone_evidence_points(
    debug: Any,
    *,
    fields: list[str],
    occ_threshold: float,
    max_points: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Extract world-space evidence point samples from backbone voxel fields."""

    if not fields:
        return []

    out = debug.backbone_out
    voxel_fields = {
        "occ_pr": out.occ_pr,
        "occ_input": out.occ_input,
        "counts": out.counts,
    }
    selected = [name for name in fields if voxel_fields.get(name) is not None]
    if not selected:
        return []

    pose = pose_first_batch(out.t_world_voxel)
    extent = out.voxel_extent.detach().cpu().numpy()
    if extent.ndim == 2:
        extent = extent[0]

    samples: list[tuple[str, np.ndarray, np.ndarray]] = []
    pts_world = out.pts_world
    pts_world_np: np.ndarray | None = None
    if isinstance(pts_world, torch.Tensor):
        pts_world = pts_world.detach().cpu()
        if pts_world.ndim == 3:
            pts_world = pts_world[0]
        if pts_world.ndim == 2 and pts_world.shape[1] == 3:
            pts_world_np = pts_world.numpy()

    for name in selected:
        tensor = voxel_fields[name]
        if tensor is None:
            continue
        field = tensor.detach().cpu()
        if field.ndim == 5:
            field = field[0, 0]
        elif field.ndim == 4:
            field = field[0]
        if name == "occ_pr":
            finite = field[torch.isfinite(field)]
            if finite.numel() > 0:
                min_val = float(finite.min().item())
                max_val = float(finite.max().item())
                if min_val < 0.0 or max_val > 1.0:
                    field = torch.sigmoid(field)
        d, h, w = field.shape

        if name == "counts":
            flat = field.reshape(-1)
            if flat.numel() == 0:
                continue
            topk = min(int(max_points), flat.numel())
            _, idx = torch.topk(flat, k=topk)
            indices = np.stack(np.unravel_index(idx.numpy(), field.shape), axis=-1)
            values = flat[idx].numpy()
        else:
            mask = field > float(occ_threshold)
            indices = np.stack(np.nonzero(mask.numpy()), axis=-1)
            values = field[mask].numpy()
            if indices.size == 0:
                flat = field.reshape(-1)
                if flat.numel() == 0:
                    continue
                topk = min(int(max_points), flat.numel())
                _, idx = torch.topk(flat, k=topk)
                indices = np.stack(np.unravel_index(idx.numpy(), field.shape), axis=-1)
                values = flat[idx].numpy()
            if indices.shape[0] > max_points:
                sel = np.random.choice(indices.shape[0], size=max_points, replace=False)
                indices = indices[sel]
                values = values[sel]

        if indices.size == 0:
            continue

        points_world = None
        if pts_world_np is not None:
            flat_idx = indices[:, 0] * (h * w) + indices[:, 1] * w + indices[:, 2]
            flat_idx = np.clip(flat_idx, 0, pts_world_np.shape[0] - 1)
            points_world = pts_world_np[flat_idx]
        if points_world is None:
            points_world = voxel_indices_to_world(indices, pose=pose, extent=extent, shape=(d, h, w))
        values_np = np.asarray(values)
        finite = np.isfinite(points_world).all(axis=1)
        if values_np.shape[0] == points_world.shape[0]:
            values_np = values_np[finite]
        points_world = points_world[finite]
        if points_world.size == 0:
            continue
        samples.append((name, points_world, values_np))

    return samples


def select_p3d_camera(cameras: PerspectiveCameras, index: int) -> PerspectiveCameras:
    """Select one camera from a possibly batched `PerspectiveCameras` object."""

    num_cams = int(cameras.R.shape[0]) if cameras.R is not None else 0
    if num_cams == 0:
        return cameras

    in_ndc = getattr(cameras, "in_ndc", False)
    if callable(in_ndc):
        in_ndc = in_ndc()
    needs_rebuild = not hasattr(cameras, "_in_ndc")

    def ensure_batch(param: torch.Tensor | None) -> torch.Tensor | None:
        if param is None:
            return None
        if param.ndim == 1:
            return param.unsqueeze(0)
        return param

    if num_cams > 1 or needs_rebuild:
        idx = min(int(index), num_cams - 1)
        rot = cameras.R[idx : idx + 1] if num_cams > 1 else cameras.R
        trans = cameras.T[idx : idx + 1] if num_cams > 1 else cameras.T
        cameras = PerspectiveCameras(
            device=cameras.device,
            R=ensure_batch(rot),
            T=ensure_batch(trans),
            focal_length=ensure_batch(cameras.focal_length[idx : idx + 1] if num_cams > 1 else cameras.focal_length),
            principal_point=ensure_batch(
                cameras.principal_point[idx : idx + 1] if num_cams > 1 else cameras.principal_point,
            ),
            image_size=ensure_batch(cameras.image_size[idx : idx + 1] if num_cams > 1 else cameras.image_size)
            if cameras.image_size is not None
            else None,
            in_ndc=bool(in_ndc),
        )
    return cameras


def voxel_indices_to_world(
    indices: np.ndarray,
    *,
    pose: PoseTW,
    extent: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Map voxel indices to world-space centers using a voxel pose and extent."""

    d, h, w = shape
    x_min, x_max, y_min, y_max, z_min, z_max = extent.tolist()

    dx = (x_max - x_min) / w
    dy = (y_max - y_min) / h
    dz = (z_max - z_min) / d

    z_idx, y_idx, x_idx = indices[:, 0], indices[:, 1], indices[:, 2]
    x = x_min + (x_idx + 0.5) * dx
    y = y_min + (y_idx + 0.5) * dy
    z = z_min + (z_idx + 0.5) * dz
    points_vox = torch.tensor(
        np.stack([x, y, z], axis=-1),
        dtype=torch.float32,
        device=pose.t.device,
    )
    points_world = pose.transform(points_vox)
    return points_world.detach().cpu().numpy()


def voxel_indices_to_world_from_cache(
    indices: np.ndarray,
    *,
    pose: PoseTW,
    extent: np.ndarray,
    shape: tuple[int, int, int],
    pts_world: torch.Tensor | None,
) -> np.ndarray:
    """Map voxel indices to world points using cached voxel centers when present."""

    if torch.is_tensor(pts_world):
        pts = pts_world.detach().cpu()
        if pts.ndim == 3:
            pts = pts[0]
        if pts.ndim == 2 and pts.shape[-1] == 3:
            d, h, w = shape
            if pts.numel() == d * h * w * 3:
                pts_grid = pts.reshape(d, h, w, 3)
                sel = pts_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
                return sel.numpy()
    return voxel_indices_to_world(indices, pose=pose, extent=extent, shape=shape)


def select_p3d_param(param: torch.Tensor | None, index: int) -> torch.Tensor | None:
    """Select one row from a possibly batched PyTorch3D camera parameter tensor."""

    if param is None:
        return None
    if param.ndim == 1:
        return param
    if param.shape[0] == 1:
        return param[0]
    return param[min(max(index, 0), param.shape[0] - 1)]


def camera_tw_from_p3d(cameras: PerspectiveCameras, index: int) -> CameraTW | None:
    """Convert one PyTorch3D camera entry into an `efm3d` `CameraTW`."""

    if int(cameras.R.shape[0]) == 0:
        return None
    focal = select_p3d_param(cameras.focal_length, index)
    principal = select_p3d_param(cameras.principal_point, index)
    image_size = select_p3d_param(cameras.image_size, index)
    if focal is None or principal is None or image_size is None:
        return None

    focal = focal.reshape(-1)
    principal = principal.reshape(-1)
    image_size = image_size.reshape(-1)
    if focal.numel() != 2 or principal.numel() != 2 or image_size.numel() != 2:
        return None

    height = image_size[0].reshape(1)
    width = image_size[1].reshape(1)
    params = torch.stack((focal[0], focal[1], principal[0], principal[1]), dim=0)

    return CameraTW.from_surreal(
        width=width,
        height=height,
        type_str="Pinhole",
        params=params,
    )


def pose_from_p3d_camera(cameras: PerspectiveCameras, index: int) -> PoseTW | None:
    """Convert one PyTorch3D camera entry into a world-camera `PoseTW`."""

    if int(cameras.R.shape[0]) == 0:
        return None
    rot = select_p3d_param(cameras.R, index)
    trans = select_p3d_param(cameras.T, index)
    if rot is None or trans is None:
        return None
    rot = rot.reshape(3, 3)
    trans = trans.reshape(3)
    t_wc = -(rot @ trans.unsqueeze(-1)).squeeze(-1)
    return PoseTW.from_Rt(rot, t_wc)


def frustum_builder_stub(
    pose_world_cam: PoseTW | None,
    *,
    title: str,
    height: int = 560,
) -> SnippetPlotBuilder | None:
    """Construct a minimal snippet-like stub so frustum plots can reuse the builder."""

    if pose_world_cam is None:
        return None
    snippet_stub = SimpleNamespace(
        trajectory=SimpleNamespace(t_world_rig=pose_world_cam),
        mesh=None,
        semidense=None,
    )
    return SnippetPlotBuilder.from_snippet(
        snippet_stub,  # type: ignore[arg-type]
        title=title,
        height=height,
    )


__all__ = [
    "BBOX_EDGE_IDX",
    "as_pose_batch",
    "as_pose_tw",
    "broadcast_pose_batch",
    "camera_tw_from_p3d",
    "candidate_valid_fraction",
    "centers_rig_from_poses",
    "collect_backbone_evidence_points",
    "frustum_builder_stub",
    "parameter_distribution",
    "pose_first_batch",
    "pose_from_p3d_camera",
    "rotate_points_yaw_cw90",
    "select_p3d_camera",
    "select_p3d_param",
    "voxel_corners",
    "voxel_indices_to_world",
    "voxel_indices_to_world_from_cache",
]
