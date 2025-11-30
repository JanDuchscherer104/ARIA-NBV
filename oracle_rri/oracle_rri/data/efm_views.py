"""Typed, zero-copy views over EFM-formatted ATEK samples.

These classes mirror the style of :mod:`oracle_rri.data.views` but read the
keys produced by ``efm3d.dataset.efm_model_adaptor.load_atek_wds_dataset_as_efm``.
All properties surface rich shape/type information to make downstream use
explicit and safe.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pprint import pformat
from typing import Any, Literal, TypedDict

import numpy as np
import torch
from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_DEPTH_TIME_NS,
    ARIA_DISTANCE_M,
    ARIA_FRAME_ID,
    ARIA_IMG,
    ARIA_IMG_TIME_NS,
    ARIA_OBB_FREQUENCY_HZ,
    ARIA_OBB_PADDED,
    ARIA_POINTS_DIST_STD,
    ARIA_POINTS_INV_DIST_STD,
    ARIA_POINTS_TIME_NS,
    ARIA_POINTS_VOL_MAX,
    ARIA_POINTS_VOL_MIN,
    ARIA_POINTS_WORLD,
    ARIA_POSE_T_WORLD_RIG,
    ARIA_POSE_TIME_NS,
)
from efm3d.aria.camera import CameraTW
from efm3d.aria.obb import ObbTW
from efm3d.aria.pose import PoseTW
from efm3d.aria.tensor_wrapper import TensorWrapper
from efm3d.utils.pointcloud import collapse_pointcloud_time
from torch import Tensor
from trimesh import Trimesh

from oracle_rri.data.mesh_cache import MeshProcessSpec, mesh_from_snippet

from ..utils import summarize


def _repr(obj: Any) -> str:
    return pformat({f.name: summarize(getattr(obj, f.name)) for f in fields(obj)}, indent=2, width=100, compact=False)


# ---------------------------------------------------------------------------
# GT views (EFM-format)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EfmGtCameraObbView:
    """Per-camera OBB ground truth for a single timestamp.

    Attributes:
        category_names: list[str]
            Human-readable labels (EFM43 class set; see efm3d.qmd and glossary).
        category_ids: Tensor["K", int64]
            Semantic ids aligned with `category_names`.
        instance_ids: Tensor["K", int64]
            Instance ids consistent across cameras at this timestamp.
        object_dimensions: Tensor["K 3", float32]
            Box side lengths (x, y, z) in metres.
        ts_world_object: Tensor["K 3 4", float32]
            Pose matrices (world←object) per instance.
    """

    category_names: list[str]
    category_ids: Tensor
    instance_ids: Tensor
    object_dimensions: Tensor
    ts_world_object: Tensor

    def __repr__(self) -> str:  # pragma: no cover
        return _repr(self)


CamerasDict = TypedDict(
    "CamerasDict",
    {"rgb": EfmGtCameraObbView, "slam-left": EfmGtCameraObbView, "slam-right": EfmGtCameraObbView},
)


@dataclass(slots=True)
class EfmGtTimestampView:
    """EFM GT slice for one timestamp across cameras."""

    time_id: str
    cameras: CamerasDict

    def __repr__(self) -> str:  # pragma: no cover
        return _repr(self)


@dataclass(slots=True)
class EfmGTView:
    """Ground-truth annotations (EFM schema) for a snippet."""

    raw: dict[str, Any]
    efm_gt: dict[str, EfmGtTimestampView]

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw or {}
        efm_raw = self.raw.get("efm_gt") or {}
        parsed: dict[str, EfmGtTimestampView] = {}
        for ts_key, cams in efm_raw.items():
            if not isinstance(cams, dict):
                continue
            cam_views: dict[str, EfmGtCameraObbView] = {}
            for cam_id, cam_dict in cams.items():
                if not isinstance(cam_dict, dict):
                    continue
                cam_views[cam_id] = EfmGtCameraObbView(
                    category_names=cam_dict.get("category_names", []),
                    category_ids=cam_dict.get("category_ids"),  # type: ignore[arg-type]
                    instance_ids=cam_dict.get("instance_ids"),  # type: ignore[arg-type]
                    object_dimensions=cam_dict.get("object_dimensions"),  # type: ignore[arg-type]
                    ts_world_object=cam_dict.get("ts_world_object"),  # type: ignore[arg-type]
                )
            parsed[str(ts_key)] = EfmGtTimestampView(time_id=str(ts_key), cameras=cam_views)
        self.efm_gt = parsed

    @property
    def timestamps(self) -> list[str]:
        """Sorted list of efm_gt timestamp keys."""

        return sorted(self.efm_gt.keys())

    def cameras_at(self, ts: str | int) -> dict[str, EfmGtCameraObbView]:
        """Get per-camera OBB GT at a timestamp."""

        key = str(ts)
        if key not in self.efm_gt:
            raise KeyError(f"No efm_gt entry for timestamp {ts}")
        return self.efm_gt[key].cameras

    def __repr__(self) -> str:  # pragma: no cover
        return _repr(self)


@dataclass(slots=True)
class EfmCameraView:
    """Camera stream in EFM schema.

    Attributes:
        images: ``Tensor["F C H W", float32]``
            - RGB or mono images normalised to ``[0, 1]``.
            - ``F`` frames at the fixed snippet length (usually 20 for 2 s @10 Hz).
        calib: :class:`CameraTW`
            - Batched intrinsics/extrinsics (fisheye624 params, valid radius, exposure, gain).
            - ``CameraTW.tensor`` shape ``(F, 34)`` storing projection params and pose.
        time_ns: ``Tensor["F", int64]`` capture timestamps (device clock).
        frame_ids: ``Tensor["F", float32|int64]`` per-frame ids.
        distance_m: Optional ``Tensor["F 1 H W", float32]`` ray distances (metric depth).
        distance_time_ns: Optional ``Tensor["F", int64]`` timestamps aligned to ``distance_m``.
    """

    images: Tensor
    """``Tensor["F C H W", float32]`` normalized camera images in Aria LUF frame."""
    calib: CameraTW
    """Per-frame camera intrinsics/extrinsics (`CameraTW.tensor` shape ``(F,34)``)."""
    time_ns: Tensor
    """``Tensor["F", int64]`` device timestamps aligned to `images`."""
    frame_ids: Tensor
    """``Tensor["F", int64|float32]`` frame ids within the snippet."""
    distance_m: Tensor | None = None
    """Optional metric ray distances ``Tensor["F 1 H W", float32]``."""
    distance_time_ns: Tensor | None = None
    """Optional ``Tensor["F", int64]`` timestamps for ``distance_m``."""

    def to(self, device: str | torch.device, *, dtype: torch.dtype | None = None) -> "EfmCameraView":
        target_device = torch.device(device)
        if self.images.device.type == target_device.type and dtype is None:
            return self  # no-op, keep zero-copy view
        return replace(
            self,
            images=self.images.to(device=target_device, dtype=dtype),
            calib=self.calib.to(device=target_device),  # type: ignore[arg-type]
            time_ns=self.time_ns.to(target_device),
            frame_ids=self.frame_ids.to(target_device),
            distance_m=None if self.distance_m is None else self.distance_m.to(device=target_device, dtype=dtype),
            distance_time_ns=None if self.distance_time_ns is None else self.distance_time_ns.to(target_device),
        )

    def get_fov(self) -> Tensor:
        """Tensor["F 2", float32] FOV in degrees (fov_x, fov_y) per frame."""
        size = self.calib.size  # (..., 2) = (F,2)
        focals = self.calib.f  # (..., 2)
        width = size[..., 0].to(dtype=torch.float32)
        height = size[..., 1].to(dtype=torch.float32)
        fx = focals[..., 0].to(dtype=torch.float32)
        fy = focals[..., 1].to(dtype=torch.float32)

        valid = (width > 0) & (height > 0) & (fx > 0) & (fy > 0)
        fov_x = torch.full_like(width, float("nan"))
        fov_y = torch.full_like(height, float("nan"))
        fov_x = torch.where(valid, torch.rad2deg(2 * torch.atan(width / (2 * fx))), fov_x)
        fov_y = torch.where(valid, torch.rad2deg(2 * torch.atan(height / (2 * fy))), fov_y)

        return torch.stack([fov_x, fov_y], dim=-1)

    @property
    def num_frames(self) -> int:
        return self.images.shape[0]

    def select_frame_indices(self, frame_indices: list[int] | None, *, default_last: bool) -> torch.Tensor:
        """Resolve user-provided frame indices, supporting negatives and defaults."""

        n_cam = self.num_frames
        if n_cam == 0:
            return torch.tensor([], device=self.time_ns.device, dtype=torch.long)
        if frame_indices is None or len(frame_indices) == 0:
            frame_indices = [0, n_cam - 1] if default_last and n_cam > 1 else [0]
        idx = torch.as_tensor(frame_indices, device=self.time_ns.device, dtype=torch.long)
        idx = torch.where(idx < 0, idx + n_cam, idx)
        return idx.clamp(0, n_cam - 1)

    def nearest_traj_indices(
        self,
        traj_ts_ns: torch.Tensor,
        frame_indices: list[int] | torch.Tensor | None = None,
        *,
        default_last: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return selected camera indices and nearest trajectory indices."""

        cam_idx = (
            frame_indices
            if isinstance(frame_indices, torch.Tensor)
            else self.select_frame_indices(frame_indices, default_last=default_last)
        )
        if cam_idx.numel() == 0 or traj_ts_ns.numel() == 0:
            empty = torch.tensor([], device=self.time_ns.device, dtype=torch.long)
            return cam_idx, empty

        cam_ts = self.time_ns.to(cam_idx.device)
        traj_ts = traj_ts_ns.to(cam_idx.device)
        cam_sel_ts = cam_ts[cam_idx]
        dt = (traj_ts.unsqueeze(1) - cam_sel_ts.unsqueeze(0)).abs()
        traj_idx = torch.argmin(dt, dim=0)
        return cam_idx, traj_idx

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        return _repr(self)


@dataclass(slots=True)
class EfmTrajectoryView:
    """Rig trajectory in world frame.

    Attributes:
        t_world_rig: ``PoseTW`` (internally ``Tensor["F 3 4", float32]``) world←rig per frame.
        time_ns: ``Tensor["F", int64]`` timestamps matching poses.
        gravity_in_world: ``Tensor["3", float32]`` gravity vector aligned to EFM convention ``[0, 0, -9.81]``.
    """

    t_world_rig: PoseTW
    """Rig SE(3) poses per frame (world←rig)."""
    time_ns: Tensor
    """``Tensor["F", int64]`` pose timestamps."""
    gravity_in_world: Tensor
    """``Tensor["3", float32]`` gravity vector in world frame (aligned to [0,0,-9.81])."""

    @property
    def final_pose(self) -> PoseTW:
        """Final rig pose in snippet. world ← rig."""

        return PoseTW.from_matrix3x4(self.t_world_rig.matrix3x4[-1])

    def to(self, device: str | torch.device, *, dtype: torch.dtype | None = None) -> "EfmTrajectoryView":
        target_device = torch.device(device)
        if self.time_ns.device.type == target_device.type and dtype is None:
            return self
        return replace(
            self,
            t_world_rig=self.t_world_rig.to(device=target_device),  # type: ignore[arg-type]
            time_ns=self.time_ns.to(target_device),
            gravity_in_world=self.gravity_in_world.to(device=target_device, dtype=dtype),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return _repr(self)


@dataclass(slots=True)
class EfmPointsView:
    """Semi-dense SLAM points (padded).

    Attributes:
        points_world: ``Tensor["F N 3", float32]``
            - World-frame points padded to ``N=semidense_points_pad (default 50000)``.
        dist_std: ``Tensor["F N", float32]`` distance standard deviation per point.
        inv_dist_std: ``Tensor["F N", float32]`` inverse distance std per point.
        time_ns: ``Tensor["F", int64]`` timestamps aligned to frames.
        volume_min: ``Tensor["3", float32]`` world AABB minimum for the snippet.
        volume_max: ``Tensor["3", float32]`` world AABB maximum for the snippet.
        lengths: ``Tensor["F", int64]`` true point counts before padding.
    """

    points_world: Tensor
    """``Tensor["F N 3", float32]`` padded world-frame SLAM points (meters)."""
    dist_std: Tensor
    """``Tensor["F N", float32]`` per-point distance std (depth uncertainty)."""
    inv_dist_std: Tensor
    """``Tensor["F N", float32]`` per-point inverse distance std."""
    time_ns: Tensor
    """``Tensor["F", int64]`` timestamps aligned to point slices."""
    volume_min: Tensor
    """``Tensor["3", float32]`` snippet AABB minimum in world coords."""
    volume_max: Tensor
    """``Tensor["3", float32]`` snippet AABB maximum in world coords."""
    lengths: Tensor
    """``Tensor["F", int64]`` true (unpadded) point counts per frame."""

    def to(self, device: str | torch.device, *, dtype: torch.dtype | None = None) -> "EfmPointsView":
        target_device = torch.device(device)
        if self.points_world.device.type == target_device.type and dtype is None:
            return self
        return replace(
            self,
            points_world=self.points_world.to(device=target_device, dtype=dtype),
            dist_std=self.dist_std.to(device=target_device, dtype=dtype),
            inv_dist_std=self.inv_dist_std.to(device=target_device, dtype=dtype),
            time_ns=self.time_ns.to(target_device),
            volume_min=self.volume_min.to(device=target_device, dtype=dtype),
            volume_max=self.volume_max.to(device=target_device, dtype=dtype),
            lengths=self.lengths.to(target_device),
        )

    def collapse_points_np(self, max_points: int | None = None) -> np.ndarray:
        """Collapse points across time and optionally subsample for plotting."""

        points = self.points_world
        if points.numel() == 0:
            return np.zeros((0, 3), dtype=np.float32)

        lengths = self.lengths.to(device=points.device)
        max_len = points.shape[1]
        valid_mask = torch.arange(max_len, device=points.device).unsqueeze(0) < lengths.clamp_max(max_len).unsqueeze(-1)
        points_collapsed = collapse_pointcloud_time(torch.where(valid_mask.unsqueeze(-1), points, torch.nan))
        if points_collapsed.numel() == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if max_points is not None and points_collapsed.shape[0] > max_points:
            idx = torch.randperm(points_collapsed.shape[0], device=points_collapsed.device)[:max_points]
            points_collapsed = points_collapsed[idx]

        return points_collapsed.detach().cpu().numpy()

    def last_frame_points_np(self, max_points: int | None = None) -> np.ndarray:
        """Return points from the latest frame with valid points, subsampled if needed."""

        points = self.points_world
        lengths = self.lengths.to(device=points.device)
        if points.numel() == 0 or lengths.numel() == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if torch.any(lengths > 0):
            last_idx = int(torch.nonzero(lengths > 0, as_tuple=False).max().item())
        else:
            return np.zeros((0, 3), dtype=np.float32)

        n_valid = min(int(lengths[last_idx].item()), points.shape[1])
        if n_valid == 0:
            return np.zeros((0, 3), dtype=np.float32)

        pts = points[last_idx, :n_valid]
        finite = torch.isfinite(pts).all(dim=-1)
        pts = pts[finite]
        if pts.numel() == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if max_points is not None and pts.shape[0] > max_points:
            idx = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
            pts = pts[idx]

        return pts.detach().cpu().numpy()

    def __repr__(self) -> str:  # pragma: no cover
        return _repr(self)


@dataclass(slots=True)
class EfmObbView:
    """Snippet-level oriented bounding boxes (OBB) in snippet frame.

    Attributes:
        obbs: :class:`ObbTW`
            - Padded OBB tensor in snippet coordinates (internally ``Tensor["K_pad 14", float32]``).
            - Contains center, sizes, rotation, class ids as per EFM3D.
        hz: ``Tensor["1", int32]`` capture frequency metadata when present.
    """

    obbs: ObbTW
    """Padded oriented boxes in snippet frame (center, size, yaw, sem-id)."""
    hz: Tensor | None = None
    """Optional ``Tensor["1", int32]`` detection frequency."""

    def __repr__(self) -> str:  # pragma: no cover
        return _repr(self)


@dataclass(slots=True)
class EfmSnippetView:
    """Typed wrapper over an EFM-formatted sample plus optional mesh."""

    efm: dict[str, Any]
    """Backing EFM sample dictionary (zero-copy)."""
    scene_id: str
    """ASE scene identifier (e.g., ``81283``)."""
    snippet_id: str
    """Snippet/shard identifier (e.g., ``shards-0000``)."""
    mesh: Trimesh | None = None
    """Optional GT mesh paired with this sample."""
    crop_bounds: tuple[torch.Tensor, torch.Tensor] | None = None
    """Optional `(min, max)` world-space AABB used for mesh cropping / occupancy."""

    mesh_verts: torch.Tensor | None = None
    """Optional cached mesh vertices tensor (float32, device-agnostic)."""
    mesh_faces: torch.Tensor | None = None
    """Optional cached mesh faces tensor (int64)."""
    mesh_cache_key: str | None = None
    """Stable key (spec hash) for shared mesh caches across components."""

    mesh_specs: MeshProcessSpec | None = None

    # ------------------------------------------------------------------
    # Mesh accessors
    # ------------------------------------------------------------------
    def mesh_tensors(self, *, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(verts, faces)`` tensors, loading from cache if needed."""

        art = mesh_from_snippet(self, device=torch.device(device) if device is not None else None)
        return art.processed.verts, art.processed.faces

    def mesh_p3d(self, *, device: torch.device | str | None = None, meshes_cls: Any | None = None) -> Any:
        """Return PyTorch3D ``Meshes`` for this snippet, cached by spec hash."""

        art = mesh_from_snippet(
            self,
            device=torch.device(device) if device is not None else None,
            want_p3d=True,
            meshes_cls=meshes_cls,
        )
        return art.p3d

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------
    def get_camera(self, prefix: Literal["rgb", "slaml", "slamr"]) -> EfmCameraView:
        idx = {"rgb": 0, "slaml": 1, "slamr": 2}[prefix]
        img_key = ARIA_IMG[idx]
        calib_key = ARIA_CALIB[idx]
        time_key = ARIA_IMG_TIME_NS[idx]
        frame_id_key = ARIA_FRAME_ID[idx]
        distance_key = ARIA_DISTANCE_M[idx]
        distance_time_key = ARIA_DEPTH_TIME_NS[idx]

        img = self.efm[img_key]
        calib: CameraTW = self.efm[calib_key]
        time_ns = self.efm[time_key]
        frame_ids = self.efm[frame_id_key]
        distance = self.efm.get(distance_key)
        distance_time_ns = self.efm.get(distance_time_key)

        return EfmCameraView(
            images=img,
            calib=calib,
            time_ns=time_ns,
            frame_ids=frame_ids,
            distance_m=distance,
            distance_time_ns=distance_time_ns,
        )

    @property
    def camera_rgb(self) -> EfmCameraView:
        """RGB stream (fisheye624, 240x240 in ASE preprocessing)."""

        return self.get_camera("rgb")

    @property
    def camera_slam_left(self) -> EfmCameraView:
        """Left SLAM mono stream (fisheye624, ~320x240)."""

        return self.get_camera("slaml")

    @property
    def camera_slam_right(self) -> EfmCameraView:
        """Right SLAM mono stream (fisheye624, ~320x240)."""

        return self.get_camera("slamr")

    # ------------------------------------------------------------------
    # Trajectory and semidense points
    # ------------------------------------------------------------------
    @property
    def trajectory(self) -> EfmTrajectoryView:
        """Rig trajectory (world←rig) aligned to snippet frames."""

        return EfmTrajectoryView(
            t_world_rig=self.efm[ARIA_POSE_T_WORLD_RIG],
            time_ns=self.efm[ARIA_POSE_TIME_NS],
            gravity_in_world=self.efm["pose/gravity_in_world"],
        )

    @property
    def semidense(self) -> EfmPointsView:
        """Semi-dense SLAM points (padded to fixed length)."""

        efm = self.efm
        points = efm[ARIA_POINTS_WORLD]
        lengths = efm.get("points/lengths")
        if lengths is None:
            lengths = torch.full(
                (points.shape[0],),
                points.shape[1],
                dtype=torch.int64,
                device=points.device,
            )
        return EfmPointsView(
            points_world=points,
            dist_std=efm[ARIA_POINTS_DIST_STD],
            inv_dist_std=efm[ARIA_POINTS_INV_DIST_STD],
            time_ns=efm[ARIA_POINTS_TIME_NS],
            volume_min=efm.get(ARIA_POINTS_VOL_MIN, efm.get("points/vol_min")),
            volume_max=efm.get(ARIA_POINTS_VOL_MAX, efm.get("points/vol_max")),
            lengths=lengths,
        )

    # ------------------------------------------------------------------
    # OBB / occupancy
    # ------------------------------------------------------------------
    @property
    def obbs(self) -> EfmObbView | None:
        """Snippet-level OBBs (if present)."""

        if "obbs/padded_snippet" not in self.efm:
            return None
        return EfmObbView(
            obbs=self.efm[ARIA_OBB_PADDED],
            hz=self.efm.get(ARIA_OBB_FREQUENCY_HZ),
        )

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    @property
    def gt(self) -> EfmGTView:
        """Ground-truth dict ``gt_data`` (EFM OBBs per timestamp/camera)."""

        return EfmGTView(raw=self.efm.get("gt_data", {}))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @property
    def has_mesh(self) -> bool:
        return self.mesh is not None

    def get_occupancy_extend(self) -> torch.Tensor:
        """Return ``[xmin, xmax, ymin, ymax, zmin, zmax]`` AABB in world frame."""
        if self.crop_bounds is None:
            raise ValueError("EfmSnippetView.crop_bounds is missing; dataset should always populate it.")
        bounds_min, bounds_max = self.crop_bounds
        return torch.stack(
            [bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1], bounds_min[2], bounds_max[2]], dim=0
        )

    def to(self, device: str | torch.device, *, dtype: torch.dtype | None = None) -> "EfmSnippetView":
        """Shallow device move for heavy tensors."""

        target_device = torch.device(device)
        target_type = target_device.type
        if dtype is None and all(
            isinstance(v, Tensor)
            and v.device.type == target_type
            or isinstance(v, (PoseTW, CameraTW, TensorWrapper, ObbTW))
            for v in self.efm.values()
        ):
            return self

        moved: dict[str, Any] = {}
        for k, v in self.efm.items():
            if isinstance(v, Tensor):
                moved[k] = v.to(device=target_device, dtype=dtype)
            elif isinstance(v, (PoseTW, CameraTW, TensorWrapper, ObbTW)):
                moved[k] = v.to(device=target_device)  # type: ignore[arg-type]
            else:
                moved[k] = v
        cb = self.crop_bounds
        if cb is not None:
            cb = (cb[0].to(target_device), cb[1].to(target_device))
        mv = self.mesh_verts
        mf = self.mesh_faces
        if mv is not None:
            mv = mv.to(target_device)
        if mf is not None:
            mf = mf.to(target_device)
        return replace(
            self,
            efm=moved,
            mesh=self.mesh,
            crop_bounds=cb,
            mesh_verts=mv,
            mesh_faces=mf,
            mesh_cache_key=self.mesh_cache_key,
            mesh_specs=self.mesh_specs,
        )

    def __repr__(self) -> str:  # pragma: no cover
        base = {
            "scene": self.scene_id,
            "snippet": self.snippet_id,
            "cameras": {
                "rgb": summarize(self.efm.get("rgb/img")),
                "slaml": summarize(self.efm.get("slaml/img")),
                "slamr": summarize(self.efm.get("slamr/img")),
            },
            "trajectory": summarize(self.efm.get(ARIA_POSE_T_WORLD_RIG)),
            "semidense": summarize(self.efm.get(ARIA_POINTS_WORLD)),
            "obbs": summarize(self.efm.get(ARIA_OBB_PADDED)),
            "mesh": str(self.mesh) if self.mesh is not None else None,
            "crop_bounds": summarize(self.crop_bounds),
        }
        return pformat(base, indent=2, width=120, compact=False)


__all__ = [
    "EfmCameraView",
    "EfmTrajectoryView",
    "EfmPointsView",
    "EfmObbView",
    "EfmGTView",
    "EfmGtTimestampView",
    "EfmGtCameraObbView",
    "EfmSnippetView",
]
