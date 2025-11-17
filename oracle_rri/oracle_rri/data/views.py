"""Typed, zero-copy views over flattened ATEK samples.

Properties fetch values lazily from the underlying dict using `.get` and
raise `KeyError` when required fields are missing. Shapes, dtypes, and frame
conventions are documented per attribute for quick reference.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pprint import pformat
from typing import Any

import torch
import trimesh
from efm3d.aria.pose import PoseTW
from rich.text import Text
from rich.tree import Tree
from torch import Tensor

from ..utils import Console


def _summ(value: Any) -> Any:
    """Summarise tensors/lists for human-friendly repr output."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() <= 10 and value.dtype.is_floating_point:
            return {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.min()),
                "max": float(value.max()),
                "mean": float(value.mean()),
            }
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        first = value[0]
        return {"len": len(value), "first": {"shape": tuple(first.shape), "dtype": str(first.dtype)}}
    if isinstance(value, (list, tuple)):
        return {"len": len(value)}
    if hasattr(value, "__dict__") or hasattr(value, "__slots__"):
        try:
            items = vars(value)
        except TypeError:  # slots
            items = {k: getattr(value, k) for k in getattr(value, "__slots__", [])}
        return {k: _summ(v) for k, v in items.items()}
    if isinstance(value, dict):
        return {"keys": list(value.keys())}
    return value


def _repr_dict(
    block: dict[str, Any],
    *,
    indent: int = 2,
    width: int = 120,
) -> str:
    """Pretty-print a nested dict with controllable indent and width."""
    return pformat(block, indent=indent, width=width, compact=False)


def _compact_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Remove None entries and truncate long lists for human-friendly repr."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) > 5:
            out[k] = f"{len(v)} items (first 3): {list(v)[:3]}"
        else:
            out[k] = v
    return out


def _truncate_list(items: list[Any], *, max_items: int = 5) -> list[Any] | str:
    """Return a shortened representation for long lists."""

    if len(items) > max_items:
        head = items[:max_items]
        return f"{len(items)} items (first {max_items}): {head}"
    return items


@dataclass(slots=True)
class BaseView:
    """Shared helpers for zero-copy tensor views."""

    def to(self, device: str | torch.device, *, dtype: torch.dtype = None):  # pragma: no cover - simple cast
        moved = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                moved[f.name] = val.to(device=device, dtype=dtype)
            elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
                moved[f.name] = [v.to(device=device, dtype=dtype) for v in val]
            else:
                moved[f.name] = val
        return replace(self, **moved)

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        summary = {}
        for f in fields(self):
            val = getattr(self, f.name)
            summary[f.name] = _summ(val)
        return _repr_dict(summary)


@dataclass(slots=True)
class CameraView(BaseView):
    """Multi-frame camera stream (Aria camera frame: +x right, +y down, +z forward).

    Image tensors are shaped `(F, C, H, W)` with C=3 for RGB or C=1 for depth.

    **Intrinsics.** `projection_params` follow Project Aria's *Fisheye624* (rad–tan–thin-prism)
    parameterisation: `[fu, fv, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2]`.
    The pinhole block is

    .. math::
        K = \\begin{bmatrix}
        f_u & 0 & c_u \\\\
        0 & f_v & c_v \\\\
        0 & 0 & 1
        \\end{bmatrix},

    with radial coefficients `k*`, tangential `p0/p1`, and thin-prism `s0..s2` applied by
    the fisheye projection model. Values are in **pixels** at the native render resolution.

    **Extrinsics.** `t_device_camera` stores `T_device_camera`: a 3×4 SE(3) that maps camera
    coordinates → device/rig frame. World poses recover as `T_world_camera = T_world_device @
    T_device_camera` using the trajectory view.
    """

    images: Tensor
    """Shape `(F,C,H,W)` uint8|float32 in camera frame (C=3 RGB, C=1 depth)."""
    projection_params: Tensor
    """Shape `(15,)` or `(F,15)` float32 intrinsics (fu, fv, cu, cv, k0..k5, p0, p1, s0..s2)."""
    t_device_camera: Tensor
    """Shape `(3,4)` or `(F,3,4)` float32; `T_device_camera` (camera→rig) per frame."""
    capture_timestamps_ns: Tensor
    """Shape `(F,)` int64 device timestamps aligned to images."""
    frame_ids: Tensor
    """Shape `(F,)` int64 frame identifiers per stream."""
    exposure_durations_s: Tensor
    """Shape `(F,)` float32 exposure durations (seconds)."""
    gains: Tensor
    """Shape `(F,)` float32 analog gains."""
    camera_model_name: str
    """Camera model name (e.g., `FISHEYE624`)."""
    camera_valid_radius: Tensor
    """Shape `(1)` or `(F,1)` float32; valid fisheye radius in pixels."""


@dataclass(slots=True)
class TrajectoryView(BaseView):
    """Rig trajectory from MPS (world frame; Z-up, metres).

    `ts_world_device` stores `T_world_device` (camera rig pose) for each frame, aligned to the
    timestamps in the camera streams. Compose with `T_device_camera` to obtain per-camera world
    poses. Gravity is reported in world coordinates to aid frame alignment.
    """

    ts_world_device: Tensor
    """Shape `(F,3,4)` float32; rig pose per frame (world←device)."""
    capture_timestamps_ns: Tensor
    """Shape `(F,)` int64; device timestamps for the trajectory samples."""
    gravity_in_world: Tensor
    """Shape `(3,)` float32; gravity vector expressed in world frame."""

    @property
    def final_pose(self) -> PoseTW:
        """Final rig pose `T_world_device` (world←device)."""

        return PoseTW.from_matrix3x4(self.ts_world_device[-1, ...])


@dataclass(slots=True)
class SemiDenseView(BaseView):
    """Semi-dense SLAM point observations (world frame, metres).

    Each element in `points_world` corresponds to one frame and lives in \n
    the same world coordinate system as `ts_world_device`. Uncertainty is\n
    captured via optional distance and inverse-distance standard deviations.\n
    `volume_min/max` provide an axis-aligned bounding box over all points in\n
    the snippet, useful for normalising or voxelisation.
    """

    points_world: list[Tensor]
    """List length F; each tensor `(N,3)` float32 world-frame points."""
    points_dist_std: list[Tensor]
    """List length F; each `(N,)` float32 per-point distance stddev."""
    points_inv_dist_std: list[Tensor]
    """List length F; each `(N,)` float32 inverse distance stddev."""
    capture_timestamps_ns: Tensor
    """Shape `(F,)` int64; timestamps corresponding to each points slice."""
    volume_min: Tensor
    """Shape `(3,)` float32; world AABB minimum for points."""
    volume_max: Tensor
    """Shape `(3,)` float32; world AABB maximum for points."""

    # repr/to inherited from BaseView


@dataclass(slots=True)
class Obb3View(BaseView):
    """3D oriented bounding boxes expressed in world frame.

    instance_ids : Tensor | None
        `(K,)` int64 instance ids, shared with 2D OBBs when present.
    category_ids : Tensor | None
        `(K,)` int64 category ids (see ATEK category map).
    category_names : list[str] | None
        Human readable category names aligned with `category_ids`.
    object_dimensions : Tensor | None
        `(K,3)` float32 box side lengths in XYZ order.
    ts_world_object : Tensor | None
        `(K,3,4)` float32 poses (world←object).
    """

    instance_ids: Tensor | None
    """Shape `(K,)` int64; unique instance ids per object."""
    category_ids: Tensor | None
    """Shape `(K,)` int64; semantic category ids."""
    category_names: list[str] | None
    """List length `K`; category labels aligned with `category_ids`."""
    object_dimensions: Tensor | None
    """Shape `(K,3)` float32; box side lengths (x, y, z) in metres."""
    ts_world_object: Tensor | None
    """Shape `(K,3,4)` float32; T_world_object per instance."""


@dataclass(slots=True)
class EfmFrame(BaseView):
    """Frame-level EVL/EFM ground truth.

    Older EFM adapters store frame-level tensors under ``efm_gt[ts]["frame"]``
    (e.g. padded OBB3D arrays). Newer ATEK shards expose the same data per
    camera directly under ``efm_gt[ts][camera_id]`` without a wrapping
    ``"frame"`` key. This view stays flexible by simply wrapping the provided
    dict; the convenience accessors return ``None`` when a given key is not
    present so callers can write defensive code.
    """

    raw: dict[str, Tensor]
    """Underlying frame-level EFM GT dict (zero-copy backing store)."""

    @property
    def obb3d(self) -> Tensor | None:
        return self.raw.get("obb3d")

    @property
    def obb3d_mask(self) -> Tensor | None:
        return self.raw.get("obb3d_mask")

    @property
    def category_ids(self) -> Tensor | None:
        return self.raw.get("category_ids")

    @property
    def instance_ids(self) -> Tensor | None:
        return self.raw.get("instance_ids")


@dataclass(slots=True)
class EfmPerCamera(BaseView):
    """Per-camera EVL/EFM ground truth.

    Mirrors :class:`EfmFrame` but for per-camera tensors such as 2D boxes or
    per-camera visibility flags. As above, keys are adapter-dependent and may
    be missing; consumers should guard for ``None``.
    """

    raw: dict[str, Tensor]
    """Underlying per-camera EFM GT dict."""

    @property
    def obb2d(self) -> Tensor | None:
        return self.raw.get("obb2d")

    @property
    def obb2d_mask(self) -> Tensor | None:
        return self.raw.get("obb2d_mask")

    @property
    def visibility(self) -> Tensor | None:
        return self.raw.get("visibility")


@dataclass(slots=True)
class GTView(BaseView):
    """Ground-truth annotations (OBB3/OBB2) wrapped in typed views."""

    raw: dict[str, Any]
    """Original `gt_data` mapping as provided by ATEK."""
    obb3_gt: dict[str, Obb3View] | None = None
    """Per-camera OBB3 annotations; keys are camera labels."""
    obb2_gt: dict[str, dict[str, Any]] | None = None
    """Optional 2D OBB annotations; kept as raw mapping for flexibility."""
    scores: Tensor | None = None
    """Optional quality scores tensor associated with this snippet."""
    efm_gt: dict[str, Any] | None = None
    """Optional EVL/EFM-formatted targets.

    Current ATEK shards expose EFM targets as a mapping from timestamp strings
    to per-camera dictionaries, e.g. ``efm_gt[ts][\"camera-rgb\"]`` with
    3D OBB tensors.
    """
    rri_targets: dict[str, Any] | None = None
    """Any precomputed RRI supervision targets."""

    def __post_init__(self) -> None:
        """Parse the raw mapping into typed sub-views."""

        raw_dict = self.raw if isinstance(self.raw, dict) else {}
        self.raw = raw_dict
        obb3_raw = raw_dict.get("obb3_gt")
        if isinstance(obb3_raw, dict):
            parsed: dict[str, Obb3View] = {}
            for cam, data in obb3_raw.items():
                if not isinstance(data, dict):
                    continue
                parsed[cam] = Obb3View(
                    instance_ids=data.get("instance_ids"),
                    category_ids=data.get("category_ids"),
                    category_names=data.get("category_names"),
                    object_dimensions=data.get("object_dimensions"),
                    ts_world_object=data.get("ts_world_object"),
                )
            self.obb3_gt = parsed or None
        obb2_raw = raw_dict.get("obb2_gt")
        if isinstance(obb2_raw, dict):
            self.obb2_gt = obb2_raw
        scores_raw = raw_dict.get("scores")
        if torch.is_tensor(scores_raw):
            self.scores = scores_raw
        efm_raw = raw_dict.get("efm_gt")
        if isinstance(efm_raw, dict):
            self.efm_gt = efm_raw
        rri_raw = raw_dict.get("rri_targets")
        if isinstance(rri_raw, dict):
            self.rri_targets = rri_raw

    @property
    def efm_keys(self) -> list[str]:
        """List of EFM variants present in :pyattr:`efm_gt`.

        Empty list if no EFM ground truth is attached.
        """
        return [] if self.efm_gt is None else sorted(self.efm_gt.keys())

    def _get_efm_entry(self, efm_key: str | int | None = None) -> dict[str, Any]:
        """Internal: resolve an EFM entry by key or integer index."""
        if self.efm_gt is None:
            raise RuntimeError("No `efm_gt` available on this GTView instance.")

        efm_key = efm_key or self.efm_keys[0]

        if isinstance(efm_key, int):
            keys = self.efm_keys
            if not keys:
                raise RuntimeError("`efm_gt` is non-empty but `efm_keys` is empty.")
            try:
                efm_key = keys[efm_key]
            except IndexError as exc:
                raise IndexError(f"efm_key index {efm_key} out of range for keys {keys}") from exc

        entry = self.efm_gt.get(efm_key)
        if entry is None:
            raise KeyError(f"Unknown efm_key {efm_key!r}; available keys: {self.efm_keys}")
        return entry

    def efm_frame(self, efm_key: str | int | None = None) -> EfmFrame | dict[str, Obb3View]:
        """Frame-level or per-camera EFM targets for the given key.

        If a legacy "frame" block exists it is wrapped as :class:`EfmFrame`.
        Otherwise the current schema (cameras directly under the timestamp
        key) is returned as a mapping of camera id → :class:`Obb3View`.
        """

        entry = self._get_efm_entry(efm_key)
        frame_dict = entry.get("frame") if isinstance(entry, dict) else None
        if isinstance(frame_dict, dict):
            return EfmFrame(raw=frame_dict)

        if not isinstance(entry, dict):
            raise TypeError(f"Unexpected efm_gt entry type {type(entry)!r}; expected dict")

        return self._efm_cameras_from_entry(entry)

    def efm_per_camera(self, efm_key: str | int | None = None) -> dict[str, EfmPerCamera | Obb3View]:
        """Per-camera EFM ground truth for a given EFM variant.

        Supports both legacy ``"per_camera"`` blocks and the current schema
        where cameras sit directly under the timestamp key. Payloads with
        3D OBB tensors are returned as :class:`Obb3View`; other payloads use
        :class:`EfmPerCamera`.
        """

        entry = self._get_efm_entry(efm_key)
        per_cam = entry.get("per_camera") if isinstance(entry, dict) else None
        if per_cam is None and isinstance(entry, dict):
            per_cam = entry  # cameras stored directly

        if not isinstance(per_cam, dict):
            raise TypeError(f"Expected per-camera EFM dict, got {type(per_cam)!r}")

        out: dict[str, EfmPerCamera | Obb3View] = {}
        for cam_id, cam_dict in per_cam.items():
            if not isinstance(cam_dict, dict):
                continue
            if {"object_dimensions", "ts_world_object"} & set(cam_dict):
                out[cam_id] = Obb3View(
                    instance_ids=cam_dict.get("instance_ids"),
                    category_ids=cam_dict.get("category_ids"),
                    category_names=cam_dict.get("category_names"),
                    object_dimensions=cam_dict.get("object_dimensions"),
                    ts_world_object=cam_dict.get("ts_world_object"),
                )
            else:
                out[cam_id] = EfmPerCamera(raw=cam_dict)
        return out

    def _efm_cameras_from_entry(self, entry: dict[str, Any]) -> dict[str, Obb3View]:
        """Helper: map camera ids to :class:`Obb3View` for a given EFM entry."""

        cameras: dict[str, Obb3View] = {}
        for cam_id, cam_dict in entry.items():
            if not isinstance(cam_dict, dict):
                continue
            cameras[cam_id] = Obb3View(
                instance_ids=cam_dict.get("instance_ids"),
                category_ids=cam_dict.get("category_ids"),
                category_names=cam_dict.get("category_names"),
                object_dimensions=cam_dict.get("object_dimensions"),
                ts_world_object=cam_dict.get("ts_world_object"),
            )
        return cameras


@dataclass(slots=True)
class TypedSample(BaseView):
    """Typed wrapper over a flattened ATEK sample plus optional mesh."""

    flat: dict[str, Any]
    """Raw flattened ATEK sample (zero-copy backing store for all views)."""
    scene_id: str
    """Scene identifier extracted from sequence name or shard path."""
    snippet_id: str
    """Snippet identifier (e.g., `shards-0000`)."""
    mesh: trimesh.Trimesh | None = None
    """Optional GT mesh paired with the snippet."""

    @property
    def gt_mesh(self) -> trimesh.Trimesh:
        """Convenience accessor for attached GT mesh (may be `None`)."""
        if self.mesh is None:
            raise KeyError("No GT mesh attached to this sample")
        return self.mesh

    @property
    def has_mesh(self) -> bool:
        """True if a GT mesh is attached."""

        return self.mesh is not None

    def _require(self, key: str) -> Any:
        val = self.flat.get(key)
        if val is None:
            raise KeyError(f"Missing required key '{key}' in sample {self.scene_id}/{self.snippet_id}")
        return val

    def _camera(self, prefix: str) -> CameraView:
        f = self.flat
        images = f.get(f"{prefix}+images")
        proj = f.get(f"{prefix}+projection_params")
        t_dev_cam = f.get(f"{prefix}+t_device_camera")
        ts = f.get(f"{prefix}+capture_timestamps_ns")
        frame_ids = f.get(f"{prefix}+frame_ids")
        exposure = f.get(f"{prefix}+exposure_durations_s")
        gains = f.get(f"{prefix}+gains")
        model = f.get(f"{prefix}+camera_model_name")
        radius = f.get(f"{prefix}+camera_valid_radius")
        if images is None or proj is None or t_dev_cam is None or ts is None or frame_ids is None:
            missing = [
                k
                for k, v in {
                    "images": images,
                    "projection_params": proj,
                    "t_device_camera": t_dev_cam,
                    "capture_timestamps_ns": ts,
                    "frame_ids": frame_ids,
                }.items()
                if v is None
            ]
            raise KeyError(f"Incomplete camera stream '{prefix}' missing {missing}")
        return CameraView(
            images=images,
            projection_params=proj,
            t_device_camera=t_dev_cam,
            capture_timestamps_ns=ts,
            frame_ids=frame_ids,
            exposure_durations_s=exposure,
            gains=gains,
            camera_model_name=model,
            camera_valid_radius=radius,
        )

    @property
    def camera_rgb(self) -> CameraView:
        """RGB stream: `mfcd#camera-rgb` (F≈20, C=3)."""
        return self._camera("mfcd#camera-rgb")

    @property
    def camera_slam_left(self) -> CameraView:
        """Left SLAM grayscale stream: `mfcd#camera-slam-left` (C=1)."""
        return self._camera("mfcd#camera-slam-left")

    @property
    def camera_slam_right(self) -> CameraView:
        """Right SLAM grayscale stream: `mfcd#camera-slam-right` (C=1)."""
        return self._camera("mfcd#camera-slam-right")

    @property
    def camera_rgb_depth(self) -> CameraView:
        """Aligned depth stream: `mfcd#camera-rgb-depth` (C=1 distance/ray-length)."""
        return self._camera("mfcd#camera-rgb-depth")

    @property
    def trajectory(self) -> TrajectoryView:
        """Rig poses `mtd#ts_world_device` aligned to camera timestamps."""
        return TrajectoryView(
            ts_world_device=self._require("mtd#ts_world_device"),
            capture_timestamps_ns=self._require("mtd#capture_timestamps_ns"),
            gravity_in_world=self.flat.get("mtd#gravity_in_world"),
        )

    @property
    def semidense(self) -> SemiDenseView:
        """Semi-dense SLAM points `msdpd#points_world` plus uncertainty and AABB."""
        f = self.flat
        points = f.get("msdpd#points_world")
        if points is None:
            raise KeyError("Missing msdpd#points_world in semidense view")
        vol_min = f.get("msdpd#points_volume_min")
        if vol_min is None:
            vol_min = f.get("msdpd#points_volumn_min")
        vol_max = f.get("msdpd#points_volume_max")
        if vol_max is None:
            vol_max = f.get("msdpd#points_volumn_max")
        return SemiDenseView(
            points_world=points,
            points_dist_std=f.get("msdpd#points_dist_std"),
            points_inv_dist_std=f.get("msdpd#points_inv_dist_std"),
            capture_timestamps_ns=f.get("msdpd#capture_timestamps_ns"),
            volume_min=vol_min,
            volume_max=vol_max,
        )

    @property
    def gt(self) -> GTView:
        """Ground-truth dict `gt_data` (OBB2/OBB3/EFM targets)."""
        return GTView(raw=self.flat.get("gt_data", {}))

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def _camera_summary(self) -> dict[str, Any]:
        return _compact_dict(
            {
                "rgb": _summ(self.flat.get("mfcd#camera-rgb+images")),
                "rgb_depth": _summ(self.flat.get("mfcd#camera-rgb-depth+images")),
                "slam_l": _summ(self.flat.get("mfcd#camera-slam-left+images")),
                "slam_r": _summ(self.flat.get("mfcd#camera-slam-right+images")),
            }
        )

    def _semidense_summary(self) -> dict[str, Any]:
        sem = self.semidense
        return _compact_dict(
            {
                "frames": len(sem.points_world) if sem.points_world else 0,
                "points_example": _summ(sem.points_world[0]) if sem.points_world else None,
                "volume": {
                    "min": _summ(sem.volume_min),
                    "max": _summ(sem.volume_max),
                },
            }
        )

    def _efm_summary(self, max_entries: int = 2) -> dict[str, Any] | None:
        gt = self.gt
        if gt.efm_gt is None:
            return None

        keys = gt.efm_keys
        examples: dict[str, Any] = {}
        for key in keys[:max_entries]:
            entry = gt._get_efm_entry(key)
            if not isinstance(entry, dict):
                continue
            cams: dict[str, Any] = {}
            for cam_id, cam_dict in entry.items():
                if not isinstance(cam_dict, dict):
                    continue
                cams[cam_id] = _compact_dict(
                    {
                        "category_ids": _summ(cam_dict.get("category_ids")),
                        "instance_ids": _summ(cam_dict.get("instance_ids")),
                        "object_dimensions": _summ(cam_dict.get("object_dimensions")),
                        "ts_world_object": _summ(cam_dict.get("ts_world_object")),
                        "category_names": _summ(cam_dict.get("category_names")),
                    }
                )
            if cams:
                examples[str(key)] = cams

        return {
            "count": len(keys),
            "keys": _truncate_list(keys),
            "examples": examples,
        }

    def _gt_summary(self, *, include_efm_details: bool = False, efm_examples: int = 2) -> dict[str, Any]:
        gt = self.gt
        efm_gt_summary: dict[str, Any] | list[str] | str | None
        if include_efm_details:
            efm_gt_summary = self._efm_summary(max_entries=efm_examples)
        elif gt.efm_gt is None:
            efm_gt_summary = None
        else:
            efm_gt_summary = _truncate_list(gt.efm_keys)

        return _compact_dict(
            {
                "keys": list(gt.raw.keys()),
                "obb3": None if gt.obb3_gt is None else {k: _summ(v.object_dimensions) for k, v in gt.obb3_gt.items()},
                "obb2": None if gt.obb2_gt is None else list(gt.obb2_gt.keys()),
                "scores": _summ(gt.scores),
                "efm_gt": efm_gt_summary,
                "rri_targets": None if gt.rri_targets is None else list(gt.rri_targets.keys()),
            }
        )

    def _base_summary(self, *, include_efm_details: bool) -> dict[str, Any]:
        cam = self._camera_summary()
        sem = self._semidense_summary()
        gt_summary = self._gt_summary(include_efm_details=include_efm_details)

        return {
            "scene": self.scene_id,
            "snippet": self.snippet_id,
            "cameras": cam,
            "traj": _summ(self.trajectory),
            "semidense": sem,
            "gt": gt_summary,
            "mesh": None if self.mesh is None else {"verts": len(self.mesh.vertices), "faces": len(self.mesh.faces)},
        }

    def to_efm_dict(
        self,
        include_mesh: bool = False,
        key_mapping: dict[str, str] = None,
    ) -> dict[str, Any]:
        """Remap to EFM3D schema using the provided key mapping."""

        if key_mapping is None:
            try:
                from efm3d.dataset.efm_model_adaptor import EfmModelAdaptor

                key_mapping = EfmModelAdaptor.get_dict_key_mapping_all()
            except ModuleNotFoundError:
                key_mapping = {
                    "mfcd#camera-rgb+images": "rgb/img",
                    "mtd#ts_world_device": "pose/t_world_rig",
                    "msdpd#points_world": "points/p3s_world",
                }

        out: dict[str, Any] = {dst: self.flat[src] for src, dst in key_mapping.items() if src in self.flat}
        if "rgb/img" not in out and "mfcd#camera-rgb+images" in self.flat:
            out["rgb/img"] = self.flat["mfcd#camera-rgb+images"]
        if "pose/t_world_rig" not in out and "mtd#ts_world_device" in self.flat:
            out["pose/t_world_rig"] = self.flat["mtd#ts_world_device"]
        out["scene_id"] = self.scene_id
        out["snippet_id"] = self.snippet_id
        if include_mesh:
            out["gt_mesh"] = self.mesh
        return out

    def __repr__(self) -> str:  # pragma: no cover - formatting-only
        base = self._base_summary(include_efm_details=True)
        return _repr_dict(
            {
                "scene_id": self.scene_id,
                "snippet_id": self.snippet_id,
                "atek": {
                    "sequence": self.flat.get("sequence_name"),
                    "cameras": base["cameras"],
                    "trajectory": base["traj"],
                    "semidense": base["semidense"],
                    "gt": base["gt"],
                },
                "gt_mesh": base["mesh"],
            }
        )

    def summary(self, width: int = 120) -> str:
        """Concise human-friendly summary; hides None fields and truncates long lists."""

        base = self._base_summary(include_efm_details=True)
        return _repr_dict(base, width=width)

    def rich_summary(self, show_semidense: bool = True, show_gt: bool = True) -> None:
        """Render a rich tree summary similar to BaseConfig.inspect().

        This uses the project Console and rich Tree formatting to display a
        structured overview of the sample: cameras, trajectory, semidense
        volume, GT annotations, and mesh stats.
        """

        console = Console.with_prefix(self.__class__.__name__, "rich_summary")
        base = self._base_summary(include_efm_details=True)

        root = Tree(Text(f"TypedSample {self.scene_id}/{self.snippet_id}", style="config.name"))

        # Basic identifiers
        meta = root.add(Text("meta", style="config.field"))
        meta.add(Text(f"scene: {self.scene_id}", style="config.field"))
        meta.add(Text(f"snippet: {self.snippet_id}", style="config.field"))

        # Cameras
        cam_node = root.add(Text("cameras", style="config.field"))
        for name, info in base["cameras"].items():
            node = cam_node.add(Text(f"{name}:", style="config.field"))
            node.add(Text(_repr_dict(info, indent=2, width=80), style="config.value"))

        # Trajectory
        traj_node = root.add(Text("traj", style="config.field"))
        for k, v in base["traj"].items():
            traj_node.add(Text(f"{k}: {v}", style="config.value"))

        # Semi-dense SLAM
        if show_semidense:
            sem_node = root.add(Text("semidense", style="config.field"))
            sem_summary = base["semidense"]
            if "frames" in sem_summary:
                sem_node.add(Text(f"frames: {sem_summary['frames']}", style="config.value"))
            points_example = sem_summary.get("points_example")
            if points_example is not None:
                sem_node.add(Text(f"points_example: {points_example}", style="config.value"))
            volume = sem_summary.get("volume", {})
            vol_node = sem_node.add(Text("volume", style="config.field"))
            if "min" in volume:
                vol_node.add(Text(f"min: {volume['min']}", style="config.value"))
            if "max" in volume:
                vol_node.add(Text(f"max: {volume['max']}", style="config.value"))

        # Ground truth annotations
        if show_gt:
            gt_summary = base["gt"]
            gt_node = root.add(Text("gt", style="config.field"))
            gt_node.add(Text(f"keys: {gt_summary.get('keys')}", style="config.value"))
            if gt_summary.get("obb3"):
                obb3_node = gt_node.add(Text("obb3", style="config.field"))
                for cam_name, obb3 in gt_summary["obb3"].items():
                    obb3_node.add(Text(f"{cam_name}: {obb3}", style="config.value"))
            if gt_summary.get("obb2"):
                gt_node.add(Text(f"obb2: {gt_summary['obb2']}", style="config.value"))
            if gt_summary.get("scores") is not None:
                gt_node.add(Text(f"scores: {gt_summary['scores']}", style="config.value"))

            efm_gt = gt_summary.get("efm_gt")
            if efm_gt:
                efm_node = gt_node.add(Text("efm_gt", style="config.field"))
                if isinstance(efm_gt, dict):
                    efm_node.add(Text(f"count: {efm_gt.get('count')}", style="config.value"))
                    efm_node.add(Text(f"keys: {efm_gt.get('keys')}", style="config.value"))
                    examples = efm_gt.get("examples", {}) or {}
                    for ts, cams in examples.items():
                        ts_node = efm_node.add(Text(str(ts), style="config.field"))
                        for cam_id, cam_info in cams.items():
                            cam_node = ts_node.add(Text(f"{cam_id}:", style="config.field"))
                            cam_node.add(Text(_repr_dict(cam_info, indent=2, width=80), style="config.value"))
                else:
                    efm_node.add(Text(str(efm_gt), style="config.value"))

            if gt_summary.get("rri_targets"):
                gt_node.add(Text(f"rri_targets: {gt_summary['rri_targets']}", style="config.value"))

        # Mesh
        mesh_node = root.add(Text("mesh", style="config.field"))
        if base["mesh"] is None:
            mesh_node.add(Text("None", style="config.value"))
        else:
            mesh_node.add(
                Text(
                    f"verts: {base['mesh']['verts']}, faces: {base['mesh']['faces']}",
                    style="config.value",
                )
            )

        console.print(root, soft_wrap=False, highlight=True, markup=True, emoji=False)

    def to(self, device: str | torch.device, *, dtype: torch.dtype | None = None) -> "TypedSample":
        """Return a shallow copy; consumers can call `.to` on sub-views explicitly."""

        return replace(self)


__all__ = [
    "TypedSample",
    "CameraView",
    "TrajectoryView",
    "SemiDenseView",
    "GTView",
    "Obb3View",
]
