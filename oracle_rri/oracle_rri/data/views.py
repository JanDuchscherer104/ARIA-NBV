"""Typed, zero-copy views over flattened ATEK samples.

Properties fetch values lazily from the underlying dict using ``.get`` and
raise ``KeyError`` when required fields are missing. Shapes, dtypes, and frame
conventions are documented per attribute for quick reference.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pprint import pformat
from typing import Any

import torch
import trimesh

Tensor = torch.Tensor


def _summ(value: Any) -> Any:
    """Summarise tensors/lists for human-friendly repr output."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return {"shape": tuple(value.shape), "dtype": str(value.dtype)}
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


def _repr_dict(block: dict[str, Any]) -> str:
    return pformat(block, indent=2, compact=False)


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

    Image tensors are shaped ``(F, C, H, W)`` with C=3 for RGB or C=1 for depth.

    **Intrinsics.** ``projection_params`` follow Project Aria's *Fisheye624* (rad–tan–thin-prism)
    parameterisation: ``[fu, fv, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2]``.
    The pinhole block is

    .. math::
        K = \\begin{bmatrix}
        f_u & 0 & c_u \\\\
        0 & f_v & c_v \\\\
        0 & 0 & 1
        \\end{bmatrix},

    with radial coefficients ``k*``, tangential ``p0/p1``, and thin-prism ``s0..s2`` applied by
    the fisheye projection model. Values are in **pixels** at the native render resolution.

    **Extrinsics.** ``t_device_camera`` stores ``T_device_camera``: a 3×4 SE(3) that maps camera
    coordinates → device/rig frame. World poses recover as ``T_world_camera = T_world_device @
    T_device_camera`` using the trajectory view.
    """

    images: Tensor
    """Shape ``(F,C,H,W)`` uint8|float32 in camera frame (C=3 RGB, C=1 depth)."""
    projection_params: Tensor
    """Shape ``(15,)`` or ``(F,15)`` float32 intrinsics (fu, fv, cu, cv, k0..k5, p0, p1, s0..s2)."""
    t_device_camera: Tensor
    """Shape ``(3,4)`` or ``(F,3,4)`` float32; ``T_device_camera`` (camera→rig) per frame."""
    capture_timestamps_ns: Tensor
    """Shape ``(F,)`` int64 device timestamps aligned to images."""
    frame_ids: Tensor
    """Shape ``(F,)`` int64 frame identifiers per stream."""
    exposure_durations_s: Tensor
    """Shape ``(F,)`` float32 exposure durations (seconds)."""
    gains: Tensor
    """Shape ``(F,)`` float32 analog gains."""
    camera_model_name: str
    """Camera model name (e.g., ``FISHEYE624``)."""
    camera_valid_radius: Tensor
    """Shape ``(1)`` or ``(F,1)`` float32; valid fisheye radius in pixels."""


@dataclass(slots=True)
class TrajectoryView(BaseView):
    """Rig trajectory from MPS (world frame; Z-up, metres).

    ``ts_world_device`` stores ``T_world_device`` (camera rig pose) for each frame, aligned to the
    timestamps in the camera streams. Compose with ``T_device_camera`` to obtain per-camera world
    poses. Gravity is reported in world coordinates to aid frame alignment.
    """

    ts_world_device: Tensor
    """Shape ``(F,3,4)`` float32; rig pose per frame (world←device)."""
    capture_timestamps_ns: Tensor
    """Shape ``(F,)`` int64; device timestamps for the trajectory samples."""
    gravity_in_world: Tensor
    """Shape ``(3,)`` float32; gravity vector expressed in world frame."""

    # repr/to inherited from BaseView


@dataclass(slots=True)
class SemiDenseView(BaseView):
    """Semi-dense SLAM point observations (world frame, metres).

    Each element in ``points_world`` corresponds to one frame and lives in \n
    the same world coordinate system as ``ts_world_device``. Uncertainty is\n
    captured via optional distance and inverse-distance standard deviations.\n
    ``volume_min/max`` provide an axis-aligned bounding box over all points in\n
    the snippet, useful for normalising or voxelisation.
    """

    points_world: list[Tensor]
    """List length F; each tensor ``(N,3)`` float32 world-frame points."""
    points_dist_std: list[Tensor]
    """List length F; each ``(N,)`` float32 per-point distance stddev."""
    points_inv_dist_std: list[Tensor]
    """List length F; each ``(N,)`` float32 inverse distance stddev."""
    capture_timestamps_ns: Tensor
    """Shape ``(F,)`` int64; timestamps corresponding to each points slice."""
    volume_min: Tensor
    """Shape ``(3,)`` float32; world AABB minimum for points."""
    volume_max: Tensor
    """Shape ``(3,)`` float32; world AABB maximum for points."""

    # repr/to inherited from BaseView


@dataclass(slots=True)
class Obb3View(BaseView):
    """3D oriented bounding boxes expressed in world frame.

    instance_ids : Tensor | None
        ``(K,)`` int64 instance ids, shared with 2D OBBs when present.
    category_ids : Tensor | None
        ``(K,)`` int64 category ids (see ATEK category map).
    category_names : list[str] | None
        Human readable category names aligned with ``category_ids``.
    object_dimensions : Tensor | None
        ``(K,3)`` float32 box side lengths in XYZ order.
    ts_world_object : Tensor | None
        ``(K,3,4)`` float32 poses (world←object).
    """

    instance_ids: Tensor | None
    """Shape ``(K,)`` int64; unique instance ids per object."""
    category_ids: Tensor | None
    """Shape ``(K,)`` int64; semantic category ids."""
    category_names: list[str] | None
    """List length ``K``; category labels aligned with ``category_ids``."""
    object_dimensions: Tensor | None
    """Shape ``(K,3)`` float32; box side lengths (x, y, z) in metres."""
    ts_world_object: Tensor | None
    """Shape ``(K,3,4)`` float32; T_world_object per instance."""


@dataclass(slots=True)
class GTView(BaseView):
    """Ground-truth annotations (OBB3/OBB2) wrapped in typed views."""

    raw: dict[str, Any]
    """Original ``gt_data`` mapping as provided by ATEK."""
    obb3_gt: dict[str, Obb3View] | None = None
    """Per-camera OBB3 annotations; keys are camera labels."""
    obb2_gt: dict[str, dict[str, Any]] | None = None
    """Optional 2D OBB annotations; kept as raw mapping for flexibility."""
    scores: Tensor | None = None
    """Optional quality scores tensor associated with this snippet."""
    efm_gt: dict[str, Any] | None = None
    """Optional EVL/EFM-formatted targets (e.g., padded OBB tensors)."""

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

    @property
    def has_obb3(self) -> bool:
        """True when 3D bounding boxes are available."""

        return bool(self.obb3_gt)

    def to_raw(self) -> dict[str, Any]:
        """Return the original ``gt_data`` mapping (unmodified)."""

        return self.raw


@dataclass(slots=True)
class TypedSample(BaseView):
    """Typed wrapper over a flattened ATEK sample plus optional mesh."""

    flat: dict[str, Any]
    """Raw flattened ATEK sample (zero-copy backing store for all views)."""
    scene_id: str
    """Scene identifier extracted from sequence name or shard path."""
    snippet_id: str
    """Snippet identifier (e.g., ``shards-0000``)."""
    mesh: trimesh.Trimesh = None
    """Optional GT mesh paired with the snippet."""

    @property
    def gt_mesh(self) -> trimesh.Trimesh:
        """Convenience accessor for attached GT mesh (may be ``None``)."""

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

    # TODO: all properties must have doc-strings!
    @property
    def camera_rgb(self) -> CameraView:
        """RGB stream: ``mfcd#camera-rgb`` (F≈20, C=3)."""
        return self._camera("mfcd#camera-rgb")

    @property
    def camera_slam_left(self) -> CameraView:
        """Left SLAM grayscale stream: ``mfcd#camera-slam-left`` (C=1)."""
        return self._camera("mfcd#camera-slam-left")

    @property
    def camera_slam_right(self) -> CameraView:
        """Right SLAM grayscale stream: ``mfcd#camera-slam-right`` (C=1)."""
        return self._camera("mfcd#camera-slam-right")

    @property
    def camera_rgb_depth(self) -> CameraView:
        """Aligned depth stream: ``mfcd#camera-rgb-depth`` (C=1 distance/ray-length)."""
        return self._camera("mfcd#camera-rgb-depth")

    @property
    def trajectory(self) -> TrajectoryView:
        """Rig poses ``mtd#ts_world_device`` aligned to camera timestamps."""
        return TrajectoryView(
            ts_world_device=self._require("mtd#ts_world_device"),
            capture_timestamps_ns=self._require("mtd#capture_timestamps_ns"),
            gravity_in_world=self.flat.get("mtd#gravity_in_world"),
        )

    @property
    def semidense(self) -> SemiDenseView:
        """Semi-dense SLAM points ``msdpd#points_world`` plus uncertainty and AABB."""
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
        """Ground-truth dict ``gt_data`` (OBB2/OBB3/EFM targets)."""
        return GTView(raw=self.flat.get("gt_data", {}))

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
        return _repr_dict(
            {
                "scene_id": self.scene_id,
                "snippet_id": self.snippet_id,
                "atek": {
                    "sequence_name": self.flat.get("sequence_name"),
                    "cameras": {
                        "RGB": _summ(self.camera_rgb),
                        "SLAM_LEFT": _summ(self.camera_slam_left),
                        "SLAM_RIGHT": _summ(self.camera_slam_right),
                        "RGB_DEPTH": _summ(self.camera_rgb_depth),
                    },
                    "trajectory": _summ(self.trajectory),
                    "semidense": _summ(self.semidense),
                    "gt_data_keys": list(self.gt.raw.keys()),
                },
                "gt_mesh": None
                if self.mesh is None
                else {"verts": len(self.mesh.vertices), "faces": len(self.mesh.faces)},
            }
        )

    def to(self, device: str | torch.device, *, dtype: torch.dtype = None) -> "TypedSample":
        """Return a shallow copy; consumers can call ``.to`` on sub-views explicitly."""

        return replace(self)


__all__ = [
    "TypedSample",
    "CameraView",
    "TrajectoryView",
    "SemiDenseView",
    "GTView",
    "Obb3View",
]
