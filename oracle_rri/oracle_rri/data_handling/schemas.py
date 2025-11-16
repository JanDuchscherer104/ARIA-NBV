"""Typed ATEK/ASE data structures used across the dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import torch
from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    MpsSemiDensePointData,
    MpsTrajData,
    MultiFrameCameraData,
    create_atek_data_sample_from_flatten_dict,
)
from efm3d.aria import CameraTW, PoseTW
from jaxtyping import Float, Int
from torch import Tensor


class CameraLabel(str, Enum):
    """Canonical camera stream labels used by ATEK."""

    RGB = "camera-rgb"
    SLAM_LEFT = "camera-slam-left"
    SLAM_RIGHT = "camera-slam-right"
    RGB_DEPTH = "camera-rgb-depth"


@dataclass(slots=True)
class CameraStream:
    """Typed view of `MultiFrameCameraData`."""

    label: CameraLabel
    images: Float[Tensor, "frames channels height width"] | None  # noqa: F722,F821
    projection_params: Float[Tensor, "frames params"] | None  # noqa: F722,F821
    t_device_camera: Float[Tensor, "frames 3 4"] | None  # noqa: F722,F821
    capture_timestamps_ns: Int[Tensor, "frames"] | None  # noqa: F722,F821
    frame_ids: Int[Tensor, "frames"] | None  # noqa: F722,F821
    exposure_durations_s: Float[Tensor, "frames"] | None  # noqa: F722,F821
    gains: Float[Tensor, "frames"] | None  # noqa: F722,F821
    camera_model_name: str | None
    camera_valid_radius: Float[Tensor, "frames 1"] | Float[Tensor, "1"] | None  # noqa: F722,F821

    @staticmethod
    def from_multiframe(camera: MultiFrameCameraData | None, label: CameraLabel) -> CameraStream | None:
        if camera is None:
            return None
        return CameraStream(
            label=label,
            images=camera.images,
            projection_params=camera.projection_params,
            t_device_camera=camera.T_Device_Camera,
            capture_timestamps_ns=camera.capture_timestamps_ns,
            frame_ids=camera.frame_ids,
            exposure_durations_s=camera.exposure_durations_s,
            gains=camera.gains,
            camera_model_name=camera.camera_model_name,
            camera_valid_radius=camera.camera_valid_radius,
        )

    def to_camera_tw(self) -> CameraTW | None:
        """Convert to ``CameraTW`` for downstream EFM3D operations."""
        if self.images is None or self.projection_params is None or self.t_device_camera is None:
            return None

        frames = self.images.shape[0]
        width = torch.full((frames, 1), self.images.shape[3], dtype=torch.float32)
        height = torch.full((frames, 1), self.images.shape[2], dtype=torch.float32)
        params = (
            self.projection_params.unsqueeze(0).expand(frames, -1)
            if self.projection_params.ndim == 1
            else self.projection_params
        )

        gain = self.gains if self.gains is not None else torch.zeros(frames, 1, dtype=torch.float32)
        exposure = (
            self.exposure_durations_s
            if self.exposure_durations_s is not None
            else torch.zeros(frames, 1, dtype=torch.float32)
        )
        valid_radius = (
            self.camera_valid_radius
            if self.camera_valid_radius is not None
            else torch.full((frames, 1), 0.0, dtype=torch.float32)
        )
        if valid_radius.dim() == 1:
            valid_radius = valid_radius.view(1, -1).expand(frames, -1)
        elif valid_radius.shape[0] == 1 and frames > 1:
            valid_radius = valid_radius.expand(frames, -1)

        return CameraTW.from_surreal(
            width=width,
            height=height,
            type_str=self.camera_model_name or "",
            params=params,
            gain=gain,
            exposure_s=exposure,
            valid_radius=valid_radius,
            T_camera_rig=PoseTW.from_matrix3x4(self.t_device_camera).inverse(),
        ).float()


@dataclass(slots=True)
class Trajectory:
    """Device trajectory in world coordinates."""

    ts_world_device: Float[Tensor, "frames 3 4"] | None  # noqa: F722,F821
    capture_timestamps_ns: Int[Tensor, "frames"] | None  # noqa: F722,F821
    gravity_in_world: Float[Tensor, "3"] | None  # noqa: F722,F821

    @staticmethod
    def from_mps(data: MpsTrajData | None) -> Trajectory | None:
        if data is None:
            return None
        return Trajectory(
            ts_world_device=data.Ts_World_Device,
            capture_timestamps_ns=data.capture_timestamps_ns,
            gravity_in_world=data.gravity_in_world,
        )

    def as_pose_tw(self) -> PoseTW | None:
        if self.ts_world_device is None:
            return None
        return PoseTW.from_matrix3x4(self.ts_world_device)


@dataclass(slots=True)
class SemiDensePoints:
    """Semi-dense SLAM point observations."""

    points_world: list[Float[Tensor, "points 3"]]  # noqa: F722,F821
    points_dist_std: list[Float[Tensor, "points"]] | None  # noqa: F722,F821
    points_inv_dist_std: list[Float[Tensor, "points"]] | None  # noqa: F722,F821
    capture_timestamps_ns: Int[Tensor, "frames"] | None  # noqa: F722,F821
    volume_min: Float[Tensor, "3"] | None  # noqa: F722,F821
    volume_max: Float[Tensor, "3"] | None  # noqa: F722,F821

    @staticmethod
    def from_mps(data: MpsSemiDensePointData | None) -> SemiDensePoints | None:
        if data is None:
            return None
        return SemiDensePoints(
            points_world=data.points_world,
            points_dist_std=data.points_dist_std,
            points_inv_dist_std=data.points_inv_dist_std,
            capture_timestamps_ns=data.capture_timestamps_ns,
            volume_min=getattr(data, "points_volume_min", None) or getattr(data, "points_volumn_min", None),
            volume_max=getattr(data, "points_volume_max", None) or getattr(data, "points_volumn_max", None),
        )

    def stacked(self) -> Float[Tensor, "frames points 3"] | None:  # noqa: F722,F821
        if not self.points_world:
            return None
        return torch.stack(self.points_world, dim=0)


@dataclass(slots=True)
class Obb3CameraView:
    """GT 3D boxes for a single camera stream."""

    instance_ids: Tensor | None
    category_ids: Tensor | None
    category_names: list[str] | None
    object_dimensions: Tensor | None  # [K,3] xyz lengths
    ts_world_object: Tensor | None  # [K,3,4]


@dataclass(slots=True)
class GTData:
    """Ground-truth annotations stored in ATEK `gt_data`."""

    obb3_gt: dict[str, Obb3CameraView] | None
    obb2_gt: dict[str, dict[str, Any]] | None
    scores: Tensor | None
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GTData":
        obb3_raw = data.get("obb3_gt")
        obb3 = None
        if isinstance(obb3_raw, dict):
            obb3 = {}
            for cam, d in obb3_raw.items():
                obb3[cam] = Obb3CameraView(
                    instance_ids=d.get("instance_ids"),
                    category_ids=d.get("category_ids"),
                    category_names=d.get("category_names"),
                    object_dimensions=d.get("object_dimensions"),
                    ts_world_object=d.get("ts_world_object"),
                )
        obb2 = data.get("obb2_gt") if isinstance(data.get("obb2_gt"), dict) else None
        scores = data.get("scores") if isinstance(data.get("scores"), Tensor) else None
        return cls(obb3_gt=obb3, obb2_gt=obb2, scores=scores, raw=data)

    def to_raw(self) -> dict[str, Any]:
        return self.raw


@dataclass(slots=True)
class AtekSnippet:
    """Typed and raw ATEK snippet."""

    sequence_name: str
    cameras: dict[CameraLabel, CameraStream]
    trajectory: Trajectory | None
    semidense: SemiDensePoints | None
    gt_data: GTData
    raw: AtekDataSample
    flat: dict[str, Any]

    @property
    def camera_rgb(self) -> CameraStream | None:
        return self.cameras.get(CameraLabel.RGB)

    @property
    def camera_slam_left(self) -> CameraStream | None:
        return self.cameras.get(CameraLabel.SLAM_LEFT)

    @property
    def camera_slam_right(self) -> CameraStream | None:
        return self.cameras.get(CameraLabel.SLAM_RIGHT)

    @property
    def camera_rgb_depth(self) -> CameraStream | None:
        return self.cameras.get(CameraLabel.RGB_DEPTH)

    @property
    def pose_world_device(self) -> PoseTW | None:
        return self.trajectory.as_pose_tw() if self.trajectory else None

    @classmethod
    def from_flat(cls, flat: Mapping[str, Any]) -> AtekSnippet:
        raw_sample = create_atek_data_sample_from_flatten_dict(flat)
        gt_data = GTData.from_dict(raw_sample.gt_data or {})

        cameras: dict[CameraLabel, CameraStream] = {}
        for label, mfcd in (
            (CameraLabel.RGB, raw_sample.camera_rgb),
            (CameraLabel.SLAM_LEFT, raw_sample.camera_slam_left),
            (CameraLabel.SLAM_RIGHT, raw_sample.camera_slam_right),
            (CameraLabel.RGB_DEPTH, raw_sample.camera_rgb_depth),
        ):
            stream = CameraStream.from_multiframe(mfcd, label=label)
            if stream is not None:
                cameras[label] = stream

        return cls(
            sequence_name=raw_sample.sequence_name or "unknown",
            cameras=cameras,
            trajectory=Trajectory.from_mps(raw_sample.mps_traj_data),
            semidense=SemiDensePoints.from_mps(raw_sample.mps_semidense_point_data),
            gt_data=gt_data,
            raw=raw_sample,
            flat=dict(flat),
        )

    def to_flatten_dict(self) -> dict[str, Any]:
        flat = dict(self.flat)
        flat["gt_data"] = self.gt_data.to_raw()
        return flat


__all__ = [
    "CameraLabel",
    "CameraStream",
    "Trajectory",
    "SemiDensePoints",
    "Obb3CameraView",
    "GTData",
    "AtekSnippet",
]
