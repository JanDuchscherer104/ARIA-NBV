"""Typed, PyTorch-friendly ASE/ATEK dataset wrapper with GT mesh pairing.

This module wraps ATEK's WebDataset loader and converts the flattened samples
into typed dataclasses that are IDE-friendly and EFM3D-compatible. Each yielded
sample carries the original `AtekDataSample`, typed views of camera/trajectory
fields, optional GT mesh, and helper utilities to remap keys to the EFM3D
schema.

All tensors follow the Aria camera frame (x-left, y-up, z-forward). Transforms
use the `T_A_B` convention (pose that maps points from frame B into frame A).
"""
# ruff: noqa: F722,F821,UP037

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import trimesh
from atek.data_loaders.atek_wds_dataloader import (
    load_atek_wds_dataset,
    select_and_remap_dict_keys,
)
from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    MpsSemiDensePointData,
    MpsTrajData,
    MultiFrameCameraData,
    create_atek_data_sample_from_flatten_dict,
)
from devtools import pformat

try:  # Prefer installed efm3d, otherwise fall back to vendored source.
    from efm3d.aria import CameraTW, PoseTW
    from efm3d.dataset.efm_model_adaptor import EfmModelAdaptor
except ModuleNotFoundError:  # pragma: no cover - exercised in CI fallback
    vendor_root = Path(__file__).resolve().parents[3] / "external" / "efm3d"
    sys.path.append(str(vendor_root))
    from efm3d.aria import CameraTW, PoseTW
    from efm3d.dataset.efm_model_adaptor import EfmModelAdaptor
from jaxtyping import Float, Int
from pydantic import Field, field_validator, model_validator
from torch import Tensor
from torch.utils.data import IterableDataset

from ..configs import PathConfig
from ..utils import BaseConfig, Console

SEQ_PATTERN = re.compile(r".*?(?P<scene_id>\\d{4,})(?:_|-)(?P<snippet_id>[\\w-]+)")


class CameraLabel(str, Enum):
    """Canonical camera stream labels used by ATEK."""

    RGB = "camera-rgb"
    SLAM_LEFT = "camera-slam-left"
    SLAM_RIGHT = "camera-slam-right"
    RGB_DEPTH = "camera-rgb-depth"


@dataclass(slots=True)
class CameraStream:
    """Typed view of `MultiFrameCameraData`.

    Attributes:
        label: ATEK camera label (e.g. ``"camera-rgb"``).
        images: Float tensor shaped ``[frames, channels, height, width]`` in the
            camera frame.
        projection_params: Float tensor of intrinsics per frame
            ``[frames, num_params]``.
        t_device_camera: Pose ``T_device_camera`` per frame shaped ``[frames, 3, 4]``.
        capture_timestamps_ns: Frame timestamps in nanoseconds ``[frames]``.
        frame_ids: Frame indices ``[frames]``.
        exposure_durations_s: Exposure duration per frame in seconds ``[frames]``.
        gains: Analog gain per frame ``[frames]``.
        camera_model_name: Camera model identifier (e.g. ``"fisheye624"``).
        camera_valid_radius: Valid pixel radius mask. Shape ``[frames, 1]`` or ``[1]``.
    """

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
        """Create a typed stream from a raw ATEK dataclass."""
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
        """Return poses as ``PoseTW`` batch if available."""
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

    @staticmethod
    def from_mps(data: MpsSemiDensePointData | None) -> SemiDensePoints | None:
        if data is None:
            return None
        return SemiDensePoints(
            points_world=data.points_world,
            points_dist_std=data.points_dist_std,
            points_inv_dist_std=data.points_inv_dist_std,
            capture_timestamps_ns=data.capture_timestamps_ns,
        )

    def stacked(self) -> Float[Tensor, "frames points 3"] | None:  # noqa: F722,F821
        """Stack per-frame points into shape ``[frames, points, 3]`` (NaNs preserved)."""
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
        """Return the original dictionary for compatibility with downstream loaders."""
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
        """Build a typed snippet from a flattened ATEK dict."""
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
        """Return the original flattened dict (unchanged)."""
        flat = dict(self.flat)
        flat["gt_data"] = self.gt_data.to_raw()
        return flat


@dataclass(slots=True)
class ASESample:
    """Single ASE snippet with typed ATEK data and optional GT mesh."""

    scene_id: str
    snippet_id: str
    atek: AtekSnippet
    gt_mesh: trimesh.Trimesh | None

    @property
    def has_mesh(self) -> bool:
        return self.gt_mesh is not None

    @property
    def has_rgb(self) -> bool:
        return self.atek.camera_rgb is not None and self.atek.camera_rgb.images is not None

    @property
    def has_slam_points(self) -> bool:
        return self.atek.semidense is not None and bool(self.atek.semidense.points_world)

    @property
    def has_depth(self) -> bool:
        return self.atek.camera_rgb_depth is not None and self.atek.camera_rgb_depth.images is not None

    def to_flatten_dict(self) -> dict[str, Any]:
        """Return a flat dict compatible with EFM3D + mesh metadata."""
        flat = self.atek.to_flatten_dict()
        flat["scene_id"] = self.scene_id
        flat["snippet_id"] = self.snippet_id
        flat["gt_mesh"] = self.gt_mesh
        return flat

    def to_efm_dict(self, include_mesh: bool = True) -> dict[str, Any]:
        """Remap keys to the EFM3D schema using ``EfmModelAdaptor`` mappings."""
        remapped = select_and_remap_dict_keys(
            sample_dict=self.atek.to_flatten_dict(),
            key_mapping=EfmModelAdaptor.get_dict_key_mapping_all(),
        )
        remapped["scene_id"] = self.scene_id
        remapped["snippet_id"] = self.snippet_id
        if include_mesh:
            remapped["gt_mesh"] = self.gt_mesh
        return remapped

    def __repr__(self) -> str:
        mesh_info = (
            f"mesh=Trimesh(V={len(self.gt_mesh.vertices)},F={len(self.gt_mesh.faces)})"
            if self.gt_mesh is not None
            else "mesh=None"
        )
        cams = ", ".join(sorted([c.name for c in self.atek.cameras.keys()])) or "no-cams"
        semidense = (
            f"{len(self.atek.semidense.points_world)} frames"
            if self.atek.semidense and self.atek.semidense.points_world
            else "no-points"
        )
        return f"ASESample(scene={self.scene_id}, snippet={self.snippet_id}, cams={cams}, semidense={semidense}, {mesh_info})"


def _parse_sequence_name(sequence_name: str) -> tuple[str, str]:
    """Extract scene/snippet identifiers from a sequence name."""
    if not sequence_name:
        return "unknown", "unknown"
    match = SEQ_PATTERN.match(sequence_name)
    if match:
        return match.group("scene_id"), match.group("snippet_id")
    parts = sequence_name.split("_")
    if len(parts) >= 2:
        return parts[1], "_".join(parts[2:]) if len(parts) > 2 else "unknown"
    return sequence_name or "unknown", "unknown"


def _explode_batched_dict(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split a batched dict (B-first) into a list of per-sample dicts."""
    batch_size: int | None = None
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            batch_size = value.shape[0]
            break
        if isinstance(value, list):
            batch_size = len(value)
            break
    if batch_size is None:
        return [dict(batch)]

    per_item: list[dict[str, Any]] = []
    for idx in range(batch_size):
        item: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                item[key] = value[idx]
            elif isinstance(value, list):
                item[key] = value[idx]
            else:
                item[key] = value
        per_item.append(item)
    return per_item


def ase_collate(batch: Sequence[ASESample]) -> dict[str, Any]:
    """Collate function for ``torch.utils.data.DataLoader``."""
    return {
        "scene_id": [sample.scene_id for sample in batch],
        "snippet_id": [sample.snippet_id for sample in batch],
        "atek": [sample.atek for sample in batch],
        "efm": [sample.to_efm_dict() for sample in batch],
        "gt_mesh": [sample.gt_mesh for sample in batch],
    }


class ASEDataset(IterableDataset[ASESample]):
    """Iterable dataset yielding typed ASE snippets with paired GT meshes."""

    def __init__(self, config: ASEDatasetConfig):
        super().__init__()
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbose(config.verbose)

        self.console.log(f"Loading ATEK WebDataset from {len(config.tar_urls)} shards")
        self._atek_wds = load_atek_wds_dataset(
            urls=config.tar_urls,
            batch_size=config.batch_size,
            shuffle_flag=config.shuffle,
            repeat_flag=config.repeat,
        )
        self._mesh_cache: dict[str, trimesh.Trimesh] = {}
        self.console.log("ATEK loader ready")

    def _load_mesh(self, scene_id: str) -> trimesh.Trimesh | None:
        """Load and optionally cache the GT mesh for a scene."""
        if scene_id in self._mesh_cache:
            return self._mesh_cache[scene_id]

        mesh_path = self.config.scene_to_mesh.get(scene_id)
        if mesh_path is None or not mesh_path.exists():
            if self.config.require_mesh:
                raise FileNotFoundError(f"GT mesh for scene {scene_id} not found (config.require_mesh=True).")
            return None

        mesh = trimesh.load(str(mesh_path), process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Loaded mesh for scene {scene_id} is not a Trimesh: {type(mesh)}")

        if self.config.mesh_simplify_ratio is not None:
            original_faces = len(mesh.faces)
            mesh = mesh.simplify_quadric_decimation(self.config.mesh_simplify_ratio)
            self.console.dbg(f"Simplified mesh {scene_id}: {original_faces:,} -> {len(mesh.faces):,} faces")

        if self.config.cache_meshes:
            self._mesh_cache[scene_id] = mesh
        return mesh

    def _iter_flat_samples(self) -> Iterator[dict[str, Any]]:
        """Iterate over flattened ATEK dict samples (handling wds batches)."""
        for raw in self._atek_wds:
            if isinstance(raw, Mapping):
                samples = _explode_batched_dict(raw) if self.config.batch_size else [dict(raw)]
                for sample in samples:
                    yield sample
            elif isinstance(raw, (list, tuple)):
                for sample in raw:
                    if not isinstance(sample, Mapping):
                        raise TypeError(f"Unexpected sample type inside batch: {type(sample)}")
                    yield from (_explode_batched_dict(sample) if self.config.batch_size else [dict(sample)])
            else:
                raise TypeError(f"Unexpected sample type from ATEK loader: {type(raw)}")

    def __iter__(self) -> Iterator[ASESample]:
        """Yield typed samples that can be consumed by a PyTorch DataLoader."""
        for flat_dict in self._iter_flat_samples():
            atek = AtekSnippet.from_flat(flat_dict)
            scene_id, snippet_id = _parse_sequence_name(atek.sequence_name)
            mesh = self._load_mesh(scene_id) if self.config.load_meshes else None

            yield ASESample(
                scene_id=scene_id,
                snippet_id=snippet_id,
                atek=atek,
                gt_mesh=mesh,
            )


class ASEDatasetConfig(BaseConfig[ASEDataset]):
    """Configuration for :class:`ASEDataset`.

    Attributes:
        target: Factory target (Dataset class).
        paths: Shared path configuration.
        tar_urls: Absolute tar file paths or glob patterns.
        scene_to_mesh: Mapping ``scene_id -> mesh path``.
        atek_variant: Subdirectory name under ``.data/ase_atek`` (e.g. ``"efm"``).
        scene_ids: Optional scene filter used to auto-resolve tar paths and meshes.
        batch_size: WebDataset batch size (set ``None`` to let DataLoader handle batching).
        shuffle: Enable WebDataset shuffling.
        repeat: Repeat dataset indefinitely.
        load_meshes: Whether to attach GT meshes to each sample.
        require_mesh: Raise if a mesh is missing for a scene.
        mesh_simplify_ratio: Optional decimation ratio (0-1).
        cache_meshes: Cache loaded meshes in memory.
        verbose: Enable verbose Console logging.
    """

    target: type[ASEDataset] = Field(default=ASEDataset, exclude=True)
    paths: PathConfig = Field(default_factory=PathConfig)

    tar_urls: list[str] = Field(default_factory=list)
    scene_to_mesh: dict[str, Path] = Field(default_factory=dict)
    atek_variant: str = Field(default="efm")
    scene_ids: list[str] | None = Field(default=None)
    atek_root: Path | None = Field(
        default=None, description="Override root directory that contains per-scene ATEK shards."
    )

    batch_size: int | None = Field(
        default=None, description="WebDataset batch size; prefer None and use DataLoader batching."
    )
    shuffle: bool = Field(default=False)
    repeat: bool = Field(default=False)

    load_meshes: bool = Field(default=True)
    require_mesh: bool = Field(default=False)

    mesh_simplify_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    cache_meshes: bool = Field(default=True)

    is_debug: bool = False
    verbose: bool = Field(default=True)

    @field_validator("tar_urls", mode="before")
    @classmethod
    def _normalize_tar_urls(cls, value: list[str | Path] | str | Path) -> list[str]:
        if isinstance(value, (str, Path)):
            value = [value]
        return [str(v) for v in value]

    @field_validator("atek_root", mode="before")
    @classmethod
    def _normalize_atek_root(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        return path

    def _resolve_atek_root(self) -> Path:
        """Resolve the directory that stores per-scene tar shards."""
        if self.atek_root is not None:
            base = self.atek_root
            if not base.is_absolute():
                base = self.paths.data_root / base
            return base

        candidates = [
            self.paths.data_root / "ase_atek" / self.atek_variant,
            self.paths.data_root / f"ase_{self.atek_variant}",
            self.paths.data_root / "ase_efm",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @model_validator(mode="after")
    def _autofill_paths(self) -> ASEDatasetConfig:
        """Populate tar paths and mesh mapping from scene IDs when unset."""
        base = self._resolve_atek_root()

        # If no tar URLs provided, auto-discover
        if not self.tar_urls:
            if self.scene_ids:
                resolved: list[str] = []
                for scene in self.scene_ids:
                    resolved.extend(str(p) for p in sorted((base / scene).glob("*.tar")))
                self.tar_urls = resolved
            else:
                self.tar_urls = [str(p) for p in sorted(base.glob("**/*.tar"))]

        if self.load_meshes and not self.scene_to_mesh:
            if self.scene_ids:
                self.scene_to_mesh = {scene: self.paths.resolve_mesh_path(scene) for scene in self.scene_ids}
            else:
                # discover meshes for all scenes present under ase_meshes
                mesh_dir = self.paths.ase_meshes
                self.scene_to_mesh = {
                    p.stem.replace("scene_ply_", ""): p
                    for p in mesh_dir.glob("scene_ply_*.ply")
                }

        if not self.tar_urls:
            raise ValueError("No tar files configured. Provide `tar_urls` or `scene_ids`.")
        return self

    def setup_target(self) -> ASEDataset:  # type: ignore[override]
        console = Console.with_prefix(self.__class__.__name__, "setup_target").set_verbose(self.verbose)
        expanded_urls: list[str] = []
        for url in self.tar_urls:
            if any(ch in url for ch in "*?[]"):
                matches = sorted(Path().glob(url))
                if not matches:
                    raise FileNotFoundError(f"No tar files matched glob: {url}")
                expanded_urls.extend(str(p) for p in matches)
            else:
                expanded_urls.append(url)

        self.tar_urls = expanded_urls

        console.log(f"Preparing ASEDataset (tar URLs: {len(self.tar_urls)}, scenes: {len(self.scene_to_mesh)})")
        return self.target(self)

    def __repr__(self) -> str:
        # TODO: add informative repr
        return pformat("")


__all__ = [
    "ASEDataset",
    "ASEDatasetConfig",
    "ASESample",
    "AtekSnippet",
    "CameraLabel",
    "ase_collate",
]
