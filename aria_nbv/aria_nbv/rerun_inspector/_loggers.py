"""Rerun logging primitives for offline VIN inspector samples."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.obb import ObbTW
from efm3d.aria.pose import PoseTW
from efm3d.aria.tensor_wrapper import TensorWrapper
from numpy.typing import DTypeLike, NDArray
from pytorch3d.renderer.cameras import PerspectiveCameras
from torch import Tensor

from aria_nbv.data_handling import VinOfflineSample

from ._config import RerunOfflineInspectorConfig
from ._metadata import OfflineVisualInventory, build_sample_metadata_document

RerunEntityFactory: TypeAlias = Callable[..., object]
LineStripPayload: TypeAlias = list[Any]


class RerunModule(Protocol):
    """Subset of the Rerun module used by the offline inspector."""

    Points3D: RerunEntityFactory
    LineStrips3D: RerunEntityFactory
    TextDocument: RerunEntityFactory
    Transform3D: RerunEntityFactory
    Mesh3D: RerunEntityFactory
    Image: RerunEntityFactory
    DepthImage: RerunEntityFactory
    Pinhole: RerunEntityFactory
    ViewCoordinates: Any
    TransformRelation: Any

    def init(self, *args: object, **kwargs: object) -> None:
        """Initialize a recording."""

    def save(self, *args: object, **kwargs: object) -> None:
        """Open a save sink."""

    def spawn(self, *args: object, **kwargs: object) -> None:
        """Open a viewer sink."""

    def connect_grpc(self, *args: object, **kwargs: object) -> None:
        """Connect to a Rerun server."""

    def log(self, entity_path: str, entity: object, *args: object, **kwargs: object) -> None:
        """Log one entity."""


ENTITY_WORLD = "world"
ENTITY_SEMIDENSE = "world/semidense"
ENTITY_REFERENCE_POSE = "world/reference/rig"
ENTITY_FRUSTA_ALL = "world/candidates/frusta/all"
ENTITY_FRUSTA_TOP_ORACLE = "world/candidates/frusta/top_oracle"
ENTITY_FRUSTA_INVALID = "world/candidates/frusta/invalid"
ENTITY_CANDIDATE_CENTERS = "world/candidates/centers"
ENTITY_METADATA_SAMPLE = "metadata/sample"
ENTITY_MESH = "world/mesh"
ENTITY_CANDIDATE_POINTS = "world/candidates/points"
ENTITY_CANDIDATE_DEPTHS = "world/candidates"
ENTITY_GT_OBBS = "world/gt/obbs"
ENTITY_DETECTED_OBBS = "world/detected/obbs"
ENTITY_TRAJECTORY = "world/trajectory/rig"
ENTITY_RGB_KEYFRAMES = "world/keyframes/rgb"
ENTITY_DEPTH_KEYFRAMES = "world/keyframes/depth"

_FRUSTUM_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),
)

_OBB_EDGES = (
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
)


def _to_numpy(value: object, *, dtype: DTypeLike = np.float32) -> NDArray[Any]:
    """Convert tensors and tensor-wrapper payloads to NumPy arrays."""

    if isinstance(value, TensorWrapper):
        value = value._data
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def deterministic_downsample(points: object, *, max_points: int, seed: int | None) -> NDArray[Any]:
    """Return a deterministic subset of ``points`` with shape ``(N, 3)``."""

    arr = _to_numpy(points).reshape(-1, 3)
    if max_points <= 0:
        return arr[:0]
    if arr.shape[0] <= max_points:
        return arr
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(arr.shape[0], size=max_points, replace=False))
    return arr[indices]


def _candidate_count(sample: VinOfflineSample) -> int:
    """Return the valid candidate width for one sample."""

    count = int(sample.oracle.candidate_count)
    pose_count = int(_to_numpy(sample.oracle.candidate_poses_world_cam._data).reshape(-1, 12).shape[0])
    return min(max(count, 0), pose_count)


def _pose_rt(poses: PoseTW, indices: Sequence[int] | None = None) -> tuple[NDArray[Any], NDArray[Any]]:
    """Extract ``R`` and ``t`` from a PoseTW-like batch."""

    r = _to_numpy(poses.R).reshape(-1, 3, 3)
    t = _to_numpy(poses.t).reshape(-1, 3)
    if indices is not None:
        idx = np.asarray(indices, dtype=np.int64)
        r = r[idx]
        t = t[idx]
    return r, t


def _p3d_param_at(values: Tensor, index: int) -> NDArray[Any]:
    """Return one PyTorch3D camera parameter row as ``float32``."""

    arr = _to_numpy(values).reshape(-1, values.shape[-1])
    if arr.shape[0] == 0:
        raise ValueError("PyTorch3D camera parameter batch is empty.")
    row = arr[0] if arr.shape[0] == 1 else arr[min(max(index, 0), arr.shape[0] - 1)]
    return np.asarray(row, dtype=np.float32)


def _p3d_pinhole_kwargs(cameras: PerspectiveCameras, index: int) -> dict[str, list[float]]:
    """Return Rerun ``Pinhole`` kwargs from a PyTorch3D camera entry.

    PyTorch3D stores ``image_size`` as ``(height, width)``; Rerun expects
    ``resolution`` as ``[width, height]``.
    """

    image_size = _p3d_param_at(cameras.image_size, index)
    focal = _p3d_param_at(cameras.focal_length, index)
    principal = _p3d_param_at(cameras.principal_point, index)
    height, width = float(image_size[0]), float(image_size[1])
    return {
        "resolution": [width, height],
        "focal_length": [float(focal[0]), float(focal[1])],
        "principal_point": [float(principal[0]), float(principal[1])],
    }


def _camera_tw_pinhole_kwargs(camera: CameraTW) -> dict[str, list[float]]:
    """Return Rerun ``Pinhole`` kwargs from one EFM ``CameraTW`` entry."""

    size = _to_numpy(camera.size).reshape(-1, 2)[0]
    focal = _to_numpy(camera.f).reshape(-1, 2)[0]
    principal = _to_numpy(camera.c).reshape(-1, 2)[0]
    return {
        "resolution": [float(size[0]), float(size[1])],
        "focal_length": [float(focal[0]), float(focal[1])],
        "principal_point": [float(principal[0]), float(principal[1])],
    }


def _fallback_candidate_centers_world(poses_world_cam: PoseTW, indices: Sequence[int]) -> NDArray[Any]:
    """Return candidate camera centers from PoseTW translations."""

    _, centers = _pose_rt(poses_world_cam, indices)
    return centers


def _fallback_frusta_line_strips(
    poses_world_cam: PoseTW,
    *,
    indices: Sequence[int],
    scale: float,
) -> tuple[list[list[list[float]]], list[str] | None]:
    """Build simple batched camera-frustum line strips in world frame.

    The helper treats candidate poses as ``T_world_cam`` and uses the positive
    camera-Z direction for display-only frustum geometry.
    """

    r, t = _pose_rt(poses_world_cam, indices)
    z = float(scale)
    xy = float(scale) * 0.55
    corners_cam = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-xy, -xy, z],
            [xy, -xy, z],
            [xy, xy, z],
            [-xy, xy, z],
        ],
        dtype=np.float32,
    )
    strips: list[list[list[float]]] = []
    for rot, trans in zip(r, t, strict=False):
        corners_world = (rot @ corners_cam.T).T + trans.reshape(1, 3)
        for start, end in _FRUSTUM_EDGES:
            strips.append([corners_world[start].tolist(), corners_world[end].tolist()])
    return strips, None


def _subset_poses(poses_world_cam: PoseTW, indices: Sequence[int]) -> PoseTW:
    """Return a PoseTW-like subset without importing data-handling internals."""

    data = poses_world_cam._data
    if data is None:
        raise ValueError("PoseTW payload is empty; cannot subset candidate poses.")
    data = data.reshape(-1, 12)
    index = torch.as_tensor(list(indices), device=data.device, dtype=torch.long)
    return cast(PoseTW, PoseTW(data.index_select(0, index)))


def _subset_cameras(cameras: PerspectiveCameras, indices: Sequence[int]) -> PerspectiveCameras:
    """Return a camera batch subset when the camera object supports indexing."""

    try:
        return cameras[list(indices)]
    except Exception:
        return cameras


def _subset_optional_1d(values: object | None, indices: Sequence[int]) -> NDArray[Any] | None:
    """Return a selected one-dimensional NumPy view for optional labels/colors."""

    if values is None:
        return None
    arr = _to_numpy(values).reshape(-1)
    return arr[np.asarray(indices, dtype=np.int64)]


def _worker_c_frusta_line_strips(
    poses_world_cam: PoseTW,
    *,
    indices: Sequence[int],
    scale: float,
    cameras: PerspectiveCameras | None = None,
    oracle_rri: Tensor | None = None,
    validity: NDArray[Any] | None = None,
) -> tuple[LineStripPayload, list[str] | None]:
    """Use Worker-C frusta helpers when available, otherwise local fallback."""

    try:
        from . import _frusta
    except ImportError:
        return _fallback_frusta_line_strips(poses_world_cam, indices=indices, scale=scale)

    p3d_helper = getattr(_frusta, "frusta_from_p3d_cameras", None)
    if callable(p3d_helper) and cameras is not None:
        result = p3d_helper(
            _subset_poses(poses_world_cam, indices),
            _subset_cameras(cameras, indices),
            depth_m=scale,
            candidate_ids=list(indices),
            oracle_rri=_subset_optional_1d(oracle_rri, indices),
            validity=_subset_optional_1d(validity, indices),
        )
        return list(result.line_strips), list(result.labels)

    for name in ("build_rerun_frusta_line_strips", "build_frusta_line_strips"):
        helper = getattr(_frusta, name, None)
        if callable(helper):
            return helper(poses_world_cam, indices=indices, scale=scale), None
    return _fallback_frusta_line_strips(poses_world_cam, indices=indices, scale=scale)


def _worker_c_candidate_centers(poses_world_cam: PoseTW, indices: Sequence[int]) -> NDArray[Any]:
    """Use Worker-C center helpers when available, otherwise local fallback."""

    try:
        from . import _frusta
    except ImportError:
        return _fallback_candidate_centers_world(poses_world_cam, indices)

    for name in ("candidate_centers_world", "pose_centers_world"):
        helper = getattr(_frusta, name, None)
        if callable(helper):
            return np.asarray(helper(poses_world_cam, indices=indices), dtype=np.float32)
    return _fallback_candidate_centers_world(poses_world_cam, indices)


def _rgba(name: str, count: int) -> list[list[int]]:
    """Return RGBA colors, preferring Worker-C palettes when available."""

    colors_module: object | None
    try:
        from . import _colors as colors_module
    except ImportError:
        colors_module = None

    if colors_module is not None:
        helper = getattr(colors_module, "rgba_u8", None)
        if callable(helper):
            colors = np.asarray(helper(name, count), dtype=np.uint8).reshape(count, 4).tolist()
            return cast(list[list[int]], colors)

    palette = {
        "semidense": [170, 176, 190, 180],
        "reference": [255, 210, 80, 255],
        "frusta_all": [88, 166, 255, 170],
        "frusta_top": [90, 210, 135, 255],
        "frusta_invalid": [240, 90, 85, 220],
        "centers": [255, 235, 120, 255],
        "candidate_points": [160, 225, 205, 150],
        "mesh": [130, 138, 150, 90],
        "gt_obbs": [255, 195, 80, 235],
        "detected_obbs": [120, 220, 255, 220],
        "trajectory": [255, 255, 255, 230],
    }
    return [palette.get(name, [255, 255, 255, 255]) for _ in range(count)]


def _semidense_points(sample: VinOfflineSample) -> NDArray[Any]:
    """Return semidense world points from a VIN snippet-like object."""

    snippet = sample.vin_snippet
    points = _to_numpy(snippet.points_world)
    if points.ndim < 2 or points.shape[-1] < 3:
        raise ValueError(f"Expected semidense points with XYZ channels, got shape {tuple(points.shape)}.")

    xyz = points[..., :3]
    length_values = _to_numpy(snippet.lengths, dtype=np.int64).reshape(-1)
    length = max(int(length_values[0]), 0) if length_values.size > 0 else None

    if xyz.ndim == 2:
        valid = xyz[:length] if length is not None else xyz
    else:
        batches = xyz.reshape(-1, xyz.shape[-2], 3)
        valid = batches[:, :length, :] if length is not None else batches
    return np.asarray(valid, dtype=np.float32).reshape(-1, 3)


def _validity_mask(sample: VinOfflineSample, candidate_count: int) -> NDArray[Any] | None:
    """Return a candidate validity mask when the sample exposes one."""

    if sample.candidates is None:
        return None
    return np.asarray(
        _to_numpy(sample.candidates.mask_valid, dtype=np.bool_).reshape(-1)[:candidate_count], dtype=np.bool_
    )


def _top_oracle_index(sample: VinOfflineSample, candidate_count: int, validity: NDArray[Any] | None) -> int | None:
    """Return the valid candidate with the highest oracle RRI value."""

    if candidate_count <= 0:
        return None
    scores = _to_numpy(sample.oracle.rri).reshape(-1)[:candidate_count]
    if scores.size == 0:
        return None
    if validity is not None and validity.shape[0] == scores.shape[0]:
        if not validity.any():
            return None
        scores = scores.copy()
        scores[~validity] = -np.inf
    return int(np.argmax(scores))


def _candidate_points(sample: VinOfflineSample, *, max_points: int, seed: int | None) -> NDArray[Any] | None:
    """Return optional candidate point-cloud points, downsampled deterministically."""

    candidate_pcs = sample.candidate_pcs
    if candidate_pcs is None:
        return None
    points = candidate_pcs.points
    lengths = candidate_pcs.lengths
    arr = (
        _to_numpy(points).reshape(points.shape[0], points.shape[1], 3)
        if isinstance(points, torch.Tensor)
        else _to_numpy(points).reshape(-1, 3)
    )
    if lengths is not None and arr.ndim == 3:
        len_arr = _to_numpy(lengths, dtype=np.int64).reshape(-1)
        chunks = [arr[idx, : int(length)] for idx, length in enumerate(len_arr[: arr.shape[0]]) if int(length) > 0]
        flat = np.concatenate(chunks, axis=0) if chunks else arr.reshape(0, 3)
    else:
        flat = arr.reshape(-1, 3)
    return deterministic_downsample(flat, max_points=max_points, seed=seed)


def _mesh_tensors(sample: VinOfflineSample) -> tuple[NDArray[Any], NDArray[Any]] | None:
    """Return optional mesh vertices/faces from an attached EFM snippet."""

    snippet = sample.efm_snippet_view
    if snippet is None:
        return None
    verts = snippet.mesh_verts
    faces = snippet.mesh_faces
    if verts is not None and faces is not None:
        return _to_numpy(verts), _to_numpy(faces, dtype=np.uint32)
    mesh = snippet.mesh
    if mesh is not None:
        return _to_numpy(mesh.vertices), _to_numpy(mesh.faces, dtype=np.uint32)
    return None


def _compact_or_live_gt_obbs(sample: VinOfflineSample) -> Tensor | ObbTW | None:
    """Return compact GT OBB tensor or live EFM OBB tensor."""

    if sample.gt_obbs is not None:
        return sample.gt_obbs.obbs
    snippet = sample.efm_snippet_view
    if snippet is None:
        return None
    obb_view = snippet.obbs
    if obb_view is None:
        return None
    return obb_view.obbs


def _detected_obbs(sample: VinOfflineSample) -> Tensor | ObbTW | None:
    """Return compact or diagnostic-backbone detected OBB tensor."""

    if sample.detected_obbs is not None:
        return sample.detected_obbs.obbs
    backbone = sample.backbone_out
    if backbone is None:
        return None
    return backbone.obb_pred_viz if backbone.obb_pred_viz is not None else backbone.obb_pred


def _obb_line_strips(obbs: Tensor | ObbTW) -> list[list[list[float]]]:
    """Convert EFM OBB tensors into Rerun 3D line strips."""

    arr = _to_numpy(obbs).reshape(-1, 34)
    valid = arr[~np.all(np.isclose(arr, -1.0), axis=1)]
    if valid.size == 0:
        return []
    bb3 = valid[:, :6]
    poses = valid[:, 18:30]
    corners_idx = np.asarray(
        [
            [0, 2, 4],
            [1, 2, 4],
            [1, 3, 4],
            [0, 3, 4],
            [0, 2, 5],
            [1, 2, 5],
            [1, 3, 5],
            [0, 3, 5],
        ],
        dtype=np.int64,
    )
    strips: list[list[list[float]]] = []
    for box, pose in zip(bb3, poses, strict=False):
        if not np.isfinite(box).all() or not np.isfinite(pose).all():
            continue
        corners_obj = box[corners_idx]
        rot = pose[:9].reshape(3, 3)
        trans = pose[9:12]
        corners_world = (rot @ corners_obj.T).T + trans.reshape(1, 3)
        for start, end in _OBB_EDGES:
            strips.append([corners_world[start].tolist(), corners_world[end].tolist()])
    return strips


def _trajectory_points(sample: VinOfflineSample) -> NDArray[Any]:
    """Return world-frame trajectory centers from compact metadata or VIN snippet."""

    _, centers = _pose_rt(sample.vin_snippet.t_world_rig)
    return centers


def _image_hwc(tensor: object, index: int) -> NDArray[Any]:
    """Convert a CHW image tensor in [0,1] or [0,255] to HWC uint8."""

    arr = _to_numpy(tensor)
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


def _depth_hw(tensor: object, index: int) -> NDArray[Any]:
    """Return one depth frame as ``float32`` with shape ``(H, W)``."""

    arr = _to_numpy(tensor)
    if arr.ndim == 4:
        arr = arr[index, 0]
    elif arr.ndim == 3:
        arr = arr[index]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
    return np.asarray(arr, dtype=np.float32)


def _keyframe_indices(num_frames: int) -> list[int]:
    """Return the first/last keyframe indices used for compact inspection."""

    if num_frames <= 0:
        return []
    return [0] if num_frames <= 1 else [0, num_frames - 1]


def _candidate_camera_entity(candidate_idx: int) -> str:
    """Return the stable per-candidate camera entity path."""

    return f"{ENTITY_CANDIDATE_DEPTHS}/candidate_{candidate_idx:03d}/camera"


class RerunOfflineLogger:
    """Stateful logger that writes one selected offline sample to Rerun."""

    def __init__(self, config: RerunOfflineInspectorConfig, *, rr_module: RerunModule | None = None) -> None:
        """Create the logger.

        Args:
            config: Inspector configuration.
            rr_module: Optional fake or imported ``rerun`` module.
        """

        self.config = config
        if rr_module is None:
            import rerun as imported_rr

            self.rr = cast(RerunModule, imported_rr)
        else:
            self.rr = rr_module
        self._metadata_warnings: list[str] = []

    def _warn_metadata(self, message: str) -> None:
        """Record a non-fatal visualization warning for the metadata panel."""

        if message not in self._metadata_warnings:
            self._metadata_warnings.append(message)

    def _log_world_coordinates(self) -> None:
        """Declare the Rerun scene root as ARIA's right-handed Z-up world."""

        self.rr.log(ENTITY_WORLD, self.rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def _log_pose_transform(
        self,
        entity_path: str,
        pose_world_child: PoseTW,
        *,
        axis_length: float | None = None,
    ) -> None:
        """Log a ``T_world_child`` pose with explicit parent-from-child relation."""

        r, t = _pose_rt(pose_world_child, [0])
        kwargs: dict[str, object] = {
            "translation": t[0].tolist(),
            "mat3x3": r[0].tolist(),
            "relation": self.rr.TransformRelation.ParentFromChild,
        }
        if axis_length is not None:
            kwargs["axis_length"] = axis_length
        self.rr.log(entity_path, self.rr.Transform3D(**kwargs), static=True)

    def start(self) -> None:
        """Initialize the Rerun recording and open the configured sink before logging."""

        output = self.config.output
        self.rr.init(output.application_id, recording_id=output.recording_id)
        if output.mode == "save":
            output.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.rr.save(output.save_path)
        elif output.mode == "spawn":
            self.rr.spawn(
                port=output.spawn_port,
                connect=True,
                memory_limit=output.spawn_memory_limit,
                hide_welcome_screen=output.hide_welcome_screen,
            )
        elif output.mode == "connect":
            self.rr.connect_grpc(output.connect_addr)
        else:  # pragma: no cover - pydantic constrains this.
            raise ValueError(f"Unsupported Rerun output mode: {output.mode}")
        self._log_world_coordinates()

    def log_sample(
        self,
        *,
        sample: VinOfflineSample,
        inventory: OfflineVisualInventory,
        selection: str,
    ) -> None:
        """Log one selected sample to stable Rerun entity paths."""

        del selection
        if self.config.primitives.log_semidense:
            self._log_semidense(sample)
        if self.config.primitives.log_reference_pose:
            self._log_reference_pose(sample)
        if (
            self.config.primitives.log_candidate_frusta
            or self.config.primitives.log_top_oracle_frustum
            or self.config.primitives.log_invalid_frusta
            or self.config.primitives.log_candidate_centers
        ):
            self._log_candidates(sample, inventory=inventory)
        if self.config.primitives.log_candidate_points and inventory.has_candidate_points:
            self._log_candidate_points(sample)
        if self.config.primitives.log_candidate_depths and inventory.has_candidate_depths:
            self._log_candidate_depths(sample)
        if (self.config.primitives.log_mesh or self.config.primitives.log_gt_mesh) and inventory.has_mesh:
            self._log_mesh(sample)
        if self.config.primitives.log_gt_obbs and inventory.has_gt_obbs:
            self._log_obbs(sample, entity_path=ENTITY_GT_OBBS, obbs=_compact_or_live_gt_obbs(sample), palette="gt_obbs")
        if self.config.primitives.log_detected_obbs and inventory.has_detected_obbs:
            self._log_obbs(
                sample,
                entity_path=ENTITY_DETECTED_OBBS,
                obbs=_detected_obbs(sample),
                palette="detected_obbs",
            )
        if self.config.primitives.log_gt_trajectory and inventory.has_trajectory:
            self._log_trajectory(sample)
        if self.config.primitives.log_rgb_keyframes and inventory.has_rgb_keyframes:
            self._log_rgb_keyframes(sample)
        if self.config.primitives.log_depth_keyframes and inventory.has_depth_keyframes:
            self._log_depth_keyframes(sample)

    def log_metadata(
        self,
        *,
        sample: VinOfflineSample,
        inventory: OfflineVisualInventory,
        selection: str,
    ) -> None:
        """Log sample metadata as a Rerun text document."""

        if not self.config.primitives.log_metadata:
            return
        document = build_sample_metadata_document(
            config=self.config,
            inventory=inventory,
            selection=selection,
            sample=sample,
            runtime_warnings=self._metadata_warnings,
        )
        self.rr.log(ENTITY_METADATA_SAMPLE, self.rr.TextDocument(document, media_type="application/json"), static=True)

    def _log_semidense(self, sample: VinOfflineSample) -> None:
        points = deterministic_downsample(
            _semidense_points(sample),
            max_points=self.config.performance.max_semidense_points,
            seed=self.config.performance.seed,
        )
        self.rr.log(
            ENTITY_SEMIDENSE,
            self.rr.Points3D(
                points,
                radii=self.config.geometry.semidense_radius,
                colors=_rgba("semidense", points.shape[0]),
            ),
            static=True,
        )

    def _log_reference_pose(self, sample: VinOfflineSample) -> None:
        self._log_pose_transform(
            ENTITY_REFERENCE_POSE,
            sample.oracle.reference_pose_world_rig,
            axis_length=self.config.geometry.reference_axis_length,
        )

    def _log_candidates(self, sample: VinOfflineSample, *, inventory: OfflineVisualInventory) -> None:
        del inventory
        poses = sample.oracle.candidate_poses_world_cam
        cameras = sample.oracle.p3d_cameras
        rri = sample.oracle.rri
        count = _candidate_count(sample)
        indices = list(range(count))
        validity = _validity_mask(sample, count)

        if self.config.primitives.log_candidate_frusta and indices:
            strips, labels = _worker_c_frusta_line_strips(
                poses,
                indices=indices,
                scale=self.config.geometry.frustum_scale,
                cameras=cameras,
                oracle_rri=rri,
                validity=validity,
            )
            self.rr.log(
                ENTITY_FRUSTA_ALL,
                self.rr.LineStrips3D(strips, colors=_rgba("frusta_all", len(strips)), labels=labels),
                static=True,
            )

        top_idx = _top_oracle_index(sample, count, validity)
        if self.config.primitives.log_top_oracle_frustum and top_idx is not None:
            strips, labels = _worker_c_frusta_line_strips(
                poses,
                indices=[top_idx],
                scale=self.config.geometry.frustum_scale,
                cameras=cameras,
                oracle_rri=rri,
                validity=validity,
            )
            self.rr.log(
                ENTITY_FRUSTA_TOP_ORACLE,
                self.rr.LineStrips3D(strips, colors=_rgba("frusta_top", len(strips)), labels=labels),
                static=True,
            )

        if self.config.primitives.log_invalid_frusta and validity is not None and validity.shape[0] == count:
            invalid = [idx for idx, is_valid in enumerate(validity.tolist()) if not is_valid]
            if invalid:
                strips, labels = _worker_c_frusta_line_strips(
                    poses,
                    indices=invalid,
                    scale=self.config.geometry.frustum_scale,
                    cameras=cameras,
                    oracle_rri=rri,
                    validity=validity,
                )
                self.rr.log(
                    ENTITY_FRUSTA_INVALID,
                    self.rr.LineStrips3D(strips, colors=_rgba("frusta_invalid", len(strips)), labels=labels),
                    static=True,
                )

        if self.config.primitives.log_candidate_centers and indices:
            centers = _worker_c_candidate_centers(poses, indices)
            self.rr.log(
                ENTITY_CANDIDATE_CENTERS,
                self.rr.Points3D(
                    centers,
                    radii=self.config.geometry.candidate_center_radius,
                    colors=_rgba("centers", centers.shape[0]),
                ),
                static=True,
            )

    def _log_candidate_points(self, sample: VinOfflineSample) -> None:
        points = _candidate_points(
            sample,
            max_points=self.config.performance.max_candidate_points,
            seed=self.config.performance.seed,
        )
        if points is None:
            return
        self.rr.log(
            ENTITY_CANDIDATE_POINTS,
            self.rr.Points3D(
                points,
                radii=self.config.geometry.candidate_point_radius,
                colors=_rgba("candidate_points", points.shape[0]),
            ),
            static=True,
        )

    def _log_mesh(self, sample: VinOfflineSample) -> None:
        mesh = _mesh_tensors(sample)
        if mesh is None:
            return
        verts, faces = mesh
        self.rr.log(
            ENTITY_MESH,
            self.rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=faces,
                albedo_factor=_rgba("mesh", 1)[0],
            ),
            static=True,
        )

    def _log_obbs(
        self,
        sample: VinOfflineSample,
        *,
        entity_path: str,
        obbs: Tensor | ObbTW | None,
        palette: str,
    ) -> None:
        del sample
        if obbs is None:
            return
        strips = _obb_line_strips(obbs)
        if not strips:
            return
        self.rr.log(
            entity_path,
            self.rr.LineStrips3D(strips, colors=_rgba(palette, len(strips))),
            static=True,
        )

    def _log_trajectory(self, sample: VinOfflineSample) -> None:
        points = _trajectory_points(sample)
        if points.shape[0] < 2:
            return
        self.rr.log(
            ENTITY_TRAJECTORY,
            self.rr.LineStrips3D(
                [points.tolist()],
                colors=_rgba("trajectory", 1),
                radii=self.config.geometry.trajectory_radius,
            ),
            static=True,
        )

    def _live_keyframe_contexts(
        self,
        sample: VinOfflineSample,
        frame_indices: Sequence[int],
    ) -> list[tuple[int, PoseTW, CameraTW]]:
        """Return ``(frame_index, T_world_cam, CameraTW)`` for live RGB frames."""

        snippet = sample.efm_snippet_view
        if snippet is None:
            self._warn_metadata("Live keyframe logging skipped: sample has no attached EFM snippet.")
            return []
        camera = snippet.camera_rgb
        try:
            trajectory = snippet.trajectory
            cam_idx, traj_idx = camera.nearest_traj_indices(trajectory.time_ns, list(frame_indices), default_last=True)
            contexts: list[tuple[int, PoseTW, CameraTW]] = []
            for cam_i_t, traj_i_t in zip(cam_idx.reshape(-1), traj_idx.reshape(-1), strict=False):
                cam_i = int(cam_i_t.item())
                traj_i = int(traj_i_t.item())
                t_world_rig = trajectory.t_world_rig[traj_i]
                t_cam_rig = camera.calib.T_camera_rig[cam_i]
                t_world_cam = t_world_rig @ t_cam_rig.inverse()
                contexts.append((cam_i, t_world_cam, camera.calib[cam_i]))
        except Exception as exc:
            self._warn_metadata(f"Live keyframe logging skipped: failed to derive camera context ({exc}).")
            return []
        return contexts

    def _log_rgb_keyframes(self, sample: VinOfflineSample) -> None:
        snippet = sample.efm_snippet_view
        if snippet is None:
            self._warn_metadata("RGB keyframes skipped: sample has no attached EFM snippet.")
            return
        camera = snippet.camera_rgb
        for idx, pose_world_cam, camera_tw in self._live_keyframe_contexts(
            sample,
            _keyframe_indices(int(camera.images.shape[0])),
        ):
            camera_entity = f"{ENTITY_RGB_KEYFRAMES}/frame_{idx:03d}/camera"
            image_entity = f"{camera_entity}/image"
            self._log_pose_transform(camera_entity, pose_world_cam)
            self.rr.log(
                image_entity,
                self.rr.Pinhole(
                    **_camera_tw_pinhole_kwargs(camera_tw),
                    camera_xyz=self.rr.ViewCoordinates.LUF,
                ),
                self.rr.Image(_image_hwc(camera.images, idx)),
                static=True,
            )

    def _log_depth_keyframes(self, sample: VinOfflineSample) -> None:
        snippet = sample.efm_snippet_view
        if snippet is None:
            self._warn_metadata("Depth keyframes skipped: sample has no attached EFM snippet.")
            return
        depth = snippet.camera_rgb.distance_m
        if depth is None:
            self._warn_metadata("Depth keyframes skipped: attached RGB camera has no metric depth.")
            return
        for idx, pose_world_cam, camera_tw in self._live_keyframe_contexts(
            sample,
            _keyframe_indices(int(depth.shape[0])),
        ):
            camera_entity = f"{ENTITY_DEPTH_KEYFRAMES}/frame_{idx:03d}/camera"
            depth_entity = f"{camera_entity}/depth"
            self._log_pose_transform(camera_entity, pose_world_cam)
            self.rr.log(
                depth_entity,
                self.rr.Pinhole(
                    **_camera_tw_pinhole_kwargs(camera_tw),
                    camera_xyz=self.rr.ViewCoordinates.LUF,
                ),
                self.rr.DepthImage(_depth_hw(depth, idx), meter=1.0),
                static=True,
            )

    def _log_candidate_depths(self, sample: VinOfflineSample) -> None:
        if sample.depths is None:
            return
        arr = _to_numpy(sample.depths.depths)
        if arr.ndim < 3 or arr.shape[0] == 0:
            return
        candidate_count = min(_candidate_count(sample), int(arr.shape[0]))
        if candidate_count <= 0:
            return
        indices = {0, candidate_count - 1}
        top_idx = _top_oracle_index(sample, candidate_count, _validity_mask(sample, candidate_count))
        if top_idx is not None:
            indices.add(top_idx)
        for idx in sorted(indices):
            image = arr[idx]
            if image.ndim == 3 and image.shape[0] == 1:
                image = image[0]
            camera_entity = _candidate_camera_entity(idx)
            depth_entity = f"{camera_entity}/depth"
            self._log_pose_transform(camera_entity, _subset_poses(sample.oracle.candidate_poses_world_cam, [idx]))
            self.rr.log(
                depth_entity,
                self.rr.Pinhole(
                    **_p3d_pinhole_kwargs(sample.oracle.p3d_cameras, idx),
                    camera_xyz=self.rr.ViewCoordinates.LUF,
                ),
                self.rr.DepthImage(np.asarray(image, dtype=np.float32), meter=1.0),
                static=True,
            )


__all__ = [
    "ENTITY_CANDIDATE_CENTERS",
    "ENTITY_CANDIDATE_DEPTHS",
    "ENTITY_CANDIDATE_POINTS",
    "ENTITY_DETECTED_OBBS",
    "ENTITY_DEPTH_KEYFRAMES",
    "ENTITY_FRUSTA_ALL",
    "ENTITY_FRUSTA_INVALID",
    "ENTITY_FRUSTA_TOP_ORACLE",
    "ENTITY_GT_OBBS",
    "ENTITY_MESH",
    "ENTITY_METADATA_SAMPLE",
    "ENTITY_REFERENCE_POSE",
    "ENTITY_RGB_KEYFRAMES",
    "ENTITY_SEMIDENSE",
    "ENTITY_TRAJECTORY",
    "ENTITY_WORLD",
    "RerunOfflineLogger",
    "deterministic_downsample",
]
