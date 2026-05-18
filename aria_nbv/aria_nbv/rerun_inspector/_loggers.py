"""Rerun logging primitives for offline VIN inspector samples."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

import numpy as np
import torch
from efm3d.aria.aria_constants import ARIA_SNIPPET_T_WORLD_SNIPPET
from efm3d.aria.obb import ObbTW, transform_obbs
from efm3d.aria.pose import PoseTW
from efm3d.aria.tensor_wrapper import TensorWrapper
from pytorch3d.transforms import matrix_to_quaternion
from torch import Tensor

from ..utils.semantic_names import semantic_class_name
from ._colors import obb_semantic_rgba
from ._metadata import OfflineVisualInventory, build_sample_metadata_document

if TYPE_CHECKING:
    from efm3d.aria.camera import CameraTW
    from numpy.typing import DTypeLike, NDArray
    from pytorch3d.renderer.cameras import PerspectiveCameras

    from aria_nbv.data_handling import VinOfflineSample

    from ._config import RerunOfflineInspectorConfig

RerunEntityFactory: TypeAlias = Callable[..., object]
LineStripPayload: TypeAlias = list[Any]


class RerunModule(Protocol):
    """Subset of the Rerun module used by the offline inspector."""

    Points3D: RerunEntityFactory
    LineStrips3D: RerunEntityFactory
    Boxes3D: RerunEntityFactory
    AnyValues: RerunEntityFactory
    TextDocument: RerunEntityFactory
    Scalar: RerunEntityFactory
    Scalars: RerunEntityFactory
    SeriesLines: RerunEntityFactory
    SeriesPoints: RerunEntityFactory
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

    def set_time(self, timeline: str, *, sequence: int | None = None, **kwargs: object) -> None:
        """Set a timeline for subsequent logs."""

    def set_time_sequence(self, timeline: str, sequence: int, **kwargs: object) -> None:
        """Set an integer timeline for subsequent logs."""


ENTITY_WORLD = "world"
ENTITY_SEMIDENSE = "world/ase/semidense"
ENTITY_REFERENCE_POSE = "world/ase/reference/rig"
ENTITY_METADATA_SAMPLE = "metadata/sample"
ENTITY_MESH = "world/gt/mesh"
ENTITY_CANDIDATE_ROOT = "world/candidates"
ENTITY_GT_OBBS = "world/gt/obbs"
ENTITY_DETECTED_OBBS = "world/efm/obbs/detected"
ENTITY_TRAJECTORY = "world/ase/trajectory/rig"
ENTITY_RGB_KEYFRAMES = "world/ase/cameras/rgb"
ENTITY_DEPTH_KEYFRAMES = "world/ase/cameras/rgb"
ENTITY_ASE_KEYFRAME_MEDIA = "media/ase/cameras/rgb"
ENTITY_EFM_VOXELS = "world/efm/voxels"
ENTITY_EFM_VOXEL_EXTENT = "world/efm/voxels/extent"

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


def _world_view_contents() -> list[str]:
    """Return the default world-view query rules for Rerun blueprints."""

    return ["+ /world/**"]


def _hidden_world_view_paths(*, hidden_world_paths: Sequence[str] = ()) -> tuple[str, ...]:
    """Return rooted world-view entity paths hidden by default but still included."""

    return (
        _normalize_blueprint_entity_path(ENTITY_EFM_VOXELS),
        _normalize_blueprint_entity_path(ENTITY_GT_OBBS),
        *(_normalize_blueprint_entity_path(path) for path in hidden_world_paths),
    )


def _normalize_blueprint_entity_path(path: str) -> str:
    """Return a rooted entity path suitable for Rerun blueprint query rules."""

    return f"/{path.lstrip('/')}"


def log_default_inspector_blueprint(
    rr_module: RerunModule,
    *,
    hidden_world_paths: Sequence[str] = (),
) -> None:
    """Send the default inspector layout when the installed Rerun SDK supports blueprints."""

    send_blueprint = getattr(rr_module, "send_blueprint", None)
    if send_blueprint is None:
        return
    try:
        import rerun.blueprint as rrb

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="World",
                    origin="/world",
                    contents=_world_view_contents(),
                    overrides={
                        path: rrb.EntityBehavior(visible=False)
                        for path in _hidden_world_view_paths(hidden_world_paths=hidden_world_paths)
                    },
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(name="Rollout RRI", origin="/plots/rollout/rri"),
                    rrb.TimeSeriesView(name="Rollout Diagnostics", origin="/plots/rollout/diagnostics"),
                    rrb.TextDocumentView(
                        name="Metadata",
                        origin="/metadata",
                        contents=["/metadata/**"],
                    ),
                    row_shares=[2, 1, 1],
                ),
                column_shares=[3, 1],
            ),
            collapse_panels=False,
        )
        send_blueprint(blueprint, make_active=True, make_default=True)
    except Exception:
        return


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
    return cast("PoseTW", PoseTW(data.index_select(0, index)))


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
            return cast("list[list[int]]", colors)

    palette = {
        "semidense": [170, 176, 190, 180],
        "reference": [255, 210, 80, 255],
        "frusta_all": [88, 166, 255, 170],
        "frusta_top": [90, 210, 135, 255],
        "frusta_invalid": [240, 90, 85, 220],
        "centers": [255, 235, 120, 255],
        "candidate_points": [160, 225, 205, 150],
        "mesh": [130, 138, 150, 51],
        "gt_obbs": [245, 158, 11, 235],
        "detected_obbs": [59, 130, 246, 220],
        "trajectory": [148, 163, 184, 180],
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


def _semidense_quality_colors(sample: VinOfflineSample, point_count: int) -> list[list[int]]:
    """Return quality colors for VIN semidense points when a fourth channel exists."""

    snippet = sample.vin_snippet
    points = _to_numpy(snippet.points_world)
    if points.ndim < 2 or points.shape[-1] < 4:
        return _rgba("semidense", point_count)

    quality = points[..., 3]
    length_values = _to_numpy(snippet.lengths, dtype=np.int64).reshape(-1)
    length = max(int(length_values[0]), 0) if length_values.size > 0 else None
    if quality.ndim == 1:
        valid = quality[:length] if length is not None else quality
    else:
        batches = quality.reshape(-1, quality.shape[-1])
        valid = batches[:, :length] if length is not None else batches
    values = np.asarray(valid, dtype=np.float32).reshape(-1)[:point_count]
    colors = _score_rgba(values, low=[92, 118, 190, 120], high=[250, 224, 100, 230])
    if len(colors) < point_count:
        colors.extend(_rgba("semidense", point_count - len(colors)))
    return colors


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


def _candidate_status(validity: NDArray[Any] | None, candidate_idx: int) -> str:
    """Return the candidate validity path component for one candidate."""

    if validity is not None and 0 <= candidate_idx < int(validity.shape[0]) and not bool(validity[candidate_idx]):
        return "invalid"
    return "valid"


def _candidate_entity(candidate_idx: int, validity: NDArray[Any] | None) -> str:
    """Return the semantic per-candidate entity path."""

    status = _candidate_status(validity, candidate_idx)
    return f"{ENTITY_CANDIDATE_ROOT}/{status}/candidate_{candidate_idx:03d}"


def _candidate_camera_entity(candidate_idx: int, validity: NDArray[Any] | None = None) -> str:
    """Return the stable per-candidate camera entity path."""

    return f"{_candidate_entity(candidate_idx, validity)}/camera"


def _candidate_scores(sample: VinOfflineSample, candidate_count: int) -> NDArray[Any]:
    """Return candidate RRI scores clipped to the active candidate count."""

    return _to_numpy(sample.oracle.rri).reshape(-1)[:candidate_count]


def _selected_candidate_index(
    sample: VinOfflineSample,
    *,
    candidate_count: int,
    validity: NDArray[Any] | None,
    config: RerunOfflineInspectorConfig,
) -> int | None:
    """Resolve the one candidate that receives expensive detail modalities."""

    selected_index = config.candidate.selected_index
    if selected_index is not None:
        return int(selected_index) if 0 <= int(selected_index) < candidate_count else None
    if config.candidate.selected_strategy == "explicit_index":
        return None
    if config.candidate.selected_strategy == "first_valid":
        if validity is None:
            return 0 if candidate_count > 0 else None
        valid = np.flatnonzero(validity[:candidate_count])
        return int(valid[0]) if valid.size > 0 else None
    return _top_oracle_index(sample, candidate_count, validity)


def _candidate_subset_indices(
    sample: VinOfflineSample,
    *,
    candidate_count: int,
    validity: NDArray[Any] | None,
    config: RerunOfflineInspectorConfig,
) -> list[int]:
    """Resolve candidate indices to log as native camera entities."""

    all_indices = list(range(candidate_count))
    mode = config.candidate.subset_mode
    if mode == "all":
        return all_indices
    if mode == "valid_only":
        if validity is None:
            return all_indices
        return [idx for idx in all_indices if bool(validity[idx])]
    if mode == "invalid_only":
        if validity is None:
            return []
        return [idx for idx in all_indices if not bool(validity[idx])]
    if mode == "indices":
        requested = set(config.candidate.subset_indices)
        return [idx for idx in all_indices if idx in requested]
    scores = _candidate_scores(sample, candidate_count)
    if scores.size == 0:
        return []
    ranking_scores = scores.copy()
    if validity is not None and validity.shape[0] == scores.shape[0] and validity.any():
        ranking_scores[~validity] = -np.inf
    top_k = min(config.candidate.subset_top_k, len(all_indices))
    return sorted(np.argsort(-ranking_scores)[:top_k].astype(np.int64).tolist())


def _candidate_points_for_index(
    sample: VinOfflineSample,
    candidate_idx: int,
    *,
    max_points: int,
    seed: int | None,
) -> NDArray[Any] | None:
    """Return one optional candidate point cloud, downsampled deterministically."""

    candidate_pcs = sample.candidate_pcs
    if candidate_pcs is None:
        return None
    points = candidate_pcs.points
    lengths = candidate_pcs.lengths
    arr = (
        _to_numpy(points).reshape(points.shape[0], points.shape[1], 3)
        if isinstance(points, torch.Tensor)
        else _to_numpy(points)
    )
    if arr.ndim != 3 or not (0 <= candidate_idx < arr.shape[0]):
        return None
    if lengths is not None and arr.ndim == 3:
        len_arr = _to_numpy(lengths, dtype=np.int64).reshape(-1)
        length = int(len_arr[candidate_idx]) if candidate_idx < len_arr.shape[0] else arr.shape[1]
        flat = arr[candidate_idx, : max(length, 0), :]
    else:
        flat = arr[candidate_idx].reshape(-1, 3)
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
    backbone = getattr(sample, "backbone_out", None)
    if backbone is None:
        return None
    return backbone.obb_pred_viz if backbone.obb_pred_viz is not None else backbone.obb_pred


def _snippet_t_world_snippet(sample: VinOfflineSample) -> PoseTW | None:
    """Return ``T_world_snippet`` for snippet-frame OBB tensors."""

    snippet = sample.efm_snippet_view
    if snippet is not None:
        value = snippet.efm.get(ARIA_SNIPPET_T_WORLD_SNIPPET)
        if isinstance(value, PoseTW):
            return PoseTW(value._data.reshape(-1, 12)[:1])
        if isinstance(value, Tensor):
            return PoseTW(value.reshape(-1, 12)[:1])

    poses = getattr(sample.vin_snippet, "t_world_rig", None)
    if isinstance(poses, PoseTW):
        data = poses._data.reshape(-1, 12)
        if data.shape[0] > 0:
            return PoseTW(data[:1])
    return None


def _transform_snippet_obbs_to_world(obbs: Tensor | ObbTW, t_world_snippet: PoseTW) -> ObbTW:
    """Transform EFM snippet-frame OBBs into ARIA world coordinates."""

    obb = obbs if isinstance(obbs, ObbTW) else ObbTW(torch.as_tensor(_to_numpy(obbs), dtype=torch.float32))
    if obb.ndim == 1 or obb.ndim == 2:
        return obb.transform(t_world_snippet)
    if obb.ndim == 3:
        return transform_obbs(obb, t_world_snippet)
    if obb.ndim == 4:
        return transform_obbs(obb, t_world_snippet.unsqueeze(0))
    raise ValueError(f"Unsupported OBB tensor rank for Rerun logging: {obb.shape}.")


def _obb_line_strips(obbs: Tensor | ObbTW, *, t_world_snippet: PoseTW) -> list[list[list[float]]]:
    """Convert EFM snippet-frame OBB tensors into world-frame Rerun line strips."""

    world_obbs = _latest_valid_obb_slice(_transform_snippet_obbs_to_world(obbs, t_world_snippet))
    valid_mask = ~world_obbs.get_padding_mask().detach().cpu().numpy()
    if not np.any(valid_mask):
        return []
    corners = world_obbs.bb3corners_world.detach().cpu().numpy()[valid_mask]
    strips: list[list[list[float]]] = []
    for corners_world in corners.reshape(-1, 8, 3):
        if not np.isfinite(corners_world).all():
            continue
        for start, end in _OBB_EDGES:
            strips.append([corners_world[start].tolist(), corners_world[end].tolist()])
    return strips


def _obb_boxes(
    obbs: Tensor | ObbTW,
    *,
    t_world_snippet: PoseTW,
    sem_id_to_name: Mapping[int, str] | Sequence[str] | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], list[str], NDArray[Any], NDArray[Any]]:
    """Convert EFM snippet-frame OBB tensors into native Rerun box fields.

    Returns centers, half sizes, quaternions in Rerun ``xyzw`` order, and
    compact labels. The input OBBs are never mutated; the snippet-to-world
    transform is display-only for this recording.
    """

    world_obbs = _latest_valid_obb_slice(_transform_snippet_obbs_to_world(obbs, t_world_snippet))
    valid_mask_t = ~world_obbs.get_padding_mask().detach().cpu()
    if not bool(valid_mask_t.any()):
        empty = np.zeros((0, 3), dtype=np.float32)
        return (
            empty,
            empty,
            np.zeros((0, 4), dtype=np.float32),
            [],
            np.zeros((0,), dtype=np.int64),
            np.zeros(
                (0,),
                dtype=np.int64,
            ),
        )

    flat_obbs = world_obbs.reshape(-1, world_obbs.shape[-1])
    flat_valid = valid_mask_t.reshape(-1)
    valid_obbs = ObbTW(flat_obbs._data[flat_valid])
    centers = valid_obbs.bb3_center_world.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
    half_sizes = (0.5 * valid_obbs.bb3_diagonal).detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
    rotations = valid_obbs.T_world_object.R.detach().cpu()
    quat_wxyz = matrix_to_quaternion(rotations).numpy().reshape(-1, 4).astype(np.float32)
    quats_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    sem_id = valid_obbs.sem_id.detach().cpu().numpy().reshape(-1)
    inst_id = valid_obbs.inst_id.detach().cpu().numpy().reshape(-1)
    prob = valid_obbs.prob.detach().cpu().numpy().reshape(-1)
    labels = [
        (
            f"class={semantic_class_name(float(sem), sem_id_to_name)} | "
            f"sem_id={int(sem)} | inst_id={int(inst)} | prob={float(score):.3f}"
        )
        for sem, inst, score in zip(sem_id, inst_id, prob, strict=False)
    ]
    return centers, half_sizes, quats_xyzw, labels, sem_id.astype(np.int64), inst_id.astype(np.int64)


def _latest_valid_obb_slice(obbs: ObbTW) -> ObbTW:
    data = obbs.tensor().detach().cpu().to(dtype=torch.float32)
    if data.ndim <= 2:
        return ObbTW(data)
    rows = data.reshape(-1, data.shape[-2], data.shape[-1])
    for index in range(rows.shape[0] - 1, -1, -1):
        candidate = ObbTW(rows[index])
        if bool((~candidate.get_padding_mask()).any().item()):
            return candidate
    return ObbTW(rows[-1])


def _obb_family(palette: str) -> str:
    """Return the visual OBB color family for a palette name."""

    return "gt" if palette == "gt_obbs" else "detected"


def _target_obb_mask(
    *,
    labels: Sequence[str],
    sem_ids: NDArray[Any],
    inst_ids: NDArray[Any],
    target_hint: str | None,
) -> NDArray[np.bool_]:
    """Return a best-effort target OBB mask from a rollout/sample target hint."""

    mask = np.zeros((len(labels),), dtype=np.bool_)
    if target_hint is None:
        return mask
    normalized = str(target_hint).strip().lower()
    if not normalized:
        return mask
    tokens = set(re.findall(r"[a-z0-9_:-]+", normalized))
    hint_sem = _structured_target_value(normalized, key="sem")
    hint_inst = _structured_target_value(normalized, key="inst")
    for index, label in enumerate(labels):
        label_lower = label.lower()
        sem = int(sem_ids[index])
        inst = int(inst_ids[index])
        exact_tokens = {
            str(sem),
            str(inst),
            f"sem={sem}",
            f"sem_id={sem}",
            f"sem:{sem}",
            f"sem_id:{sem}",
            f"inst={inst}",
            f"inst_id={inst}",
            f"inst:{inst}",
            f"inst_id:{inst}",
        }
        structured_match = (
            (hint_sem is not None or hint_inst is not None)
            and (hint_sem is None or hint_sem == sem)
            and (hint_inst is None or hint_inst == inst)
        )
        token_match = not (hint_sem is not None or hint_inst is not None) and bool(tokens.intersection(exact_tokens))
        if normalized in label_lower or token_match or structured_match:
            mask[index] = True
    return mask


def _structured_target_value(target_hint: str, *, key: str) -> int | None:
    match = re.search(rf"(?:^|[:/_-]){key}(?:_id)?[=:](\d+)(?:$|[:/_-])", target_hint)
    return None if match is None else int(match.group(1))


def _gt_obb_semantic_names(sample: VinOfflineSample) -> Mapping[int, str] | Sequence[str] | None:
    """Return GT OBB semantic names when the offline payload exposes them."""

    gt_obbs = getattr(sample, "gt_obbs", None)
    return getattr(gt_obbs, "sem_id_to_name", None)


def _detected_obb_semantic_names(sample: VinOfflineSample) -> Mapping[int, str] | Sequence[str] | None:
    """Return detected OBB semantic names from compact or backbone payloads."""

    detected_obbs = getattr(sample, "detected_obbs", None)
    names = getattr(detected_obbs, "sem_id_to_name", None)
    if names is not None:
        return names
    backbone = getattr(sample, "backbone_out", None)
    if backbone is None:
        return None
    return getattr(backbone, "obb_pred_sem_id_to_name", None)


def _voxel_extent_bounds(voxel_extent: Tensor) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return voxel-frame min/max XYZ bounds from EVL extent metadata."""

    extent = _to_numpy(voxel_extent).reshape(-1, 6)[0].astype(np.float32)
    mins = np.asarray([extent[0], extent[2], extent[4]], dtype=np.float32)
    maxs = np.asarray([extent[1], extent[3], extent[5]], dtype=np.float32)
    return mins, maxs


def _voxel_extent_world_box_fields(
    voxel_extent: Tensor,
    *,
    t_world_voxel: PoseTW,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return world-space oriented Rerun box fields for the EVL voxel extent."""

    mins, maxs = _voxel_extent_bounds(voxel_extent)
    center_voxel = 0.5 * (mins + maxs)
    half_size = 0.5 * (maxs - mins)
    r, t = _pose_rt(t_world_voxel, [0])
    center_world = (r[0] @ center_voxel.reshape(3, 1)).reshape(1, 3) + t[0].reshape(1, 3)
    quat_wxyz = matrix_to_quaternion(torch.as_tensor(r, dtype=torch.float32)).numpy().reshape(-1, 4)
    quats_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    return center_world.astype(np.float32), half_size.reshape(1, 3).astype(np.float32), quats_xyzw.astype(np.float32)


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


def _display_rot90_cw(array: NDArray[Any]) -> NDArray[Any]:
    """Apply ARIA's display-only 90 degree clockwise image convention."""

    return np.ascontiguousarray(np.rot90(array, k=-1))


def _keyframe_indices(num_frames: int) -> list[int]:
    """Return the first/last keyframe indices used for compact inspection."""

    if num_frames <= 0:
        return []
    return [0] if num_frames <= 1 else [0, num_frames - 1]


def _score_rgba(values: NDArray[Any], *, low: Sequence[int], high: Sequence[int]) -> list[list[int]]:
    """Map scalar scores to RGBA colors with a simple linear ramp."""

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return []
    finite = arr[np.isfinite(arr)]
    lo = float(np.min(finite)) if finite.size else 0.0
    hi = float(np.max(finite)) if finite.size else 1.0
    denom = hi - lo if hi > lo else 1.0
    alpha = np.clip((arr - lo) / denom, 0.0, 1.0)
    low_arr = np.asarray(low, dtype=np.float32)
    high_arr = np.asarray(high, dtype=np.float32)
    colors = low_arr[None, :] * (1.0 - alpha[:, None]) + high_arr[None, :] * alpha[:, None]
    return np.clip(colors, 0, 255).astype(np.uint8).tolist()


def _field_voxel_centers_world(
    field: Tensor,
    *,
    t_world_voxel: PoseTW,
    voxel_extent: Tensor,
    threshold: float,
    max_points: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return thresholded EFM voxel centers in world coordinates and their scores."""

    if field.ndim != 5 or max_points <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    values = field.detach().cpu().float()[0, 0]
    mask = values >= float(threshold)
    if not bool(mask.any()):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    indices = torch.nonzero(mask, as_tuple=False)
    scores = values[mask]
    if scores.numel() > max_points:
        top = torch.topk(scores, k=max_points, largest=True, sorted=False).indices
        indices = indices[top]
        scores = scores[top]

    d, h, w = values.shape
    extent = voxel_extent.detach().cpu().float().reshape(-1, 6)[0]
    x_min, x_max, y_min, y_max, z_min, z_max = extent.tolist()
    z_idx = indices[:, 0].to(dtype=torch.float32)
    y_idx = indices[:, 1].to(dtype=torch.float32)
    x_idx = indices[:, 2].to(dtype=torch.float32)
    x = float(x_min) + (x_idx + 0.5) * ((float(x_max) - float(x_min)) / float(w))
    y = float(y_min) + (y_idx + 0.5) * ((float(y_max) - float(y_min)) / float(h))
    z = float(z_min) + (z_idx + 0.5) * ((float(z_max) - float(z_min)) / float(d))
    centers_voxel = torch.stack([x, y, z], dim=-1)

    pose = t_world_voxel
    if pose.ndim != 1:
        pose = PoseTW(pose._data.reshape(-1, 12)[:1])
    centers_world = (pose * centers_voxel.to(device=pose._data.device)).detach().cpu().numpy().reshape(-1, 3)
    return centers_world.astype(np.float32), scores.detach().cpu().numpy().astype(np.float32)


class RerunOfflineLogger:
    """Stateful logger that writes one selected offline sample to Rerun."""

    def __init__(
        self,
        config: RerunOfflineInspectorConfig,
        *,
        rr_module: RerunModule | None = None,
        target_obb_hint: str | None = None,
    ) -> None:
        """Create the logger.

        Args:
            config: Inspector configuration.
            rr_module: Optional fake or imported ``rerun`` module.
            target_obb_hint: Optional rollout/sample target id used to highlight
                a matching OBB in diagnostics when labels expose the id.
        """

        self.config = config
        if rr_module is None:
            import rerun as imported_rr

            self.rr = cast("RerunModule", imported_rr)
        else:
            self.rr = rr_module
        self._metadata_warnings: list[str] = []
        self._target_obb_hint = target_obb_hint

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

    def _log_camera_pose_and_pinhole(
        self,
        entity_path: str,
        pose_world_camera: PoseTW,
        pinhole_kwargs: dict[str, list[float]],
    ) -> None:
        """Log one native Rerun camera entity with pose and ARIA LUF intrinsics."""

        r, t = _pose_rt(pose_world_camera, [0])
        self.rr.log(
            entity_path,
            self.rr.Transform3D(
                translation=t[0].tolist(),
                mat3x3=r[0].tolist(),
                relation=self.rr.TransformRelation.ParentFromChild,
            ),
            self.rr.Pinhole(
                **pinhole_kwargs,
                camera_xyz=self.rr.ViewCoordinates.LUF,
                image_plane_distance=self.config.geometry.frustum_scale,
            ),
            static=True,
        )

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
        log_default_inspector_blueprint(self.rr)
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
        if self.config.primitives.log_candidate_frusta or self.config.primitives.log_candidate_centers:
            self._log_candidates(sample, inventory=inventory)
        if self.config.primitives.log_candidate_points and inventory.has_candidate_points:
            self._log_candidate_points(sample)
        if self.config.primitives.log_candidate_depths and inventory.has_candidate_depths:
            self._log_candidate_depths(sample)
        if (self.config.primitives.log_mesh or self.config.primitives.log_gt_mesh) and inventory.has_mesh:
            self._log_mesh(sample)
        if self.config.primitives.log_gt_obbs and inventory.has_gt_obbs:
            self._log_obbs(
                sample,
                entity_path=ENTITY_GT_OBBS,
                obbs=_compact_or_live_gt_obbs(sample),
                palette="gt_obbs",
                sem_id_to_name=_gt_obb_semantic_names(sample),
                show_scene_labels=self.config.primitives.show_gt_obb_labels,
            )
        if self.config.primitives.log_detected_obbs and inventory.has_detected_obbs:
            self._log_obbs(
                sample,
                entity_path=ENTITY_DETECTED_OBBS,
                obbs=_detected_obbs(sample),
                palette="detected_obbs",
                sem_id_to_name=_detected_obb_semantic_names(sample),
                show_scene_labels=self.config.primitives.show_detected_obb_labels,
            )
        if self.config.primitives.log_efm_voxels and self.config.efm_voxels.enabled:
            self._log_efm_voxels(sample)
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
                colors=_semidense_quality_colors(sample, points.shape[0]),
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
        count = _candidate_count(sample)
        validity = _validity_mask(sample, count)
        selected_idx = _selected_candidate_index(sample, candidate_count=count, validity=validity, config=self.config)
        indices = _candidate_subset_indices(sample, candidate_count=count, validity=validity, config=self.config)

        if self.config.primitives.log_candidate_frusta:
            for idx in indices:
                camera_entity = _candidate_camera_entity(idx, validity)
                self._log_camera_pose_and_pinhole(
                    camera_entity,
                    _subset_poses(poses, [idx]),
                    _p3d_pinhole_kwargs(cameras, idx),
                )

        if self.config.primitives.log_candidate_centers and selected_idx is not None:
            center = _worker_c_candidate_centers(poses, [selected_idx])
            self.rr.log(
                f"{_candidate_entity(selected_idx, validity)}/center",
                self.rr.Points3D(
                    center,
                    radii=self.config.geometry.candidate_center_radius,
                    colors=_rgba("centers", center.shape[0]),
                ),
                static=True,
            )

    def _log_candidate_points(self, sample: VinOfflineSample) -> None:
        count = _candidate_count(sample)
        validity = _validity_mask(sample, count)
        selected_idx = _selected_candidate_index(
            sample,
            candidate_count=count,
            validity=validity,
            config=self.config,
        )
        if selected_idx is None:
            return
        points = _candidate_points_for_index(
            sample,
            selected_idx,
            max_points=self.config.performance.max_candidate_points,
            seed=self.config.performance.seed,
        )
        if points is None:
            return
        self.rr.log(
            f"{_candidate_entity(selected_idx, validity)}/points",
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
        mesh_color = _rgba("mesh", 1)[0]
        mesh_color[3] = self.config.geometry.mesh_alpha
        self.rr.log(
            ENTITY_MESH,
            self.rr.Mesh3D(
                vertex_positions=np.asarray(verts, dtype=np.float32),
                triangle_indices=np.asarray(faces, dtype=np.uint32),
                albedo_factor=mesh_color,
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
        sem_id_to_name: Mapping[int, str] | Sequence[str] | None,
        show_scene_labels: bool,
    ) -> None:
        if obbs is None:
            return
        t_world_snippet = _snippet_t_world_snippet(sample)
        if t_world_snippet is None:
            self._warn_metadata(f"{entity_path} skipped: missing T_world_snippet for snippet-frame OBBs.")
            return
        centers, half_sizes, quaternions, labels, sem_ids, inst_ids = _obb_boxes(
            obbs,
            t_world_snippet=t_world_snippet,
            sem_id_to_name=sem_id_to_name,
        )
        if centers.shape[0] == 0:
            return
        target_mask = _target_obb_mask(
            labels=labels,
            sem_ids=sem_ids,
            inst_ids=inst_ids,
            target_hint=self._target_obb_hint,
        )
        box_kwargs: dict[str, object] = {
            "centers": centers,
            "half_sizes": half_sizes,
            "quaternions": quaternions,
            "colors": obb_semantic_rgba(
                sem_ids,
                family=cast("Any", _obb_family(palette)),
                target_mask=target_mask,
                alpha=235 if palette == "gt_obbs" else 220,
            ).tolist(),
        }
        if show_scene_labels:
            box_kwargs["labels"] = labels
        self.rr.log(
            entity_path,
            self.rr.Boxes3D(**box_kwargs),
            self.rr.AnyValues(
                obb_label=labels,
                obb_sem_id=sem_ids.astype(int).tolist(),
                obb_inst_id=inst_ids.astype(int).tolist(),
                obb_is_target=target_mask.astype(bool).tolist(),
                target_obb_hint=self._target_obb_hint or "",
            ),
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
            image_entity = f"{ENTITY_ASE_KEYFRAME_MEDIA}/frame_{idx:03d}/image"
            self._log_camera_pose_and_pinhole(camera_entity, pose_world_cam, _camera_tw_pinhole_kwargs(camera_tw))
            self.rr.log(
                image_entity,
                self.rr.Image(_display_rot90_cw(_image_hwc(camera.images, idx))),
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
            depth_entity = f"{ENTITY_ASE_KEYFRAME_MEDIA}/frame_{idx:03d}/depth"
            self._log_camera_pose_and_pinhole(camera_entity, pose_world_cam, _camera_tw_pinhole_kwargs(camera_tw))
            self.rr.log(
                depth_entity,
                self.rr.DepthImage(_display_rot90_cw(_depth_hw(depth, idx)), meter=1.0),
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
        validity = _validity_mask(sample, candidate_count)
        selected_idx = _selected_candidate_index(
            sample,
            candidate_count=candidate_count,
            validity=validity,
            config=self.config,
        )
        if selected_idx is None:
            return
        image = arr[selected_idx]
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        camera_entity = _candidate_camera_entity(selected_idx, validity)
        depth_entity = f"{_candidate_entity(selected_idx, validity)}/depth"
        self._log_camera_pose_and_pinhole(
            camera_entity,
            _subset_poses(sample.oracle.candidate_poses_world_cam, [selected_idx]),
            _p3d_pinhole_kwargs(sample.oracle.p3d_cameras, selected_idx),
        )
        self.rr.log(
            depth_entity,
            self.rr.DepthImage(np.asarray(image, dtype=np.float32), meter=1.0),
            static=True,
        )

    def _log_efm_voxels(self, sample: VinOfflineSample) -> None:
        """Log curated EVL/EFM voxel evidence as thresholded world-space points."""

        backbone = getattr(sample, "backbone_out", None)
        if backbone is None:
            return
        cfg = self.config.efm_voxels
        self._log_efm_voxel_extent(backbone)
        field_specs = (
            ("occ_pr", cfg.log_occ_pr, cfg.occ_threshold, [30, 70, 150, 80], [90, 180, 255, 210]),
            ("cent_pr", cfg.log_cent_pr, cfg.cent_threshold, [80, 40, 130, 80], [245, 120, 255, 230]),
            ("cent_pr_nms", cfg.log_cent_pr_nms, cfg.cent_nms_threshold, [80, 130, 80, 80], [130, 255, 150, 240]),
        )
        for name, enabled, threshold, low, high in field_specs:
            if not enabled:
                continue
            field = getattr(backbone, name, None)
            if field is None:
                self._warn_metadata(f"{ENTITY_EFM_VOXELS}/{name} skipped: backbone field is missing.")
                continue
            points, scores = _field_voxel_centers_world(
                field,
                t_world_voxel=backbone.t_world_voxel,
                voxel_extent=backbone.voxel_extent,
                threshold=threshold,
                max_points=cfg.max_points_per_field,
            )
            if points.shape[0] == 0:
                self._warn_metadata(f"{ENTITY_EFM_VOXELS}/{name} skipped: no voxels passed threshold {threshold}.")
                continue
            self.rr.log(
                f"{ENTITY_EFM_VOXELS}/{name}",
                self.rr.Points3D(
                    points,
                    radii=cfg.point_radius,
                    colors=_score_rgba(scores, low=low, high=high),
                ),
                static=True,
            )

    def _log_efm_voxel_extent(self, backbone: object) -> None:
        """Log the EFM voxel extent as a native box posed by ``T_world_voxel``."""

        t_world_voxel = getattr(backbone, "t_world_voxel", None)
        voxel_extent = getattr(backbone, "voxel_extent", None)
        if t_world_voxel is None or voxel_extent is None:
            self._warn_metadata(f"{ENTITY_EFM_VOXEL_EXTENT} skipped: missing voxel pose or extent.")
            return
        centers, half_sizes, quaternions = _voxel_extent_world_box_fields(
            voxel_extent,
            t_world_voxel=t_world_voxel,
        )
        self.rr.log(
            ENTITY_EFM_VOXEL_EXTENT,
            self.rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                quaternions=quaternions,
                colors=[[255, 255, 255, 80]],
                labels=["EFM voxel extent"],
            ),
            static=True,
        )


__all__ = [
    "ENTITY_ASE_KEYFRAME_MEDIA",
    "ENTITY_CANDIDATE_ROOT",
    "ENTITY_DETECTED_OBBS",
    "ENTITY_DEPTH_KEYFRAMES",
    "ENTITY_EFM_VOXELS",
    "ENTITY_EFM_VOXEL_EXTENT",
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
    "log_default_inspector_blueprint",
]
