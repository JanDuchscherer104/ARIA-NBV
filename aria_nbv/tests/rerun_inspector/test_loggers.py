"""Rerun logger tests using a fake rerun module."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from efm3d.aria.aria_constants import ARIA_SNIPPET_T_WORLD_SNIPPET
from efm3d.aria.pose import PoseTW
from pytorch3d.renderer.cameras import PerspectiveCameras

from aria_nbv.rerun_inspector._config import RerunOfflineInspectorConfig
from aria_nbv.rerun_inspector._loggers import (
    ENTITY_CANDIDATE_CENTERS,
    ENTITY_CANDIDATE_DEPTHS,
    ENTITY_CANDIDATE_POINTS,
    ENTITY_DEPTH_KEYFRAMES,
    ENTITY_DETECTED_OBBS,
    ENTITY_FRUSTA_ALL,
    ENTITY_FRUSTA_INVALID,
    ENTITY_FRUSTA_TOP_ORACLE,
    ENTITY_GT_OBBS,
    ENTITY_METADATA_SAMPLE,
    ENTITY_REFERENCE_POSE,
    ENTITY_RGB_KEYFRAMES,
    ENTITY_SEMIDENSE,
    ENTITY_TRAJECTORY,
    ENTITY_WORLD,
    RerunOfflineLogger,
)
from aria_nbv.rerun_inspector._metadata import OfflineVisualInventory, normalize_visual_inventory


class _Archetype:
    """Simple fake Rerun archetype that stores constructor data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeRerun:
    """Fake subset of the Rerun module used by the inspector."""

    Points3D = _Archetype
    LineStrips3D = _Archetype
    TextDocument = _Archetype
    Transform3D = _Archetype
    Mesh3D = _Archetype
    Image = _Archetype
    DepthImage = _Archetype
    Pinhole = _Archetype

    class ViewCoordinates:
        """Fake Rerun view-coordinate constants."""

        RIGHT_HAND_Z_UP = _Archetype("RIGHT_HAND_Z_UP")
        LUF = "LUF"

    class TransformRelation:
        """Fake Rerun transform-relation constants."""

        ParentFromChild = "ParentFromChild"

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.logged: dict[str, _Archetype] = {}
        self.logged_extras: dict[str, tuple[Any, ...]] = {}
        self.logged_kwargs: dict[str, dict[str, Any]] = {}

    def init(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("init", args, kwargs))

    def save(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("save", args, kwargs))

    def spawn(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("spawn", args, kwargs))

    def connect_grpc(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("connect_grpc", args, kwargs))

    def log(self, entity_path: str, entity: _Archetype, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("log", entity_path, args, kwargs))
        self.logged[entity_path] = entity
        self.logged_extras[entity_path] = args
        self.logged_kwargs[entity_path] = kwargs


def _poses(translations: list[list[float]]) -> PoseTW:
    """Build a PoseTW batch from identity rotations and translations."""

    t = torch.tensor(translations, dtype=torch.float32)
    r = torch.eye(3, dtype=torch.float32).expand(t.shape[0], 3, 3)
    return PoseTW.from_Rt(r, t)


def _sample(
    num_points: int = 24,
    *,
    candidate_count: int = 3,
    validity: list[bool] | None = None,
) -> SimpleNamespace:
    """Build a minimal VinOfflineSample-like object."""

    semidense = torch.arange(num_points * 3, dtype=torch.float32).reshape(num_points, 3)
    candidate_points = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    mask_valid = torch.tensor([True, False, True] if validity is None else validity)
    return SimpleNamespace(
        sample_key="sample-0",
        scene_id="scene",
        snippet_id="snippet",
        vin_snippet=SimpleNamespace(
            points_world=semidense,
            lengths=torch.tensor([num_points]),
            t_world_rig=_poses([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ),
        oracle=SimpleNamespace(
            candidate_poses_world_cam=_poses([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            reference_pose_world_rig=_poses([[0.25, 0.5, 0.75]]),
            candidate_count=candidate_count,
            rri=torch.tensor([0.1, 0.9, 0.3], dtype=torch.float32),
            p3d_cameras=PerspectiveCameras(
                R=torch.eye(3, dtype=torch.float32).expand(3, 3, 3).clone(),
                T=torch.zeros(3, 3, dtype=torch.float32),
                focal_length=torch.full((3, 2), 2.0, dtype=torch.float32),
                principal_point=torch.full((3, 2), 2.0, dtype=torch.float32),
                image_size=torch.full((3, 2), 4.0, dtype=torch.float32),
                in_ndc=False,
            ),
        ),
        candidates=SimpleNamespace(mask_valid=mask_valid),
        candidate_pcs=SimpleNamespace(points=candidate_points, lengths=torch.tensor([3, 4])),
        depths=SimpleNamespace(depths=torch.ones((3, 4, 4), dtype=torch.float32)),
        efm_snippet_view=None,
    )


def _obb_tensor(offset: float = 0.0) -> torch.Tensor:
    """Build one EFM-layout OBB tensor for line-strip logging."""

    data = torch.full((1, 34), -1.0, dtype=torch.float32)
    data[0, :6] = torch.tensor([-0.5, 0.5, -0.25, 0.25, -0.1, 0.9], dtype=torch.float32)
    data[0, 18:27] = torch.eye(3, dtype=torch.float32).reshape(-1)
    data[0, 27:30] = torch.tensor([offset, 0.0, 0.0], dtype=torch.float32)
    return data


def test_logger_initializes_save_sink_before_logging(tmp_path) -> None:
    """Rerun 0.22.1 sink setup should happen before any entity log calls."""

    cfg = RerunOfflineInspectorConfig()
    cfg.output.save_path = tmp_path / "out.rrd"
    fake = _FakeRerun()
    logger = RerunOfflineLogger(cfg, rr_module=fake)

    logger.start()
    logger.log_sample(
        sample=_sample(),
        inventory=OfflineVisualInventory(has_candidate_validity=True, has_candidate_points=True),
        selection="split=val index=0",
    )
    logger.log_metadata(
        sample=_sample(),
        inventory=OfflineVisualInventory(has_candidate_validity=True, has_candidate_points=True),
        selection="split=val index=0",
    )

    assert [call[0] for call in fake.calls[:2]] == ["init", "save"]  # noqa: S101
    assert fake.calls[2][0:2] == ("log", ENTITY_WORLD)  # noqa: S101
    first_log = next(idx for idx, call in enumerate(fake.calls) if call[0] == "log")
    assert first_log > 1  # noqa: S101
    assert ENTITY_WORLD in fake.logged  # noqa: S101
    assert ENTITY_SEMIDENSE in fake.logged  # noqa: S101
    assert ENTITY_REFERENCE_POSE in fake.logged  # noqa: S101
    assert fake.logged[ENTITY_REFERENCE_POSE].kwargs["relation"] == "ParentFromChild"  # noqa: S101
    assert ENTITY_FRUSTA_ALL in fake.logged  # noqa: S101
    assert ENTITY_FRUSTA_TOP_ORACLE in fake.logged  # noqa: S101
    assert ENTITY_FRUSTA_INVALID in fake.logged  # noqa: S101
    assert ENTITY_CANDIDATE_CENTERS in fake.logged  # noqa: S101
    assert ENTITY_METADATA_SAMPLE in fake.logged  # noqa: S101


def test_defaults_keep_candidate_point_clouds_opt_in() -> None:
    """Candidate point-cloud loading/logging should be disabled by default."""

    cfg = RerunOfflineInspectorConfig()

    assert not cfg.dataset.offline.load_candidate_pcs  # noqa: S101
    assert not cfg.primitives.log_candidate_points  # noqa: S101
    assert cfg.performance.max_semidense_points == 50_000  # noqa: S101
    assert cfg.performance.max_candidate_points == 20_000  # noqa: S101

    fake = _FakeRerun()
    RerunOfflineLogger(cfg, rr_module=fake).log_sample(
        sample=_sample(),
        inventory=OfflineVisualInventory(has_candidate_validity=True, has_candidate_points=True),
        selection="sample_key=sample-0",
    )
    assert ENTITY_CANDIDATE_POINTS not in fake.logged  # noqa: S101


def test_normalized_inventory_preserves_worker_b_diagnostics() -> None:
    """Worker-B warnings/errors/metadata should survive into Rerun metadata."""

    cfg = RerunOfflineInspectorConfig()
    worker_b_inventory = SimpleNamespace(
        has_semidense_points=True,
        has_candidate_mask=True,
        has_candidate_pcs=True,
        warnings=("candidate point clouds missing",),
        errors=("sample.oracle.rri missing",),
        metadata={"vin_snippet.valid_semidense_points": 42},
    )
    inventory = normalize_visual_inventory(worker_b_inventory)
    fake = _FakeRerun()

    RerunOfflineLogger(cfg, rr_module=fake).log_metadata(
        sample=_sample(),
        inventory=inventory,
        selection="sample_key=sample-0",
    )

    document = json.loads(fake.logged[ENTITY_METADATA_SAMPLE].args[0])
    details = document["inventory"]["details"]
    assert details["warnings"] == ["candidate point clouds missing"]  # noqa: S101
    assert details["errors"] == ["sample.oracle.rri missing"]  # noqa: S101
    assert details["metadata"]["vin_snippet.valid_semidense_points"] == 42  # noqa: S101


def test_worker_b_candidate_pcs_semidense_flag_does_not_disable_required_semidense() -> None:
    """Worker-B candidate-pc semidense absence should not reject VIN semidense."""

    inventory = normalize_visual_inventory(
        {
            "has_semidense_points": False,
            "has_candidate_pcs": False,
            "metadata": {"vin_snippet.valid_semidense_points": 9},
        },
    )

    assert inventory.has_semidense  # noqa: S101
    assert not inventory.has_candidate_points  # noqa: S101
    assert inventory.details is not None  # noqa: S101
    assert inventory.details["candidate_pcs_has_semidense_points"] is False  # noqa: S101
    assert inventory.details["metadata"]["vin_snippet.valid_semidense_points"] == 9  # noqa: S101


def test_semidense_downsampling_is_deterministic(tmp_path) -> None:
    """Configured seed should make semidense downsampling repeatable."""

    cfg = RerunOfflineInspectorConfig()
    cfg.output.save_path = tmp_path / "out.rrd"
    cfg.performance.max_semidense_points = 8
    cfg.performance.seed = 123
    cfg.primitives.log_reference_pose = False
    cfg.primitives.log_candidate_frusta = False
    cfg.primitives.log_top_oracle_frustum = False
    cfg.primitives.log_invalid_frusta = False
    cfg.primitives.log_candidate_centers = False
    sample = _sample(num_points=80)

    fake_a = _FakeRerun()
    fake_b = _FakeRerun()
    RerunOfflineLogger(cfg, rr_module=fake_a).log_sample(
        sample=sample,
        inventory=OfflineVisualInventory(),
        selection="sample_key=sample-0",
    )
    RerunOfflineLogger(cfg, rr_module=fake_b).log_sample(
        sample=sample,
        inventory=OfflineVisualInventory(),
        selection="sample_key=sample-0",
    )

    points_a = np.asarray(fake_a.logged[ENTITY_SEMIDENSE].args[0])
    points_b = np.asarray(fake_b.logged[ENTITY_SEMIDENSE].args[0])
    assert points_a.shape == (8, 3)  # noqa: S101
    np.testing.assert_array_equal(points_a, points_b)


def test_semidense_logging_uses_valid_xyz_prefix() -> None:
    """VIN semidense logging should handle padded point features with extra channels."""

    cfg = RerunOfflineInspectorConfig()
    cfg.primitives.log_reference_pose = False
    cfg.primitives.log_candidate_frusta = False
    cfg.primitives.log_top_oracle_frustum = False
    cfg.primitives.log_invalid_frusta = False
    cfg.primitives.log_candidate_centers = False
    points = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    sample = _sample()
    sample.vin_snippet = SimpleNamespace(points_world=points, lengths=torch.tensor([3]))
    fake = _FakeRerun()

    RerunOfflineLogger(cfg, rr_module=fake).log_sample(
        sample=sample,
        inventory=OfflineVisualInventory(),
        selection="sample_key=sample-0",
    )

    logged = np.asarray(fake.logged[ENTITY_SEMIDENSE].args[0])
    assert logged.shape == (3, 3)  # noqa: S101
    np.testing.assert_array_equal(logged, points[:3, :3].numpy())


def test_candidate_points_are_logged_only_when_inventory_reports_present() -> None:
    """Candidate point clouds are optional and gated by inventory."""

    cfg = RerunOfflineInspectorConfig()
    cfg.primitives.log_candidate_points = True
    cfg.primitives.log_semidense = False
    cfg.primitives.log_reference_pose = False
    cfg.primitives.log_candidate_frusta = False
    cfg.primitives.log_top_oracle_frustum = False
    cfg.primitives.log_invalid_frusta = False
    cfg.primitives.log_candidate_centers = False
    sample = _sample()

    missing = _FakeRerun()
    RerunOfflineLogger(cfg, rr_module=missing).log_sample(
        sample=sample,
        inventory=OfflineVisualInventory(has_candidate_points=False),
        selection="sample_key=sample-0",
    )
    assert ENTITY_CANDIDATE_POINTS not in missing.logged  # noqa: S101

    present = _FakeRerun()
    RerunOfflineLogger(cfg, rr_module=present).log_sample(
        sample=sample,
        inventory=OfflineVisualInventory(has_candidate_points=True),
        selection="sample_key=sample-0",
    )
    assert ENTITY_CANDIDATE_POINTS in present.logged  # noqa: S101


def test_zero_candidate_samples_log_no_candidate_layers() -> None:
    """A no-candidate sample should not visualize padded candidate poses."""

    cfg = RerunOfflineInspectorConfig()
    cfg.primitives.log_semidense = False
    cfg.primitives.log_reference_pose = False
    sample = _sample(candidate_count=0, validity=[False, False, False])
    fake = _FakeRerun()

    RerunOfflineLogger(cfg, rr_module=fake).log_sample(
        sample=sample,
        inventory=OfflineVisualInventory(has_candidate_validity=True),
        selection="sample_key=sample-0",
    )

    assert ENTITY_FRUSTA_ALL not in fake.logged  # noqa: S101
    assert ENTITY_FRUSTA_TOP_ORACLE not in fake.logged  # noqa: S101
    assert ENTITY_FRUSTA_INVALID not in fake.logged  # noqa: S101
    assert ENTITY_CANDIDATE_CENTERS not in fake.logged  # noqa: S101


def test_all_invalid_candidates_do_not_log_top_oracle_frustum() -> None:
    """Top-oracle visualization should be omitted when every candidate is invalid."""

    cfg = RerunOfflineInspectorConfig()
    cfg.primitives.log_semidense = False
    cfg.primitives.log_reference_pose = False
    sample = _sample(validity=[False, False, False])
    fake = _FakeRerun()

    RerunOfflineLogger(cfg, rr_module=fake).log_sample(
        sample=sample,
        inventory=OfflineVisualInventory(has_candidate_validity=True),
        selection="sample_key=sample-0",
    )

    assert ENTITY_FRUSTA_ALL in fake.logged  # noqa: S101
    assert ENTITY_FRUSTA_INVALID in fake.logged  # noqa: S101
    assert ENTITY_FRUSTA_TOP_ORACLE not in fake.logged  # noqa: S101


def test_compact_modalities_log_to_stable_entity_paths() -> None:
    """Compact OBBs, trajectory, and candidate depths should have fixed Rerun paths."""

    cfg = RerunOfflineInspectorConfig()
    cfg.primitives.log_semidense = False
    cfg.primitives.log_reference_pose = False
    cfg.primitives.log_candidate_frusta = False
    cfg.primitives.log_top_oracle_frustum = False
    cfg.primitives.log_invalid_frusta = False
    cfg.primitives.log_candidate_centers = False
    cfg.primitives.log_candidate_depths = True
    sample = _sample()
    sample.gt_obbs = SimpleNamespace(obbs=_obb_tensor(0.0))
    sample.detected_obbs = SimpleNamespace(obbs=_obb_tensor(1.0))
    sample.trajectory = SimpleNamespace(time_ns=torch.tensor([100, 200, 300], dtype=torch.int64))
    fake = _FakeRerun()

    RerunOfflineLogger(cfg, rr_module=fake).log_sample(
        sample=sample,
        inventory=OfflineVisualInventory(
            has_gt_obbs=True,
            has_detected_obbs=True,
            has_trajectory=True,
            has_candidate_depths=True,
        ),
        selection="sample_key=sample-0",
    )

    assert ENTITY_GT_OBBS in fake.logged  # noqa: S101
    assert ENTITY_DETECTED_OBBS in fake.logged  # noqa: S101
    assert ENTITY_TRAJECTORY in fake.logged  # noqa: S101
    camera_path = f"{ENTITY_CANDIDATE_DEPTHS}/candidate_000/camera"
    depth_path = f"{camera_path}/depth"
    invalid_raw_top_path = f"{ENTITY_CANDIDATE_DEPTHS}/candidate_001/camera"
    valid_top_path = f"{ENTITY_CANDIDATE_DEPTHS}/candidate_002/camera"
    assert camera_path in fake.logged  # noqa: S101
    assert depth_path in fake.logged  # noqa: S101
    assert invalid_raw_top_path not in fake.logged  # noqa: S101
    assert valid_top_path in fake.logged  # noqa: S101
    assert fake.logged[camera_path].kwargs["relation"] == "ParentFromChild"  # noqa: S101
    assert isinstance(fake.logged[depth_path], _Archetype)  # noqa: S101
    assert fake.logged[depth_path].kwargs["camera_xyz"] == "LUF"  # noqa: S101
    assert fake.logged_extras[depth_path][0].kwargs["meter"] == 1.0  # noqa: S101
    assert len(fake.logged[ENTITY_GT_OBBS].args[0]) == 12  # noqa: S101


def test_obbs_are_transformed_from_snippet_to_world_before_logging() -> None:
    """Snippet-frame OBB payloads should be drawn under world after applying T_world_snippet."""

    cfg = RerunOfflineInspectorConfig()
    cfg.primitives.log_semidense = False
    cfg.primitives.log_reference_pose = False
    cfg.primitives.log_candidate_frusta = False
    cfg.primitives.log_top_oracle_frustum = False
    cfg.primitives.log_invalid_frusta = False
    cfg.primitives.log_candidate_centers = False
    sample = _sample()
    sample.gt_obbs = SimpleNamespace(obbs=_obb_tensor(0.0))
    sample.efm_snippet_view = SimpleNamespace(efm={ARIA_SNIPPET_T_WORLD_SNIPPET: _poses([[10.0, 0.0, 0.0]])})
    fake = _FakeRerun()

    RerunOfflineLogger(cfg, rr_module=fake).log_sample(
        sample=sample,
        inventory=OfflineVisualInventory(has_gt_obbs=True),
        selection="sample_key=sample-0",
    )

    strips = np.asarray(fake.logged[ENTITY_GT_OBBS].args[0], dtype=np.float32)
    assert np.allclose(strips[0, 0], np.asarray([9.5, -0.25, -0.1], dtype=np.float32))  # noqa: S101
    assert np.allclose(strips[0, 1], np.asarray([10.5, -0.25, -0.1], dtype=np.float32))  # noqa: S101


def test_missing_keyframe_context_is_reported_in_metadata() -> None:
    """Requested live keyframes should not become orphan images without camera context."""

    cfg = RerunOfflineInspectorConfig()
    cfg.primitives.log_semidense = False
    cfg.primitives.log_reference_pose = False
    cfg.primitives.log_candidate_frusta = False
    cfg.primitives.log_top_oracle_frustum = False
    cfg.primitives.log_invalid_frusta = False
    cfg.primitives.log_candidate_centers = False
    cfg.primitives.log_rgb_keyframes = True
    cfg.primitives.log_depth_keyframes = True
    sample = _sample()
    fake = _FakeRerun()
    logger = RerunOfflineLogger(cfg, rr_module=fake)

    logger.log_sample(
        sample=sample,
        inventory=OfflineVisualInventory(has_rgb_keyframes=True, has_depth_keyframes=True),
        selection="sample_key=sample-0",
    )
    logger.log_metadata(
        sample=sample,
        inventory=OfflineVisualInventory(has_rgb_keyframes=True, has_depth_keyframes=True),
        selection="sample_key=sample-0",
    )

    document = json.loads(fake.logged[ENTITY_METADATA_SAMPLE].args[0])
    assert not any(path.startswith(ENTITY_RGB_KEYFRAMES) for path in fake.logged)  # noqa: S101
    assert not any(path.startswith(ENTITY_DEPTH_KEYFRAMES) for path in fake.logged)  # noqa: S101
    assert "RGB keyframes skipped: sample has no attached EFM snippet." in document["runtime_warnings"]  # noqa: S101
    assert "Depth keyframes skipped: sample has no attached EFM snippet." in document["runtime_warnings"]  # noqa: S101
