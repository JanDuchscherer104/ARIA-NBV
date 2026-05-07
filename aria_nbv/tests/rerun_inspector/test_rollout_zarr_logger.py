"""Rerun rollout-Zarr logger tests using a fake Rerun module."""

# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("efm3d")

from efm3d.aria.pose import PoseTW

from aria_nbv.data_handling import RolloutZarrStoreReader, write_rollout_zarr_store
from aria_nbv.pose_generation import INVALID_REASON_CODES, build_synthetic_rollout_traces
from aria_nbv.rerun_inspector._config import RerunOfflineInspectorConfig
from aria_nbv.rerun_inspector._loggers import ENTITY_WORLD
from aria_nbv.rerun_inspector._rollout_zarr import (
    ENTITY_ROLLOUT_INVALID_ROOT,
    ENTITY_ROLLOUT_METADATA,
    ENTITY_ROLLOUT_SELECTED_PATH,
    ENTITY_ROLLOUT_SELECTED_ROOT,
    ENTITY_ROLLOUT_STEP_METADATA,
    ENTITY_ROLLOUT_VALID_ROOT,
    RerunRolloutZarrLogger,
)


class _Archetype:
    """Simple fake Rerun archetype that stores constructor data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeRerun:
    """Fake subset of the Rerun module used by the rollout inspector."""

    Points3D = _Archetype
    LineStrips3D = _Archetype
    TextDocument = _Archetype
    Scalar = _Archetype
    Transform3D = _Archetype
    Pinhole = _Archetype
    AnyValues = _Archetype

    class ViewCoordinates:
        """Fake Rerun view-coordinate constants."""

        RIGHT_HAND_Z_UP = _Archetype("RIGHT_HAND_Z_UP")
        LUF = _Archetype("LUF")

    class TransformRelation:
        """Fake transform relation constants."""

        ParentFromChild = "ParentFromChild"

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.logged: dict[str, _Archetype] = {}
        self.logged_extras: dict[str, tuple[Any, ...]] = {}

    def init(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("init", args, kwargs))

    def save(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("save", args, kwargs))

    def spawn(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("spawn", args, kwargs))

    def connect_grpc(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("connect_grpc", args, kwargs))

    def set_time_sequence(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("set_time_sequence", args, kwargs))

    def log(self, entity_path: str, entity: _Archetype, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("log", entity_path, args, kwargs))
        self.logged[entity_path] = entity
        self.logged_extras[entity_path] = args


def _synthetic_rollout_store(tmp_path: Path, *, synthetic_attrs: bool = True) -> Path:
    traces = build_synthetic_rollout_traces(horizon=2, num_samples=6, seed=13)
    sampler_reject = INVALID_REASON_CODES["SAMPLER_RULE_REJECTED"]
    valid_bit = INVALID_REASON_CODES["VALID"]
    for trace in traces:
        for step in trace.steps:
            candidate_valid = step.candidate_valid.clone()
            invalid = torch.zeros_like(candidate_valid, dtype=torch.bool)
            invalid[::4] = True
            invalid[int(step.selected_shell_index)] = False
            candidate_valid[invalid] = False
            step.candidate_valid = candidate_valid
            step.candidate_invalid_reason_bitset = torch.full(
                candidate_valid.shape,
                1 << valid_bit,
                dtype=torch.int64,
            )
            step.candidate_invalid_reason_bitset[invalid] = 1 << sampler_reject
            step.candidate_primary_invalid_reason = torch.full(
                candidate_valid.shape,
                valid_bit,
                dtype=torch.int64,
            )
            step.candidate_primary_invalid_reason[invalid] = sampler_reject
            _mask_vector(step.candidate_scores, invalid, fill=float("nan"))
            _mask_vector(step.selection_logits, invalid, fill=float("nan"))
            _mask_vector(step.selection_probabilities, invalid, fill=0.0, renormalize=True, valid=candidate_valid)
            _mask_vector(
                step.selection_log_probabilities, invalid, fill=float("-inf"), log_from=step.selection_probabilities
            )
            for values in step.metric_vectors.values():
                _mask_vector(values, invalid, fill=float("nan"))

    if synthetic_attrs:
        result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", traces)
    else:
        result = write_rollout_zarr_store(
            tmp_path / "rollouts.zarr",
            traces,
            target_protocol_version="v1-test",
            source_offline_store_version="vin-test",
        )
    return result.store_dir


def _mask_vector(
    values: torch.Tensor | None,
    invalid: torch.Tensor,
    *,
    fill: float,
    renormalize: bool = False,
    valid: torch.Tensor | None = None,
    log_from: torch.Tensor | None = None,
) -> None:
    if values is None:
        return
    if log_from is not None:
        values.copy_(torch.log(log_from.clamp_min(torch.finfo(log_from.dtype).tiny)))
        values[~torch.isfinite(log_from) | (log_from <= 0.0)] = fill
    values[invalid] = fill
    if renormalize and valid is not None:
        denom = values[valid].sum()
        if float(denom.item()) > 0.0:
            values[valid] = values[valid] / denom


def test_rollout_zarr_logger_logs_multistep_candidate_layers(tmp_path: Path) -> None:
    store_dir = _synthetic_rollout_store(tmp_path)
    cfg = RerunOfflineInspectorConfig()
    cfg.output.save_path = tmp_path / "rollout.rrd"
    cfg.selection.rollout_context_mode = "off"
    fake = _FakeRerun()
    logger = RerunRolloutZarrLogger(cfg, rr_module=fake)

    logger.start()
    rows = logger.log_store(store_dir=store_dir, rollout_index=0)

    assert [call[0] for call in fake.calls[:2]] == ["init", "save"]
    assert rows.rollout_row_id == 0
    assert rows.step_rows.shape[0] == 2
    assert ENTITY_WORLD in fake.logged
    assert ENTITY_ROLLOUT_METADATA in fake.logged
    assert ENTITY_ROLLOUT_SELECTED_PATH in fake.logged
    assert any(path.startswith(f"{ENTITY_ROLLOUT_VALID_ROOT}/candidate_") for path in fake.logged)
    assert any(path.startswith(f"{ENTITY_ROLLOUT_INVALID_ROOT}/candidate_") for path in fake.logged)
    assert any(path.startswith(f"{ENTITY_ROLLOUT_SELECTED_ROOT}/candidate_") for path in fake.logged)

    timeline_calls = [call for call in fake.calls if call[0] == "set_time_sequence"]
    assert [(call[1][0], call[1][1]) for call in timeline_calls] == [("rollout_step", 0), ("rollout_step", 1)]

    camera_calls = [
        call for call in fake.calls if call[0] == "log" and str(call[1]).startswith(f"{ENTITY_ROLLOUT_VALID_ROOT}/")
    ]
    assert camera_calls
    assert len(camera_calls[0][2]) == 2
    assert "labels" not in camera_calls[0][2][0].kwargs
    assert camera_calls[0][2][1].kwargs["candidate_row_id"] >= 0

    metadata = json.loads(fake.logged[ENTITY_ROLLOUT_METADATA].args[0])
    assert metadata["validation"]["ok"]
    assert metadata["selected"]["rollout_row_id"] == 0
    assert metadata["context"]["mode"] == "off"

    step_metadata = json.loads(fake.logged[ENTITY_ROLLOUT_STEP_METADATA].args[0])
    assert step_metadata["q_h"]["state_row_found"]
    assert not step_metadata["q_h"]["dense_q_targets_available"]
    assert step_metadata["invalid_candidate_count"] > 0
    assert step_metadata["display_validity_trusted"]

    selected_path = fake.logged[ENTITY_ROLLOUT_SELECTED_PATH]
    assert len(selected_path.args[0]) == 2
    np.testing.assert_equal(np.asarray(selected_path.args[0][0]).shape, (2, 3))
    assert len(selected_path.kwargs["colors"]) == 2
    assert selected_path.kwargs["colors"][0] != selected_path.kwargs["colors"][1]


def test_rollout_zarr_logger_logs_matching_static_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto context should reuse the normal VIN sample logger when rollout lineage resolves."""

    store_dir = _synthetic_rollout_store(tmp_path, synthetic_attrs=False)
    cfg = RerunOfflineInspectorConfig()
    cfg.output.save_path = tmp_path / "rollout.rrd"
    fake = _FakeRerun()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.select_rerun_sample",
        lambda **kwargs: calls.append(("select", kwargs["selection"]))
        or type("Selected", (), {"sample": object(), "description": "scene_id=synthetic_box snippet_id=smoke"})(),
    )
    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.collect_visual_inventory",
        lambda sample: calls.append(("inventory", sample)) or object(),
    )
    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.validate_required_inventory",
        lambda config, inventory: calls.append(("validate", inventory)),
    )

    class _FakeOfflineLogger:
        def __init__(
            self,
            config: RerunOfflineInspectorConfig,
            *,
            rr_module: object | None = None,
            target_obb_hint: str | None = None,
        ) -> None:
            del target_obb_hint
            calls.append(("logger_init", rr_module))

        def log_sample(self, *, sample: object, inventory: object, selection: str) -> None:
            calls.append(("log_sample", selection))

        def log_metadata(self, *, sample: object, inventory: object, selection: str) -> None:
            calls.append(("log_metadata", selection))

    monkeypatch.setattr("aria_nbv.rerun_inspector._rollout_zarr.RerunOfflineLogger", _FakeOfflineLogger)

    logger = RerunRolloutZarrLogger(cfg, rr_module=fake)
    logger.start()
    logger.log_store(store_dir=store_dir, rollout_index=0)

    assert [name for name, _ in calls] == [
        "select",
        "inventory",
        "validate",
        "logger_init",
        "log_sample",
        "log_metadata",
    ]
    selection = calls[0][1]
    assert selection.scene_id == "synthetic_box"
    assert selection.snippet_id == "smoke"


def test_rollout_zarr_logger_required_context_uses_selection_for_synthetic_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required context should allow synthetic rollout overlays on an explicitly selected VIN sample."""

    store_dir = _synthetic_rollout_store(tmp_path)
    cfg = RerunOfflineInspectorConfig()
    cfg.output.save_path = tmp_path / "rollout.rrd"
    cfg.selection.rollout_context_mode = "required"
    cfg.selection.split = "val"
    cfg.selection.index = 3
    fake = _FakeRerun()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.select_rerun_sample",
        lambda **kwargs: calls.append(("select", kwargs["selection"]))
        or type("Selected", (), {"sample": object(), "description": "split=val index=3"})(),
    )
    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.collect_visual_inventory",
        lambda sample: object(),
    )
    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.validate_required_inventory",
        lambda config, inventory: None,
    )

    class _FakeOfflineLogger:
        def __init__(
            self,
            config: RerunOfflineInspectorConfig,
            *,
            rr_module: object | None = None,
            target_obb_hint: str | None = None,
        ) -> None:
            del config, rr_module, target_obb_hint

        def log_sample(self, *, sample: object, inventory: object, selection: str) -> None:
            calls.append(("log_sample", selection))

        def log_metadata(self, *, sample: object, inventory: object, selection: str) -> None:
            calls.append(("log_metadata", selection))

    monkeypatch.setattr("aria_nbv.rerun_inspector._rollout_zarr.RerunOfflineLogger", _FakeOfflineLogger)

    logger = RerunRolloutZarrLogger(cfg, rr_module=fake)
    logger.start()
    logger.log_store(store_dir=store_dir, rollout_index=0)

    assert [name for name, _ in calls] == ["select", "log_sample", "log_metadata"]
    selection = calls[0][1]
    assert selection.scene_id is None
    assert selection.snippet_id is None
    assert selection.split == "val"
    assert selection.index == 3


def test_rollout_zarr_logger_aligns_synthetic_poses_to_context_world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic rollout overlays should be transformed into the selected VIN world frame."""

    store_dir = _synthetic_rollout_store(tmp_path)
    reader = RolloutZarrStoreReader(store_dir)
    store_root = reader.array("rollouts/root_pose_world")[0].astype(np.float32)
    step_rollout_ids = reader.array("steps/rollout_row_id").astype(np.int64)
    step_ids = reader.array("steps/step_row_id").astype(np.int64)
    step_indices = reader.array("steps/step_index").astype(np.int64)
    rollout_step_rows = np.nonzero(step_rollout_ids == 0)[0]
    final_step_row = rollout_step_rows[np.argmax(step_indices[rollout_step_rows])]
    final_step_id = int(step_ids[final_step_row])
    candidate_step_ids = reader.array("candidates/step_row_id").astype(np.int64)
    shell_indices = reader.array("candidates/shell_index").astype(np.int64)
    candidate_row = np.nonzero((candidate_step_ids == final_step_id) & (shell_indices == 0))[0][0]
    final_store_pose = reader.array("candidates/pose_world_cam")[candidate_row].astype(np.float32)
    context_root = store_root.copy()
    context_root[9:12] = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    expected_pose = (
        PoseTW(torch.as_tensor(context_root))
        .compose(PoseTW(torch.as_tensor(store_root)).inverse().compose(PoseTW(torch.as_tensor(final_store_pose))))
        .tensor()
        .numpy()
        .reshape(12)
    )

    cfg = RerunOfflineInspectorConfig()
    cfg.output.save_path = tmp_path / "rollout.rrd"
    cfg.selection.rollout_context_mode = "required"
    fake = _FakeRerun()

    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.select_rerun_sample",
        lambda **kwargs: type(
            "Selected",
            (),
            {
                "sample": type(
                    "Sample",
                    (),
                    {
                        "oracle": type(
                            "Oracle", (), {"reference_pose_world_rig": PoseTW(torch.as_tensor(context_root))}
                        )()
                    },
                )(),
                "description": "split=val index=0",
            },
        )(),
    )
    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.collect_visual_inventory",
        lambda sample: object(),
    )
    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.validate_required_inventory",
        lambda config, inventory: None,
    )

    class _FakeOfflineLogger:
        def __init__(
            self,
            config: RerunOfflineInspectorConfig,
            *,
            rr_module: object | None = None,
            target_obb_hint: str | None = None,
        ) -> None:
            del config, rr_module, target_obb_hint

        def log_sample(self, *, sample: object, inventory: object, selection: str) -> None:
            pass

        def log_metadata(self, *, sample: object, inventory: object, selection: str) -> None:
            pass

    monkeypatch.setattr("aria_nbv.rerun_inspector._rollout_zarr.RerunOfflineLogger", _FakeOfflineLogger)

    logger = RerunRolloutZarrLogger(cfg, rr_module=fake)
    logger.start()
    logger.log_store(store_dir=store_dir, rollout_index=0)

    camera_path = next(path for path in fake.logged if path.endswith("/candidate_000/camera"))
    transform = fake.logged[camera_path]
    np.testing.assert_allclose(
        np.asarray(transform.kwargs["translation"], dtype=np.float32),
        expected_pose[9:12],
        atol=1e-5,
    )

    candidate_values = [
        extras[1].kwargs
        for path, extras in fake.logged_extras.items()
        if path.startswith("world/rollout/step/") and path.endswith("/camera")
    ]
    assert candidate_values
    assert not any(values["valid_mask"] for values in candidate_values)
    assert any(values["stored_valid_mask"] for values in candidate_values)
    assert not any(values["display_validity_trusted"] for values in candidate_values)

    metadata = json.loads(fake.logged[ENTITY_ROLLOUT_METADATA].args[0])
    assert any("stored candidate validity" in warning for warning in metadata["context"]["warnings"])
    step_metadata = json.loads(fake.logged[ENTITY_ROLLOUT_STEP_METADATA].args[0])
    assert step_metadata["num_valid_candidates"] == 0
    assert step_metadata["stored_num_valid_candidates"] > 0
    assert not step_metadata["display_validity_trusted"]
