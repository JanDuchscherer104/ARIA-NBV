"""Rerun rollout-Zarr logger tests using a fake Rerun module."""

# ruff: noqa: S101

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

pytest.importorskip("efm3d")

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
from aria_nbv.rollouts import write_rollout_zarr_store
from tests.rollout_fixtures import build_rollout_records

if TYPE_CHECKING:
    from pathlib import Path


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
    Scalars = _Archetype
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


def _fixture_rollout_store(tmp_path: Path) -> Path:
    records = build_rollout_records(horizon=2, num_samples=6, seed=13)
    for record in records:
        for trajectory in record.result.trajectories:
            for step in trajectory.steps:
                candidate_valid = step.candidates.mask_valid.clone()
                invalid = torch.zeros_like(candidate_valid, dtype=torch.bool)
                invalid[::4] = True
                invalid[int(step.selected_shell_index)] = False
                candidate_valid[invalid] = False
                step.candidates.mask_valid = candidate_valid
                step.candidates.masks["FixtureInvalidRule"] = candidate_valid
                _mask_vector(step.selection_scores, invalid, fill=float("nan"))
                _mask_vector(step.selection_logits, invalid, fill=float("nan"))
                _mask_vector(step.selection_probabilities, invalid, fill=0.0, renormalize=True, valid=candidate_valid)
                _mask_vector(
                    step.selection_log_probabilities,
                    invalid,
                    fill=float("-inf"),
                    log_from=step.selection_probabilities,
                )
                for values in step.metric_vectors.values():
                    _mask_vector(values, invalid, fill=float("nan"))

    result = write_rollout_zarr_store(
        tmp_path / "rollouts.zarr",
        records,
        target_protocol_version="v1-observed",
        source_offline_store_version="7",
        split_manifest_hash="fixture-split-manifest",
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
    store_dir = _fixture_rollout_store(tmp_path)
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
    assert step_metadata["q_h"]["selected_transition_available"]
    assert step_metadata["invalid_candidate_count"] > 0
    assert step_metadata["display_validity_trusted"]

    selected_path = fake.logged[ENTITY_ROLLOUT_SELECTED_PATH]
    assert len(selected_path.args[0]) == 2
    np.testing.assert_equal(np.asarray(selected_path.args[0][0]).shape, (2, 3))
    assert len(selected_path.kwargs["colors"]) == 2
    assert selected_path.kwargs["colors"][0] != selected_path.kwargs["colors"][1]


def test_rollout_zarr_logger_logs_matching_static_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto context should reuse the normal VIN sample logger when rollout lineage resolves."""

    store_dir = _fixture_rollout_store(tmp_path)
    cfg = RerunOfflineInspectorConfig()
    cfg.output.save_path = tmp_path / "rollout.rrd"
    fake = _FakeRerun()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.select_rerun_sample",
        lambda **kwargs: (
            calls.append(("select", kwargs["selection"]))
            or type("Selected", (), {"sample": object(), "description": "scene_id=fixture_box snippet_id=smoke"})()
        ),
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
    assert selection.scene_id == "fixture_box"
    assert selection.snippet_id == "smoke"


def test_rollout_zarr_logger_required_context_uses_rollout_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required context should use rollout scene/snippet lineage when no explicit selector is set."""

    store_dir = _fixture_rollout_store(tmp_path)
    cfg = RerunOfflineInspectorConfig()
    cfg.output.save_path = tmp_path / "rollout.rrd"
    cfg.selection.rollout_context_mode = "required"
    cfg.selection.split = "val"
    cfg.selection.index = 3
    fake = _FakeRerun()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "aria_nbv.rerun_inspector._rollout_zarr.select_rerun_sample",
        lambda **kwargs: (
            calls.append(("select", kwargs["selection"]))
            or type("Selected", (), {"sample": object(), "description": "split=val index=3"})()
        ),
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
    assert selection.scene_id == "fixture_box"
    assert selection.snippet_id == "smoke"
