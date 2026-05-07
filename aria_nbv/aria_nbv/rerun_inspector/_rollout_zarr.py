"""Rerun logging for standalone rollout Zarr replay stores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from aria_nbv.data_handling import RolloutZarrStoreReader, validate_rollout_zarr_store

from ._colors import INVALID_RGBA, oracle_rri_to_rgba
from ._config import RerunInspectorSelectionConfig, RerunOfflineInspectorConfig
from ._loggers import ENTITY_WORLD, RerunModule, RerunOfflineLogger, log_default_inspector_blueprint
from ._metadata import collect_visual_inventory, validate_required_inventory
from ._sample import select_rerun_sample

ENTITY_ROLLOUT_ROOT = "world/rollout"
ENTITY_ROLLOUT_STEP_ROOT = f"{ENTITY_ROLLOUT_ROOT}/step"
ENTITY_ROLLOUT_VALID_ROOT = f"{ENTITY_ROLLOUT_STEP_ROOT}/valid"
ENTITY_ROLLOUT_INVALID_ROOT = f"{ENTITY_ROLLOUT_STEP_ROOT}/invalid"
ENTITY_ROLLOUT_SELECTED_ROOT = f"{ENTITY_ROLLOUT_STEP_ROOT}/selected"
ENTITY_ROLLOUT_SELECTED_PATH = f"{ENTITY_ROLLOUT_ROOT}/selected_path"
ENTITY_ROLLOUT_METADATA = "metadata/rollout_zarr"
ENTITY_ROLLOUT_STEP_METADATA = "metadata/rollout_zarr/current_step"
ENTITY_ROLLOUT_VALID_COUNT = "plots/rollout/valid_candidates"
ENTITY_ROLLOUT_SELECTED_PROBABILITY = "plots/rollout/selected_probability"
ENTITY_ROLLOUT_SELECTED_TARGET_RRI = "plots/rollout/selected_target_rri"

ROLLOUT_STEP_TIMELINE = "rollout_step"


@dataclass(frozen=True, slots=True)
class SelectedRolloutRows:
    """Resolved row ids for one rollout chain in a standalone replay store."""

    rollout_row_id: int
    rollout_index: int
    step_rows: NDArray[np.int64]


class RerunRolloutZarrLogger:
    """Log one multistep rollout chain from ``rollouts.zarr`` to Rerun."""

    def __init__(self, config: RerunOfflineInspectorConfig, *, rr_module: RerunModule | None = None) -> None:
        """Create a rollout-store logger."""

        self.config = config
        if rr_module is None:
            import rerun as imported_rr

            self.rr = cast(RerunModule, imported_rr)
        else:
            self.rr = rr_module
        self._context_warnings: list[str] = []

    def start(self) -> None:
        """Initialize the Rerun recording and configured output sink."""

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
        self.rr.log(ENTITY_WORLD, self.rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def log_store(
        self,
        *,
        store_dir: Path | str,
        rollout_index: int = 0,
        rollout_row_id: int | None = None,
    ) -> SelectedRolloutRows:
        """Log one rollout chain from a validated rollout Zarr store."""

        reader = RolloutZarrStoreReader(store_dir)
        validation = validate_rollout_zarr_store(store_dir)
        rows = _resolve_rollout_rows(reader, rollout_index=rollout_index, rollout_row_id=rollout_row_id)
        self._log_static_context(reader=reader, rows=rows)
        self._log_static_metadata(reader=reader, rows=rows, validation_errors=validation.errors)

        selected_path: list[list[float]] = []
        for order, step_row_position in enumerate(rows.step_rows.tolist()):
            self.rr.set_time_sequence(ROLLOUT_STEP_TIMELINE, order)
            step = _step_payload(reader, step_row_position=step_row_position)
            self._log_step(step)
            if step.selected_center is not None:
                selected_path.append(step.selected_center.tolist())
            self._log_selected_path(selected_path)
        return rows

    def _log_static_context(self, *, reader: RolloutZarrStoreReader, rows: SelectedRolloutRows) -> None:
        """Log matching VIN offline sample context before rollout-step layers."""

        mode = self.config.selection.rollout_context_mode
        if mode == "off":
            self._context_warnings.append("VIN context logging disabled by selection.rollout_context_mode='off'.")
            return
        if _synthetic_rollout_store(reader) and mode == "auto" and not _has_explicit_context_selector(
            self.config.selection
        ):
            message = "VIN context logging skipped for synthetic rollout store."
            self._context_warnings.append(message)
            return
        selection = _rollout_context_selection(reader, rows=rows, fallback=self.config.selection)
        if selection is None:
            message = "No rollout scene/snippet or explicit sample selector available for VIN context logging."
            if mode == "required":
                raise LookupError(message)
            self._context_warnings.append(message)
            return
        try:
            selected = select_rerun_sample(dataset_config=self.config.dataset.offline, selection=selection)
            inventory = collect_visual_inventory(selected.sample)
            validate_required_inventory(self.config, inventory)
            logger = RerunOfflineLogger(self.config, rr_module=self.rr)
            logger.log_sample(sample=selected.sample, inventory=inventory, selection=selected.description)
            logger.log_metadata(sample=selected.sample, inventory=inventory, selection=selected.description)
        except Exception as exc:
            if mode == "required":
                raise
            self._context_warnings.append(f"VIN context logging skipped: {exc}")

    def _log_static_metadata(
        self,
        *,
        reader: RolloutZarrStoreReader,
        rows: SelectedRolloutRows,
        validation_errors: list[str],
    ) -> None:
        attrs = dict(reader.root.attrs)
        document = {
            "store_dir": str(reader.store_dir),
            "root_attrs": attrs,
            "selected": {
                "rollout_row_id": rows.rollout_row_id,
                "rollout_index": rows.rollout_index,
                "step_rows": rows.step_rows.astype(int).tolist(),
            },
            "validation": {"ok": not validation_errors, "errors": validation_errors},
            "context": {
                "mode": self.config.selection.rollout_context_mode,
                "warnings": list(self._context_warnings),
            },
            "dictionaries": _dictionary_preview(reader),
        }
        self.rr.log(
            ENTITY_ROLLOUT_METADATA,
            self.rr.TextDocument(json.dumps(document, indent=2, sort_keys=True), media_type="application/json"),
            static=True,
        )

    def _log_step(self, step: "_RolloutStepPayload") -> None:
        for candidate in step.candidates:
            self._log_candidate_camera(candidate)
            self._log_candidate_center(candidate)
        self.rr.log(ENTITY_ROLLOUT_VALID_COUNT, self.rr.Scalar(float(step.valid_candidate_count)))
        self.rr.log(ENTITY_ROLLOUT_SELECTED_PROBABILITY, self.rr.Scalar(_finite_or_zero(step.selected_probability)))
        self.rr.log(ENTITY_ROLLOUT_SELECTED_TARGET_RRI, self.rr.Scalar(_finite_or_zero(step.selected_target_rri)))
        self.rr.log(
            ENTITY_ROLLOUT_STEP_METADATA,
            self.rr.TextDocument(json.dumps(step.metadata, indent=2, sort_keys=True), media_type="application/json"),
        )

    def _log_candidate_camera(self, candidate: "_RolloutCandidatePayload") -> None:
        rotation = candidate.pose[:9].reshape(3, 3)
        translation = candidate.pose[9:12]
        self.rr.log(
            candidate.camera_entity,
            self.rr.Transform3D(
                translation=translation.astype(float).tolist(),
                mat3x3=rotation.astype(float).tolist(),
                relation=self.rr.TransformRelation.ParentFromChild,
            ),
            self.rr.Pinhole(
                fov_y=float(np.pi / 2.0),
                aspect_ratio=1.0,
                camera_xyz=self.rr.ViewCoordinates.LUF,
                image_plane_distance=self.config.geometry.frustum_scale,
            ),
            self.rr.AnyValues(
                candidate_row_id=candidate.row_id,
                shell_index=candidate.shell_index,
                compact_valid_index=candidate.compact_valid_index,
                valid_mask=candidate.valid,
                selected_mask=candidate.selected,
                target_rri=candidate.target_rri,
                selection_probability=candidate.probability,
                selection_logit=candidate.logit,
                selection_entropy=candidate.entropy,
                invalid_reason_bitset=candidate.reason_bitset,
                primary_invalid_reason=candidate.primary_reason,
            ),
        )

    def _log_candidate_center(self, candidate: "_RolloutCandidatePayload") -> None:
        self.rr.log(
            candidate.center_entity,
            self.rr.Points3D(
                candidate.center.reshape(1, 3),
                radii=self.config.geometry.candidate_center_radius,
                colors=[candidate.color],
            ),
        )

    def _log_selected_path(self, selected_path: list[list[float]]) -> None:
        strips = [selected_path] if len(selected_path) >= 2 else []
        self.rr.log(
            ENTITY_ROLLOUT_SELECTED_PATH,
            self.rr.LineStrips3D(
                strips,
                colors=_repeat_color([255, 255, 255, 230], len(strips)),
                radii=self.config.geometry.trajectory_radius,
            ),
        )


@dataclass(frozen=True, slots=True)
class _RolloutCandidatePayload:
    row_id: int
    shell_index: int
    compact_valid_index: int
    valid: bool
    selected: bool
    pose: NDArray[np.float32]
    center: NDArray[np.float32]
    target_rri: float
    probability: float
    logit: float
    entropy: float
    reason_bitset: int
    primary_reason: int
    color: list[int]
    camera_entity: str
    center_entity: str


@dataclass(frozen=True, slots=True)
class _RolloutStepPayload:
    step_row_id: int
    step_index: int
    candidates: list[_RolloutCandidatePayload]
    selected_center: NDArray[np.float32] | None
    valid_candidate_count: int
    selected_probability: float
    selected_target_rri: float
    metadata: dict[str, Any]


def _resolve_rollout_rows(
    reader: RolloutZarrStoreReader,
    *,
    rollout_index: int,
    rollout_row_id: int | None,
) -> SelectedRolloutRows:
    rollout_ids = reader.array("rollouts/rollout_row_id").astype(np.int64).reshape(-1)
    if rollout_row_id is None:
        if int(rollout_index) < 0 or int(rollout_index) >= int(rollout_ids.shape[0]):
            raise IndexError(f"rollout_index {rollout_index} is outside [0, {rollout_ids.shape[0]}).")
        resolved_row_id = int(rollout_ids[int(rollout_index)])
        resolved_index = int(rollout_index)
    else:
        matches = np.nonzero(rollout_ids == int(rollout_row_id))[0]
        if matches.size != 1:
            raise KeyError(f"rollout_row_id {rollout_row_id} is not present in rollouts/rollout_row_id.")
        resolved_row_id = int(rollout_row_id)
        resolved_index = int(matches[0])

    step_rollout_ids = reader.array("steps/rollout_row_id").astype(np.int64).reshape(-1)
    step_indices = reader.array("steps/step_index").astype(np.int64).reshape(-1)
    step_rows = np.nonzero(step_rollout_ids == resolved_row_id)[0].astype(np.int64)
    if step_rows.size == 0:
        raise ValueError(f"Rollout row {resolved_row_id} has no step rows.")
    order = np.argsort(step_indices[step_rows], kind="stable")
    return SelectedRolloutRows(
        rollout_row_id=resolved_row_id,
        rollout_index=resolved_index,
        step_rows=step_rows[order],
    )


def _step_payload(reader: RolloutZarrStoreReader, *, step_row_position: int) -> _RolloutStepPayload:
    step_row_id = int(reader.array("steps/step_row_id")[step_row_position])
    step_index = int(reader.array("steps/step_index")[step_row_position])
    selected_candidate_row_id = int(reader.array("steps/selected_candidate_row_id")[step_row_position])
    candidate_step_ids = reader.array("candidates/step_row_id").astype(np.int64).reshape(-1)
    row_positions = np.nonzero(candidate_step_ids == step_row_id)[0].astype(np.int64)
    shell_indices = reader.array("candidates/shell_index")[row_positions].astype(np.int64)
    order = np.argsort(shell_indices, kind="stable")
    row_positions = row_positions[order]

    valid = reader.array("candidates/candidate_valid_mask")[row_positions].astype(bool)
    selected = reader.array("candidates/selected_mask")[row_positions].astype(bool)
    poses = reader.array("candidates/pose_world_cam")[row_positions].astype(np.float32).reshape(-1, 12)
    centers = _pose_centers(poses)
    target_rri = reader.array("candidates/target_rri")[row_positions].astype(np.float32).reshape(-1)
    probabilities = reader.array("candidates/selection_probabilities")[row_positions].astype(np.float32).reshape(-1)
    logits = reader.array("candidates/selection_logits")[row_positions].astype(np.float32).reshape(-1)
    entropy = reader.array("candidates/selection_entropy")[row_positions].astype(np.float32).reshape(-1)
    reason_bitsets = reader.array("candidates/invalid_reason_bitset")[row_positions].astype(np.uint32).reshape(-1)
    primary_reasons = reader.array("candidates/primary_invalid_reason")[row_positions].astype(np.uint16).reshape(-1)
    compact_valid = reader.array("candidates/compact_valid_index")[row_positions].astype(np.int64).reshape(-1)
    candidate_row_ids = reader.array("candidates/candidate_row_id")[row_positions].astype(np.int64).reshape(-1)

    selected_local = int(np.nonzero(selected)[0][0]) if selected.any() else -1
    candidate_payloads = _candidate_payloads(
        candidate_row_ids=candidate_row_ids,
        shell_indices=shell_indices,
        compact_valid=compact_valid,
        valid=valid,
        selected=selected,
        poses=poses,
        centers=centers,
        target_rri=target_rri,
        probabilities=probabilities,
        logits=logits,
        entropy=entropy,
        reason_bitsets=reason_bitsets,
        primary_reasons=primary_reasons,
    )
    metadata = {
        "step_row_id": step_row_id,
        "step_index": step_index,
        "selected_candidate_row_id": selected_candidate_row_id,
        "num_candidates": int(row_positions.shape[0]),
        "num_valid_candidates": int(valid.sum()),
        "selected_local_index": selected_local,
        "selected_shell_index": int(shell_indices[selected_local]) if selected_local >= 0 else None,
        "selected_probability": float(probabilities[selected_local]) if selected_local >= 0 else None,
        "selected_target_rri": float(target_rri[selected_local]) if selected_local >= 0 else None,
        "selection_entropy": float(entropy[selected_local]) if selected_local >= 0 else None,
        "invalid_candidate_count": int((~valid).sum()),
        "q_h": _q_h_metadata(reader, step_row_id=step_row_id),
    }
    return _RolloutStepPayload(
        step_row_id=step_row_id,
        step_index=step_index,
        candidates=candidate_payloads,
        selected_center=centers[selected_local] if selected_local >= 0 else None,
        valid_candidate_count=int(valid.sum()),
        selected_probability=float(probabilities[selected_local]) if selected_local >= 0 else float("nan"),
        selected_target_rri=float(target_rri[selected_local]) if selected_local >= 0 else float("nan"),
        metadata=metadata,
    )


def _q_h_metadata(reader: RolloutZarrStoreReader, *, step_row_id: int) -> dict[str, Any]:
    state_step_ids = reader.array("q_h/state_step_row_id").astype(np.int64).reshape(-1)
    matches = np.nonzero(state_step_ids == int(step_row_id))[0]
    if matches.size != 1:
        return {"state_row_found": False}
    row = int(matches[0])
    valid_mask = reader.array("q_h/valid_action_mask")[row].astype(bool)
    train_mask = reader.array("q_h/q_train_mask")[row].astype(bool)
    dense_available = reader.array("q_h/q_target_available_mask")[row].astype(bool)
    return {
        "state_row_found": True,
        "q_h_state_row": row,
        "valid_action_count": int(valid_mask.sum()),
        "trainable_action_count": int(train_mask.sum()),
        "selected_candidate_index": int(reader.array("q_h/selected_candidate_index")[row]),
        "td_selected_candidate_row_id": int(reader.array("q_h/td_selected_candidate_row_id")[row]),
        "td_next_step_row_id": int(reader.array("q_h/td_next_step_row_id")[row]),
        "td_terminal": bool(reader.array("q_h/td_terminal_mask")[row]),
        "dense_q_targets_available": bool(dense_available.any()),
    }


def _pose_centers(pose_rows: NDArray[np.float32]) -> NDArray[np.float32]:
    if pose_rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return pose_rows.reshape(-1, 12)[:, 9:12].astype(np.float32, copy=True)


def _candidate_payloads(
    *,
    candidate_row_ids: NDArray[Any],
    shell_indices: NDArray[Any],
    compact_valid: NDArray[Any],
    valid: NDArray[Any],
    selected: NDArray[Any],
    poses: NDArray[np.float32],
    centers: NDArray[np.float32],
    target_rri: NDArray[Any],
    probabilities: NDArray[Any],
    logits: NDArray[Any],
    entropy: NDArray[Any],
    reason_bitsets: NDArray[Any],
    primary_reasons: NDArray[Any],
) -> list[_RolloutCandidatePayload]:
    payloads: list[_RolloutCandidatePayload] = []
    for values in zip(
        candidate_row_ids,
        shell_indices,
        compact_valid,
        valid,
        selected,
        poses,
        centers,
        target_rri,
        probabilities,
        logits,
        entropy,
        reason_bitsets,
        primary_reasons,
        strict=False,
    ):
        row_id, shell, compact, is_valid, is_selected, pose, center, rri, prob, logit, ent, reason, primary = values
        shell_index = int(shell)
        group = _candidate_group(valid=bool(is_valid), selected=bool(is_selected))
        root = {
            "selected": ENTITY_ROLLOUT_SELECTED_ROOT,
            "valid": ENTITY_ROLLOUT_VALID_ROOT,
            "invalid": ENTITY_ROLLOUT_INVALID_ROOT,
        }[group]
        color = _candidate_color(valid=bool(is_valid), selected=bool(is_selected), target_rri=float(rri))
        candidate_root = f"{root}/candidate_{shell_index:03d}"
        payloads.append(
            _RolloutCandidatePayload(
                row_id=int(row_id),
                shell_index=shell_index,
                compact_valid_index=int(compact),
                valid=bool(is_valid),
                selected=bool(is_selected),
                pose=np.asarray(pose, dtype=np.float32).reshape(12),
                center=np.asarray(center, dtype=np.float32).reshape(3),
                target_rri=float(rri),
                probability=float(prob),
                logit=float(logit),
                entropy=float(ent),
                reason_bitset=int(reason),
                primary_reason=int(primary),
                color=color,
                camera_entity=f"{candidate_root}/camera",
                center_entity=f"{candidate_root}/center",
            ),
        )
    return payloads


def _candidate_group(*, valid: bool, selected: bool) -> str:
    if selected:
        return "selected"
    return "valid" if valid else "invalid"


def _candidate_color(*, valid: bool, selected: bool, target_rri: float) -> list[int]:
    if selected:
        return [255, 235, 120, 255]
    if not valid:
        return INVALID_RGBA.tolist()
    return (
        oracle_rri_to_rgba(np.asarray([target_rri], dtype=np.float32), alpha=220).reshape(1, 4)[0].astype(int).tolist()
    )


def _rollout_context_selection(
    reader: RolloutZarrStoreReader,
    *,
    rows: SelectedRolloutRows,
    fallback: RerunInspectorSelectionConfig,
) -> RerunInspectorSelectionConfig | None:
    scene_id = _rollout_dictionary_value(reader, group="scene", array_path="rollouts/scene_id", row=rows.rollout_index)
    snippet_id = _rollout_dictionary_value(
        reader,
        group="snippet",
        array_path="rollouts/snippet_id",
        row=rows.rollout_index,
    )
    if scene_id and snippet_id:
        return fallback.model_copy(
            deep=True, update={"scene_id": scene_id, "snippet_id": snippet_id, "sample_key": None}
        )
    if fallback.sample_key or (fallback.scene_id and fallback.snippet_id) or fallback.rollout_context_mode == "required":
        return fallback.model_copy(deep=True)
    return None


def _has_explicit_context_selector(selection: RerunInspectorSelectionConfig) -> bool:
    return bool(selection.sample_key or (selection.scene_id and selection.snippet_id))


def _synthetic_rollout_store(reader: RolloutZarrStoreReader) -> bool:
    attrs = dict(reader.root.attrs)
    synthetic_values = {
        str(attrs.get("source_offline_store_version", "")).lower(),
        str(attrs.get("target_protocol_version", "")).lower(),
    }
    return "synthetic" in synthetic_values


def _rollout_dictionary_value(
    reader: RolloutZarrStoreReader,
    *,
    group: str,
    array_path: str,
    row: int,
) -> str | None:
    try:
        dictionary = _read_string_dictionary(reader, f"dictionaries/{group}")
        index = int(reader.array(array_path)[row])
    except Exception:
        return None
    if index < 0 or index >= len(dictionary):
        return None
    value = dictionary[index].strip()
    return value or None


def _dictionary_preview(reader: RolloutZarrStoreReader) -> dict[str, list[str]]:
    preview: dict[str, list[str]] = {}
    for name in ("scene", "snippet", "rollout", "target", "policy", "termination_reason"):
        try:
            preview[name] = _read_string_dictionary(reader, f"dictionaries/{name}")[:20]
        except Exception:
            preview[name] = []
    return preview


def _read_string_dictionary(reader: RolloutZarrStoreReader, path: str) -> list[str]:
    encoded = reader.array(path).astype(np.uint8).reshape(-1).tobytes()
    values = json.loads(encoded.decode("utf-8"))
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _repeat_color(color: list[int], count: int) -> list[list[int]]:
    return [list(map(int, color)) for _ in range(max(int(count), 0))]


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def run_rollout_zarr_inspector(
    config: RerunOfflineInspectorConfig,
    *,
    store_dir: Path | str,
    rollout_index: int = 0,
    rollout_row_id: int | None = None,
    rr_module: RerunModule | None = None,
) -> SelectedRolloutRows:
    """Run the Rerun rollout-store inspector for tests and CLI callers."""

    logger = RerunRolloutZarrLogger(config, rr_module=rr_module)
    logger.start()
    return logger.log_store(store_dir=store_dir, rollout_index=rollout_index, rollout_row_id=rollout_row_id)


__all__ = [
    "ENTITY_ROLLOUT_INVALID_ROOT",
    "ENTITY_ROLLOUT_METADATA",
    "ENTITY_ROLLOUT_SELECTED_PATH",
    "ENTITY_ROLLOUT_SELECTED_ROOT",
    "ENTITY_ROLLOUT_STEP_ROOT",
    "ENTITY_ROLLOUT_VALID_ROOT",
    "RerunRolloutZarrLogger",
    "SelectedRolloutRows",
    "run_rollout_zarr_inspector",
]
