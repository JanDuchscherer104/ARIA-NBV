"""Rerun logging for standalone rollout Zarr replay stores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from aria_nbv.rollouts import RolloutZarrStoreReader, validate_rollout_zarr_store

from ._colors import INVALID_RGBA, step_to_rgba
from ._loggers import ENTITY_WORLD, RerunModule, RerunOfflineLogger, log_default_inspector_blueprint
from ._metadata import collect_visual_inventory, validate_required_inventory
from ._sample import select_rerun_sample

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ._config import RerunInspectorSelectionConfig, RerunOfflineInspectorConfig

ENTITY_ROLLOUT_ROOT = "world/rollout"
ENTITY_ROLLOUT_STEP_ROOT = f"{ENTITY_ROLLOUT_ROOT}/step"
ENTITY_ROLLOUT_VALID_ROOT = f"{ENTITY_ROLLOUT_STEP_ROOT}/valid"
ENTITY_ROLLOUT_INVALID_ROOT = f"{ENTITY_ROLLOUT_STEP_ROOT}/invalid"
ENTITY_ROLLOUT_SELECTED_ROOT = f"{ENTITY_ROLLOUT_STEP_ROOT}/selected"
ENTITY_ROLLOUT_SELECTED_PATH = f"{ENTITY_ROLLOUT_ROOT}/selected_path"
ENTITY_ROLLOUT_METADATA = "metadata/rollout_zarr"
ENTITY_ROLLOUT_STEP_METADATA = "metadata/rollout_zarr/current_step"
ENTITY_ROLLOUT_RRI_ROOT = "plots/rollout/rri"
ENTITY_ROLLOUT_DIAGNOSTICS_ROOT = "plots/rollout/diagnostics"
ENTITY_ROLLOUT_VALID_COUNT = f"{ENTITY_ROLLOUT_DIAGNOSTICS_ROOT}/selected/valid_candidates"
ENTITY_ROLLOUT_SELECTED_PROBABILITY = f"{ENTITY_ROLLOUT_DIAGNOSTICS_ROOT}/selected/selected_probability"
ENTITY_ROLLOUT_SELECTED_TARGET_RRI = f"{ENTITY_ROLLOUT_RRI_ROOT}/selected/selected_target_rri"

ROLLOUT_STEP_TIMELINE = "rollout_step"

_PLOT_PALETTE = (
    (56, 189, 248, 255),
    (251, 191, 36, 255),
    (168, 85, 247, 255),
    (34, 197, 94, 255),
    (244, 114, 182, 255),
    (249, 115, 22, 255),
)


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

            self.rr = cast("RerunModule", imported_rr)
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
        self._log_rollout_plots(reader=reader, selected_rows=rows)

        selected_path: list[list[float]] = _rollout_root_path(reader, rows=rows)
        for order, step_row_position in enumerate(rows.step_rows.tolist()):
            self._set_rollout_step_time(order)
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
            logger = RerunOfflineLogger(
                self.config,
                rr_module=self.rr,
                target_obb_hint=_rollout_target_hint(reader, rows=rows),
            )
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
        manifest_bundle = reader.manifest()
        document = {
            "store_dir": str(reader.store_dir),
            "root_attrs": attrs,
            "manifest": manifest_bundle["manifest"],
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
        self.rr.log(ENTITY_ROLLOUT_VALID_COUNT, self.rr.Scalars(float(step.valid_candidate_count)))
        self.rr.log(ENTITY_ROLLOUT_SELECTED_PROBABILITY, self.rr.Scalars(_finite_or_zero(step.selected_probability)))
        self.rr.log(ENTITY_ROLLOUT_SELECTED_TARGET_RRI, self.rr.Scalars(_finite_or_zero(step.selected_target_rri)))
        self.rr.log(
            ENTITY_ROLLOUT_STEP_METADATA,
            self.rr.TextDocument(json.dumps(step.metadata, indent=2, sort_keys=True), media_type="application/json"),
        )

    def _log_rollout_plots(self, *, reader: RolloutZarrStoreReader, selected_rows: SelectedRolloutRows) -> None:
        if not self.config.rollout_plots.enabled:
            return
        plot_rows = _resolve_plot_rollout_rows(
            reader,
            selected_rows=selected_rows,
            branch_scope=self.config.rollout_plots.branch_scope,
        )
        for branch_order, rows in enumerate(plot_rows):
            branch = _branch_plot_descriptor(reader, rows=rows, selected_row_id=selected_rows.rollout_row_id)
            self._log_branch_series_descriptors(branch=branch, branch_order=branch_order)
            for order, step_row_position in enumerate(rows.step_rows.tolist()):
                self._set_rollout_step_time(order)
                step = _plot_step_payload(
                    reader,
                    step_row_position=step_row_position,
                    candidate_top_k=self.config.rollout_plots.candidate_top_k,
                )
                self._log_branch_plot_step(branch=branch, step=step)

    def _log_branch_series_descriptors(self, *, branch: "_RolloutBranchPlot", branch_order: int) -> None:
        color = _plot_color(branch_order=branch_order, selected=branch.selected)
        muted = color.copy()
        muted[3] = min(muted[3], 160)
        series = {
            f"{branch.rri_root}/cumulative_target_rri": ("cumulative target RRI", color),
            f"{branch.rri_root}/selected_target_rri": ("selected target RRI", color),
            f"{branch.rri_root}/candidate_fanout_min": ("candidate RRI min", muted),
            f"{branch.rri_root}/candidate_fanout_mean": ("candidate RRI mean", muted),
            f"{branch.rri_root}/candidate_fanout_max": ("candidate RRI max", muted),
            f"{branch.diagnostics_root}/selected_probability": ("selected probability", color),
            f"{branch.diagnostics_root}/valid_candidates": ("valid candidates", color),
            f"{branch.diagnostics_root}/selected_entropy": ("selected entropy", muted),
            f"{branch.diagnostics_root}/selected_scene_rri": ("selected scene RRI", muted),
        }
        for rank in range(self.config.rollout_plots.candidate_top_k):
            alpha = max(90, 210 - 25 * rank)
            top_color = color.copy()
            top_color[3] = alpha
            series[f"{branch.rri_root}/candidate_top_{rank + 1:02d}"] = (f"candidate top-{rank + 1} RRI", top_color)
        for path, (label, line_color) in series.items():
            self.rr.log(
                path,
                self.rr.SeriesLines(colors=[line_color], names=[f"{branch.label} | {label}"]),
                self.rr.SeriesPoints(colors=[line_color], names=[f"{branch.label} | {label}"], marker_sizes=[5.0]),
                static=True,
            )

    def _log_branch_plot_step(self, *, branch: "_RolloutBranchPlot", step: "_RolloutPlotStep") -> None:
        self._log_scalar(f"{branch.rri_root}/cumulative_target_rri", step.cumulative_target_rri)
        self._log_scalar(f"{branch.rri_root}/selected_target_rri", step.selected_target_rri)
        self._log_scalar(f"{branch.rri_root}/candidate_fanout_min", step.candidate_min_target_rri)
        self._log_scalar(f"{branch.rri_root}/candidate_fanout_mean", step.candidate_mean_target_rri)
        self._log_scalar(f"{branch.rri_root}/candidate_fanout_max", step.candidate_max_target_rri)
        for rank, value in enumerate(step.top_candidate_target_rri, start=1):
            self._log_scalar(f"{branch.rri_root}/candidate_top_{rank:02d}", value)
        self._log_scalar(f"{branch.diagnostics_root}/selected_probability", step.selected_probability)
        self._log_scalar(f"{branch.diagnostics_root}/valid_candidates", float(step.valid_candidate_count))
        self._log_scalar(f"{branch.diagnostics_root}/selected_entropy", step.selected_entropy)
        self._log_scalar(f"{branch.diagnostics_root}/selected_scene_rri", step.selected_scene_rri)

    def _set_rollout_step_time(self, order: int) -> None:
        set_time = getattr(self.rr, "set_time", None)
        if callable(set_time):
            set_time(ROLLOUT_STEP_TIMELINE, sequence=int(order))
            return
        self.rr.set_time_sequence(ROLLOUT_STEP_TIMELINE, int(order))

    def _log_scalar(self, entity_path: str, value: float) -> None:
        if not np.isfinite(value):
            return
        self.rr.log(entity_path, self.rr.Scalars(float(value)))

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
                valid_mask=candidate.display_valid,
                stored_valid_mask=candidate.stored_valid,
                display_validity_trusted=candidate.display_validity_trusted,
                selected_mask=candidate.selected,
                target_rri=candidate.target_rri,
                selection_probability=candidate.probability,
                selection_logit=candidate.logit,
                selection_entropy=candidate.entropy,
                invalid_reason_bitset=candidate.reason_bitset,
                primary_invalid_reason=candidate.primary_reason,
                color_rgba=candidate.color,
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
        strips = [[selected_path[index], selected_path[index + 1]] for index in range(max(len(selected_path) - 1, 0))]
        colors = step_to_rgba(np.arange(len(strips), dtype=np.int64), alpha=245).astype(int).tolist()
        self.rr.log(
            ENTITY_ROLLOUT_SELECTED_PATH,
            self.rr.LineStrips3D(
                strips,
                colors=colors,
                radii=self.config.geometry.trajectory_radius,
            ),
        )


@dataclass(frozen=True, slots=True)
class _RolloutCandidatePayload:
    row_id: int
    shell_index: int
    compact_valid_index: int
    stored_valid: bool
    display_valid: bool
    display_validity_trusted: bool
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


@dataclass(frozen=True, slots=True)
class _CandidateRriSummary:
    selected_target_rri: float
    selected_scene_rri: float
    selected_probability: float
    selected_entropy: float
    valid_candidate_count: int
    candidate_min_target_rri: float
    candidate_mean_target_rri: float
    candidate_max_target_rri: float
    top_candidate_target_rri: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _RolloutPlotStep:
    step_row_id: int
    cumulative_target_rri: float
    selected_target_rri: float
    selected_scene_rri: float
    selected_probability: float
    selected_entropy: float
    valid_candidate_count: int
    candidate_min_target_rri: float
    candidate_mean_target_rri: float
    candidate_max_target_rri: float
    top_candidate_target_rri: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _RolloutBranchPlot:
    rollout_row_id: int
    rollout_index: int
    selected: bool
    label: str
    rri_root: str
    diagnostics_root: str


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


def _resolve_plot_rollout_rows(
    reader: RolloutZarrStoreReader,
    *,
    selected_rows: SelectedRolloutRows,
    branch_scope: str,
) -> list[SelectedRolloutRows]:
    """Return rollout rows included in branch-aware scalar plots."""

    if branch_scope == "selected":
        return [selected_rows]
    if branch_scope != "same_source_target":
        raise ValueError(f"Unsupported rollout plot branch_scope={branch_scope!r}.")

    rollout_ids = reader.array("rollouts/rollout_row_id").astype(np.int64).reshape(-1)
    source_ids = reader.array("rollouts/source_row_id").astype(np.int64).reshape(-1)
    target_ids = reader.array("rollouts/target_row_id").astype(np.int64).reshape(-1)
    selected_source = int(source_ids[selected_rows.rollout_index])
    selected_target = int(target_ids[selected_rows.rollout_index])
    positions = np.nonzero((source_ids == selected_source) & (target_ids == selected_target))[0]
    if positions.size == 0:
        return [selected_rows]
    rows = [
        _resolve_rollout_rows(reader, rollout_index=int(position), rollout_row_id=int(rollout_ids[int(position)]))
        for position in positions.tolist()
    ]
    rows.sort(key=lambda value: (value.rollout_row_id != selected_rows.rollout_row_id, value.rollout_row_id))
    return rows


def _branch_plot_descriptor(
    reader: RolloutZarrStoreReader,
    *,
    rows: SelectedRolloutRows,
    selected_row_id: int,
) -> _RolloutBranchPlot:
    policy = _rollout_dictionary_value(reader, group="policy", array_path="rollouts/policy_id", row=rows.rollout_index)
    chain_id = int(reader.array("rollouts/chain_id")[rows.rollout_index])
    selected = rows.rollout_row_id == selected_row_id
    suffix = f"{_safe_entity_token(policy or 'unknown_policy')}/chain_{rows.rollout_row_id:06d}"
    label = f"{policy or 'unknown'} chain={chain_id} row={rows.rollout_row_id}"
    if selected:
        label = f"selected | {label}"
    return _RolloutBranchPlot(
        rollout_row_id=rows.rollout_row_id,
        rollout_index=rows.rollout_index,
        selected=selected,
        label=label,
        rri_root=f"{ENTITY_ROLLOUT_RRI_ROOT}/{suffix}",
        diagnostics_root=f"{ENTITY_ROLLOUT_DIAGNOSTICS_ROOT}/{suffix}",
    )


def _step_payload(
    reader: RolloutZarrStoreReader,
    *,
    step_row_position: int,
) -> _RolloutStepPayload:
    step_row_id = int(reader.array("steps/step_row_id")[step_row_position])
    step_index = int(reader.array("steps/step_index")[step_row_position])
    selected_candidate_row_id = int(reader.array("steps/selected_candidate_row_id")[step_row_position])
    candidate_step_ids = reader.array("candidates/step_row_id").astype(np.int64).reshape(-1)
    row_positions = np.nonzero(candidate_step_ids == step_row_id)[0].astype(np.int64)
    shell_indices = reader.array("candidates/shell_index")[row_positions].astype(np.int64)
    order = np.argsort(shell_indices, kind="stable")
    row_positions = row_positions[order]

    stored_valid = reader.array("candidates/candidate_valid_mask")[row_positions].astype(bool)
    display_validity_trusted = True
    display_valid = stored_valid
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
        stored_valid=stored_valid,
        display_valid=display_valid,
        selected=selected,
        display_validity_trusted=display_validity_trusted,
        step_index=step_index,
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
        "num_valid_candidates": int(display_valid.sum()),
        "stored_num_valid_candidates": int(stored_valid.sum()),
        "display_validity_trusted": display_validity_trusted,
        "selected_local_index": selected_local,
        "selected_shell_index": int(shell_indices[selected_local]) if selected_local >= 0 else None,
        "selected_probability": float(probabilities[selected_local]) if selected_local >= 0 else None,
        "selected_target_rri": float(target_rri[selected_local]) if selected_local >= 0 else None,
        "selection_entropy": float(entropy[selected_local]) if selected_local >= 0 else None,
        "invalid_candidate_count": int((~display_valid).sum()),
        "stored_invalid_candidate_count": int((~stored_valid).sum()),
        "pose_frame": "stored_pose_world_cam",
        "q_h": _q_h_metadata(reader, step_row_id=step_row_id),
    }
    return _RolloutStepPayload(
        step_row_id=step_row_id,
        step_index=step_index,
        candidates=candidate_payloads,
        selected_center=centers[selected_local] if selected_local >= 0 else None,
        valid_candidate_count=int(display_valid.sum()),
        selected_probability=float(probabilities[selected_local]) if selected_local >= 0 else float("nan"),
        selected_target_rri=float(target_rri[selected_local]) if selected_local >= 0 else float("nan"),
        metadata=metadata,
    )


def _plot_step_payload(
    reader: RolloutZarrStoreReader,
    *,
    step_row_position: int,
    candidate_top_k: int,
) -> _RolloutPlotStep:
    step_row_id = int(reader.array("steps/step_row_id")[step_row_position])
    row_positions = _candidate_rows_for_step(reader, step_row_id=step_row_id)
    candidate_valid = reader.array("candidates/candidate_valid_mask")[row_positions].astype(bool)
    selected = reader.array("candidates/selected_mask")[row_positions].astype(bool)
    target_rri = reader.array("candidates/target_rri")[row_positions].astype(np.float32).reshape(-1)
    scene_rri = reader.array("candidates/scene_rri")[row_positions].astype(np.float32).reshape(-1)
    probabilities = reader.array("candidates/selection_probabilities")[row_positions].astype(np.float32).reshape(-1)
    entropy = reader.array("candidates/selection_entropy")[row_positions].astype(np.float32).reshape(-1)
    summary = _candidate_rri_summary(
        target_rri=target_rri,
        scene_rri=scene_rri,
        probabilities=probabilities,
        entropy=entropy,
        valid_mask=candidate_valid,
        selected_mask=selected,
        top_k=candidate_top_k,
    )
    return _RolloutPlotStep(
        step_row_id=step_row_id,
        cumulative_target_rri=float(reader.array("steps/cumulative_target_rri")[step_row_position]),
        selected_target_rri=summary.selected_target_rri,
        selected_scene_rri=summary.selected_scene_rri,
        selected_probability=summary.selected_probability,
        selected_entropy=summary.selected_entropy,
        valid_candidate_count=summary.valid_candidate_count,
        candidate_min_target_rri=summary.candidate_min_target_rri,
        candidate_mean_target_rri=summary.candidate_mean_target_rri,
        candidate_max_target_rri=summary.candidate_max_target_rri,
        top_candidate_target_rri=summary.top_candidate_target_rri,
    )


def _candidate_rows_for_step(reader: RolloutZarrStoreReader, *, step_row_id: int) -> NDArray[np.int64]:
    candidate_step_ids = reader.array("candidates/step_row_id").astype(np.int64).reshape(-1)
    row_positions = np.nonzero(candidate_step_ids == int(step_row_id))[0].astype(np.int64)
    shell_indices = reader.array("candidates/shell_index")[row_positions].astype(np.int64)
    return row_positions[np.argsort(shell_indices, kind="stable")]


def _candidate_rri_summary(
    *,
    target_rri: NDArray[Any],
    scene_rri: NDArray[Any],
    probabilities: NDArray[Any],
    entropy: NDArray[Any],
    valid_mask: NDArray[Any],
    selected_mask: NDArray[Any],
    top_k: int,
) -> _CandidateRriSummary:
    values = np.asarray(target_rri, dtype=np.float32).reshape(-1)
    scene_values = np.asarray(scene_rri, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    selected = np.asarray(selected_mask, dtype=bool).reshape(-1)
    finite_valid = valid & np.isfinite(values)
    finite_values = values[finite_valid]
    selected_index = int(np.nonzero(selected)[0][0]) if selected.any() else -1
    if finite_values.size:
        sorted_values = np.sort(finite_values)[::-1]
        minimum = float(np.min(finite_values))
        mean = float(np.mean(finite_values))
        maximum = float(np.max(finite_values))
        top_values = tuple(float(value) for value in sorted_values[: int(top_k)])
    else:
        minimum = mean = maximum = float("nan")
        top_values = ()
    selected_target = float(values[selected_index]) if selected_index >= 0 else float("nan")
    selected_scene = float(scene_values[selected_index]) if selected_index >= 0 else float("nan")
    selected_probability = (
        float(np.asarray(probabilities, dtype=np.float32).reshape(-1)[selected_index])
        if selected_index >= 0
        else float("nan")
    )
    selected_entropy = (
        float(np.asarray(entropy, dtype=np.float32).reshape(-1)[selected_index])
        if selected_index >= 0
        else float("nan")
    )
    return _CandidateRriSummary(
        selected_target_rri=selected_target,
        selected_scene_rri=selected_scene,
        selected_probability=selected_probability,
        selected_entropy=selected_entropy,
        valid_candidate_count=int(valid.sum()),
        candidate_min_target_rri=minimum,
        candidate_mean_target_rri=mean,
        candidate_max_target_rri=maximum,
        top_candidate_target_rri=top_values,
    )


def _q_h_metadata(reader: RolloutZarrStoreReader, *, step_row_id: int) -> dict[str, Any]:
    q_h = reader.q_h_view()
    state_step_ids = q_h["state_step_row_id"].astype(np.int64).reshape(-1)
    matches = np.nonzero(state_step_ids == int(step_row_id))[0]
    if matches.size != 1:
        return {"state_row_found": False}
    row = int(matches[0])
    valid_mask = q_h["valid_action_mask"][row].astype(bool)
    train_mask = q_h["q_train_mask"][row].astype(bool)
    return {
        "state_row_found": True,
        "q_h_state_row": row,
        "valid_action_count": int(valid_mask.sum()),
        "trainable_action_count": int(train_mask.sum()),
        "selected_candidate_index": int(q_h["selected_candidate_index"][row]),
        "td_selected_candidate_row_id": int(q_h["td_selected_candidate_row_id"][row]),
        "td_reward_target_rri": float(q_h["td_reward_target_rri"][row]),
        "td_next_step_row_id": int(q_h["td_next_step_row_id"][row]),
        "td_terminal": bool(q_h["td_terminal_mask"][row]),
        "selected_transition_available": int(q_h["td_selected_candidate_row_id"][row]) >= 0,
    }


def _pose_centers(pose_rows: NDArray[np.float32]) -> NDArray[np.float32]:
    if pose_rows.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return pose_rows.reshape(-1, 12)[:, 9:12].astype(np.float32, copy=True)


def _rollout_root_path(
    reader: RolloutZarrStoreReader,
    *,
    rows: SelectedRolloutRows,
) -> list[list[float]]:
    """Return the selected-path seed point in the displayed world frame."""

    root = reader.array("rollouts/root_pose_world")[rows.rollout_index].astype(np.float32).reshape(1, 12)
    return [_pose_centers(root)[0].tolist()]


def _candidate_payloads(
    *,
    candidate_row_ids: NDArray[Any],
    shell_indices: NDArray[Any],
    compact_valid: NDArray[Any],
    stored_valid: NDArray[Any],
    display_valid: NDArray[Any],
    selected: NDArray[Any],
    display_validity_trusted: bool,
    step_index: int,
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
        stored_valid,
        display_valid,
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
        (
            row_id,
            shell,
            compact,
            is_stored_valid,
            is_display_valid,
            is_selected,
            pose,
            center,
            rri,
            prob,
            logit,
            ent,
            reason,
            primary,
        ) = values
        shell_index = int(shell)
        group = _candidate_group(display_valid=bool(is_display_valid), selected=bool(is_selected))
        root = {
            "selected": ENTITY_ROLLOUT_SELECTED_ROOT,
            "valid": ENTITY_ROLLOUT_VALID_ROOT,
            "invalid": ENTITY_ROLLOUT_INVALID_ROOT,
        }[group]
        color = _candidate_color(
            display_valid=bool(is_display_valid),
            selected=bool(is_selected),
            step_index=step_index,
        )
        candidate_root = f"{root}/candidate_{shell_index:03d}"
        payloads.append(
            _RolloutCandidatePayload(
                row_id=int(row_id),
                shell_index=shell_index,
                compact_valid_index=int(compact),
                stored_valid=bool(is_stored_valid),
                display_valid=bool(is_display_valid),
                display_validity_trusted=display_validity_trusted,
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


def _candidate_group(*, display_valid: bool, selected: bool) -> str:
    if selected:
        return "selected"
    return "valid" if display_valid else "invalid"


def _candidate_color(*, display_valid: bool, selected: bool, step_index: int) -> list[int]:
    step_color = step_to_rgba([step_index], alpha=255 if selected else 220).reshape(1, 4)[0].astype(int).tolist()
    if selected:
        return step_color
    if display_valid:
        return step_color
    invalid_color = step_color.copy()
    invalid_color[3] = int(INVALID_RGBA[3])
    return invalid_color


def _rollout_context_selection(
    reader: RolloutZarrStoreReader,
    *,
    rows: SelectedRolloutRows,
    fallback: RerunInspectorSelectionConfig,
) -> RerunInspectorSelectionConfig | None:
    if fallback.sample_key or (fallback.scene_id and fallback.snippet_id):
        return fallback.model_copy(deep=True)

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
    if fallback.rollout_context_mode == "required":
        return fallback.model_copy(deep=True)
    return None


def _rollout_dictionary_value(
    reader: RolloutZarrStoreReader,
    *,
    group: str,
    array_path: str,
    row: int,
) -> str | None:
    dictionary = _read_string_dictionary(reader, f"dictionaries/{group}")
    index = int(reader.array(array_path)[row])
    if index < 0 or index >= len(dictionary):
        return None
    value = dictionary[index].strip()
    return value or None


def _dictionary_preview(reader: RolloutZarrStoreReader) -> dict[str, list[str]]:
    return {
        name: _read_string_dictionary(reader, f"dictionaries/{name}")[:20]
        for name in ("scene", "snippet", "rollout", "target", "policy", "termination_reason")
    }


def _rollout_target_hint(reader: RolloutZarrStoreReader, *, rows: SelectedRolloutRows) -> str | None:
    """Return a target dictionary value for optional OBB highlighting."""

    target_rows = reader.array("targets/target_row_id").astype(np.int64).reshape(-1)
    target_row_id = int(reader.array("rollouts/target_row_id")[rows.rollout_index])
    names = _read_string_dictionary(reader, "dictionaries/target")
    matches = np.nonzero(target_rows == target_row_id)[0]
    if matches.size != 1:
        return str(target_row_id)
    match_index = int(matches[0])
    gt_target_ids = reader.array("targets/matched_gt_target_id").astype(np.int64).reshape(-1)
    target_ids = reader.array("targets/target_id").astype(np.int64).reshape(-1)
    name_index = int(gt_target_ids[match_index])
    if name_index < 0:
        name_index = int(target_ids[match_index])
    if 0 <= name_index < len(names):
        return names[name_index]
    return str(target_row_id)


def _read_string_dictionary(reader: RolloutZarrStoreReader, path: str) -> list[str]:
    encoded = reader.array(path).astype(np.uint8).reshape(-1).tobytes()
    values = json.loads(encoded.decode("utf-8"))
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _safe_entity_token(value: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    token = "_".join(part for part in token.split("_") if part)
    return token or "unknown"


def _plot_color(*, branch_order: int, selected: bool) -> list[int]:
    color = list(_PLOT_PALETTE[branch_order % len(_PLOT_PALETTE)])
    color[3] = 255 if selected else 150
    return color


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
    "ENTITY_ROLLOUT_DIAGNOSTICS_ROOT",
    "ENTITY_ROLLOUT_METADATA",
    "ENTITY_ROLLOUT_RRI_ROOT",
    "ENTITY_ROLLOUT_SELECTED_PATH",
    "ENTITY_ROLLOUT_SELECTED_ROOT",
    "ENTITY_ROLLOUT_STEP_ROOT",
    "ENTITY_ROLLOUT_VALID_ROOT",
    "RerunRolloutZarrLogger",
    "SelectedRolloutRows",
    "run_rollout_zarr_inspector",
]
