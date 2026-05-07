"""Standalone Zarr replay store for counterfactual rollout traces.

The store is the first concrete rollout-data path for finite-candidate
``Q_H`` training. It writes compact row tables from :class:`RolloutTrace`
objects, keeps full-shell candidate rows for replayability, and derives padded
``q_h`` arrays for selected-action TD/replay training. Dense all-action oracle-Q
targets are present as schema-ready arrays but intentionally remain unavailable
until a later converter materializes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from efm3d.aria.pose import PoseTW
from pydantic import Field, field_validator

from ..configs import PathConfig
from ..pose_generation import INVALID_REASON_CODES, INVALID_REASON_VERSION, RolloutTrace
from ..utils import BaseConfig
from ._config_utils import resolve_cache_artifact_dir

ROLLOUT_ZARR_SCHEMA_ID = "aria_nbv.rollout_zarr_q_invalidity"
"""Schema id stored as a root attribute on rollout replay stores."""

ROLLOUT_ZARR_SCHEMA_VERSION = "0.1-tracer"
"""First implemented tracer-bullet rollout replay schema version."""

DEFAULT_RETURN_SEMANTICS = "cumulative_target_rri"
"""Default return target family for initial ``Q_H`` replay views."""


@dataclass(slots=True)
class RolloutZarrWriteResult:
    """Summary of one rollout Zarr write."""

    store_dir: Path
    num_rollouts: int
    num_steps: int
    num_candidates: int
    q_h_state_count: int
    max_candidates_per_step: int


@dataclass(slots=True)
class RolloutZarrValidationResult:
    """Validation summary for a rollout Zarr store."""

    store_dir: Path
    num_rollouts: int
    num_steps: int
    num_candidates: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return ``True`` when no validation errors were found."""

        return not self.errors


class RolloutZarrStoreConfig(BaseConfig):
    """Filesystem and target metadata for one standalone rollout replay store."""

    @property
    def target(self) -> type["RolloutZarrStoreWriter"]:
        return RolloutZarrStoreWriter

    paths: PathConfig = Field(default_factory=PathConfig)
    store_dir: Path = Field(default_factory=lambda: PathConfig().offline_cache_dir / "rollouts.zarr")
    return_semantics: str = DEFAULT_RETURN_SEMANTICS
    discount_gamma: float = 1.0
    target_protocol_version: str = "synthetic"
    reason_code_version: str = INVALID_REASON_VERSION
    field_retention_policy: str = "compact"
    source_offline_store_version: str = "synthetic"
    split_manifest_hash: str = "synthetic"

    _resolve_store_dir = field_validator("store_dir", mode="before")(resolve_cache_artifact_dir)

    @field_validator("discount_gamma")
    @classmethod
    def _valid_discount(cls, value: float) -> float:
        value = float(value)
        if value < 0.0:
            raise ValueError("discount_gamma must be >= 0.")
        return value


class RolloutZarrStoreWriter:
    """Write standalone rollout replay stores from :class:`RolloutTrace` objects."""

    def __init__(self, config: RolloutZarrStoreConfig) -> None:
        self.config = config

    def write(self, traces: list[RolloutTrace]) -> RolloutZarrWriteResult:
        """Materialize a rollout replay store.

        Args:
            traces: Rollout traces to write. Each trace is one rollout chain.

        Returns:
            Row-count summary for the written store.
        """

        return write_rollout_zarr_store(
            self.config.store_dir,
            traces,
            return_semantics=self.config.return_semantics,
            discount_gamma=self.config.discount_gamma,
            target_protocol_version=self.config.target_protocol_version,
            reason_code_version=self.config.reason_code_version,
            field_retention_policy=self.config.field_retention_policy,
            source_offline_store_version=self.config.source_offline_store_version,
            split_manifest_hash=self.config.split_manifest_hash,
        )


class RolloutZarrStoreReader:
    """Open and validate standalone rollout replay stores."""

    def __init__(self, store_dir: Path | str) -> None:
        self.store_dir = Path(store_dir).expanduser().resolve()
        self.root = zarr.open_group(store=zarr.storage.LocalStore(str(self.store_dir), read_only=True), mode="r")

    def array(self, path: str) -> np.ndarray:
        """Read an array by slash-separated Zarr path."""

        return np.asarray(self.root[path])

    def validate(self) -> RolloutZarrValidationResult:
        """Validate row linkage, masks, and initial ``Q_H`` target availability."""

        return validate_rollout_zarr_store(self.store_dir)


def write_rollout_zarr_store(
    store_dir: Path | str,
    traces: list[RolloutTrace],
    *,
    return_semantics: str = DEFAULT_RETURN_SEMANTICS,
    discount_gamma: float = 1.0,
    target_protocol_version: str = "synthetic",
    reason_code_version: str = INVALID_REASON_VERSION,
    field_retention_policy: str = "compact",
    source_offline_store_version: str = "synthetic",
    split_manifest_hash: str = "synthetic",
) -> RolloutZarrWriteResult:
    """Write rollout traces into a standalone ``rollouts.zarr`` store."""

    output_dir = Path(store_dir).expanduser().resolve()
    root = zarr.open_group(str(output_dir), mode="w")
    _write_root_metadata(
        root,
        traces=traces,
        return_semantics=return_semantics,
        discount_gamma=discount_gamma,
        target_protocol_version=target_protocol_version,
        reason_code_version=reason_code_version,
        field_retention_policy=field_retention_policy,
        source_offline_store_version=source_offline_store_version,
        split_manifest_hash=split_manifest_hash,
    )
    groups = {name: root.create_group(name, overwrite=True) for name in _required_groups()}

    dictionaries = _build_dictionaries(traces)
    _write_dictionaries(groups["dictionaries"], dictionaries)
    _write_metadata_group(groups["metadata"], field_retention_policy=field_retention_policy)
    _write_targets(groups["targets"], traces, dictionaries, target_protocol_version=target_protocol_version)
    _write_mesh_refs(groups["mesh_refs"])

    table = _flatten_traces(traces, dictionaries)
    _write_rollout_tables(groups, table)
    q_h = _build_q_h_arrays(table, horizon=max((trace.horizon for trace in traces), default=0), gamma=discount_gamma)
    _write_q_h(groups["q_h"], q_h, return_semantics=return_semantics, target_protocol_version=target_protocol_version)

    return RolloutZarrWriteResult(
        store_dir=output_dir,
        num_rollouts=len(traces),
        num_steps=int(table["step_step_row_id"].shape[0]),
        num_candidates=int(table["candidate_candidate_row_id"].shape[0]),
        q_h_state_count=int(q_h["state_step_row_id"].shape[0]),
        max_candidates_per_step=int(q_h["candidate_row_id"].shape[1]) if q_h["candidate_row_id"].ndim == 2 else 0,
    )


def validate_rollout_zarr_store(store_dir: Path | str) -> RolloutZarrValidationResult:
    """Validate a standalone rollout replay store and return all discovered errors."""

    root = zarr.open_group(
        store=zarr.storage.LocalStore(str(Path(store_dir).expanduser().resolve()), read_only=True),
        mode="r",
    )
    errors: list[str] = []
    for group_name in _required_groups():
        if group_name not in root:
            errors.append(f"Missing required group {group_name!r}.")

    if errors:
        return RolloutZarrValidationResult(Path(store_dir), 0, 0, 0, errors)

    candidate_row_id = np.asarray(root["candidates/candidate_row_id"])
    q_candidate_row_id = np.asarray(root["q_h/candidate_row_id"])
    q_train_mask = np.asarray(root["q_h/q_train_mask"])
    q_target = np.asarray(root["q_h/q_target_target_rri"])
    valid_action_mask = np.asarray(root["q_h/valid_action_mask"])
    one_step_target_rri = np.asarray(root["q_h/one_step_target_rri"])

    real_q_ids = q_candidate_row_id[q_candidate_row_id >= 0]
    if not np.isin(real_q_ids, candidate_row_id).all():
        errors.append("Q_H candidate_row_id contains ids not present in candidates/candidate_row_id.")
    if np.any(q_train_mask & (~valid_action_mask)):
        errors.append("Q_H q_train_mask is true for invalid or padded candidates.")
    if np.any(q_train_mask & (~np.isfinite(one_step_target_rri))):
        errors.append("Q_H q_train_mask is true without a finite explicit target-RRI label.")
    if np.isfinite(q_target[~np.asarray(root["q_h/q_target_available_mask"])]).any():
        errors.append("Unavailable dense Q targets must remain NaN.")

    selected_mask = np.asarray(root["candidates/selected_mask"])
    actor_action_mask = np.asarray(root["candidates/actor_action_mask"])
    if np.any(selected_mask & (~actor_action_mask)):
        errors.append("Selected candidates must be actor-selectable.")

    target_row_id = np.asarray(root["targets/target_row_id"])
    rollout_target_row_id = np.asarray(root["rollouts/target_row_id"])
    if not np.isin(rollout_target_row_id, target_row_id).all():
        errors.append("Rollout target_row_id contains ids not present in targets/target_row_id.")
    if "root_pose_world" not in root["rollouts"]:
        errors.append("Missing required rollout root_pose_world field.")
    else:
        root_pose_world = np.asarray(root["rollouts/root_pose_world"])
        if root_pose_world.shape != (int(np.asarray(root["rollouts/rollout_row_id"]).shape[0]), 12):
            errors.append("rollouts/root_pose_world must have shape (num_rollouts, 12).")
        elif not np.isfinite(root_pose_world).all():
            errors.append("rollouts/root_pose_world contains non-finite values.")
    q_target_row_id = np.asarray(root["q_h/target_row_id"])
    step_rollout_row_id = np.asarray(root["steps/rollout_row_id"])
    rollout_row_id = np.asarray(root["rollouts/rollout_row_id"])
    target_by_rollout = {
        int(row_id): int(target) for row_id, target in zip(rollout_row_id, rollout_target_row_id, strict=True)
    }
    expected_q_target = np.asarray(
        [target_by_rollout.get(int(row_id), -1) for row_id in step_rollout_row_id], dtype=np.int64
    )
    if q_target_row_id.shape == expected_q_target.shape and not np.array_equal(q_target_row_id, expected_q_target):
        errors.append("Q_H target_row_id does not match the parent rollout target_row_id.")
    elif q_target_row_id.shape != expected_q_target.shape:
        errors.append("Q_H target_row_id shape does not match the steps table.")

    if "synthetic" not in str(root.attrs.get("target_protocol_version", "")).lower():
        for attr_name in ("source_offline_store_version", "split_manifest_hash", "target_protocol_version"):
            if _missing_lineage_token(root.attrs.get(attr_name)):
                errors.append(f"Non-synthetic rollout store is missing required root attr {attr_name!r}.")
        required_lineage = (
            "candidate_config_id",
            "oracle_config_id",
            "rollout_config_id",
            "source_offline_store_manifest_hash_id",
            "split_manifest_hash_id",
            "target_protocol_version_id",
            "reason_code_version_id",
        )
        for name in required_lineage:
            if name not in root["lineage"] or np.any(np.asarray(root[f"lineage/{name}"]) < 0):
                errors.append(f"Non-synthetic rollout store is missing required lineage field {name!r}.")
        for name in (
            "candidate_config_id",
            "oracle_config_id",
            "rollout_config_id",
            "source_offline_store_manifest_hash_id",
        ):
            values = _encoded_values(root, dictionary_name="config", array_path=f"lineage/{name}")
            if any(_missing_lineage_token(value) for value in values):
                errors.append(f"Non-synthetic rollout store has synthetic or empty lineage field {name!r}.")
        expected_config_values = {
            "split_manifest_hash_id": str(root.attrs.get("split_manifest_hash", "")),
            "target_protocol_version_id": str(root.attrs.get("target_protocol_version", "")),
            "reason_code_version_id": str(root.attrs.get("reason_code_version", "")),
        }
        for name, expected in expected_config_values.items():
            values = _encoded_values(root, dictionary_name="config", array_path=f"lineage/{name}")
            if any(value != expected for value in values):
                errors.append(f"Non-synthetic rollout store lineage field {name!r} does not match root metadata.")

    for name, array in root["candidates"].arrays():
        if int(array.shape[0]) != int(candidate_row_id.shape[0]):
            errors.append(
                f"Candidate table field {name!r} has {array.shape[0]} rows, expected {candidate_row_id.shape[0]}."
            )

    return RolloutZarrValidationResult(
        store_dir=Path(store_dir).expanduser().resolve(),
        num_rollouts=int(np.asarray(root["rollouts/rollout_row_id"]).shape[0]),
        num_steps=int(np.asarray(root["steps/step_row_id"]).shape[0]),
        num_candidates=int(candidate_row_id.shape[0]),
        errors=errors,
    )


def _required_groups() -> tuple[str, ...]:
    return (
        "metadata",
        "dictionaries",
        "splits",
        "lineage",
        "mesh_refs",
        "targets",
        "rollouts",
        "steps",
        "candidates",
        "q_h",
        "diagnostics",
    )


def _write_root_metadata(
    root: Any,
    *,
    traces: list[RolloutTrace],
    return_semantics: str,
    discount_gamma: float,
    target_protocol_version: str,
    reason_code_version: str,
    field_retention_policy: str,
    source_offline_store_version: str,
    split_manifest_hash: str,
) -> None:
    root.attrs.update(
        {
            "schema_id": ROLLOUT_ZARR_SCHEMA_ID,
            "schema_version": ROLLOUT_ZARR_SCHEMA_VERSION,
            "zarr_format": 3,
            "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source_offline_store_version": source_offline_store_version,
            "split_manifest_hash": split_manifest_hash,
            "reason_code_version": reason_code_version,
            "target_protocol_version": target_protocol_version,
            "return_semantics": return_semantics,
            "discount_gamma": float(discount_gamma),
            "field_retention_policy": field_retention_policy,
            "num_rollouts": len(traces),
        }
    )


def _write_metadata_group(group: Any, *, field_retention_policy: str) -> None:
    reason_names = [name for name, _bit in sorted(INVALID_REASON_CODES.items(), key=lambda item: item[1])]
    reason_bits = [bit for _name, bit in sorted(INVALID_REASON_CODES.items(), key=lambda item: item[1])]
    _write_array(group, "reason_code_bits", np.asarray(reason_bits, dtype=np.uint16))
    _write_string_array(group, "reason_code_names", reason_names)
    manifest = {
        "schema_id": ROLLOUT_ZARR_SCHEMA_ID,
        "schema_version": ROLLOUT_ZARR_SCHEMA_VERSION,
        "field_retention_policy": field_retention_policy,
        "dense_all_action_oracle_q_materialized": False,
    }
    _write_array(group, "generation_manifest_json", np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8))
    _write_string_array(group, "field_retention_policy", [field_retention_policy])


def _build_dictionaries(traces: list[RolloutTrace]) -> dict[str, list[str]]:
    policy_values = {trace.selection_policy for trace in traces}
    policy_values.update(step.selection_policy for trace in traces for step in trace.steps)
    target_values = {trace.lineage.target_id or "synthetic-target" for trace in traces}
    score_source_values = {step.score_source or step.selection_score_label for trace in traces for step in trace.steps}
    split_values = {trace.lineage.split or "synthetic" for trace in traces}
    return {
        "scene": sorted({trace.lineage.scene_id or "" for trace in traces}),
        "snippet": sorted({trace.lineage.snippet_id or "" for trace in traces}),
        "rollout": [trace.lineage.rollout_id for trace in traces],
        "target": sorted(target_values),
        "policy": sorted(policy_values),
        "score_source": sorted(score_source_values),
        "split": sorted(split_values),
        "config": sorted(
            {
                value
                for trace in traces
                for value in (
                    trace.lineage.candidate_config_hash,
                    trace.lineage.oracle_config_hash,
                    trace.lineage.rollout_config_hash,
                    trace.lineage.model_checkpoint_hash,
                    trace.lineage.mesh_version,
                    trace.lineage.source_cache_version,
                    trace.lineage.source_offline_store_manifest_hash,
                    trace.lineage.split_manifest_hash,
                    trace.lineage.branch_schedule_id,
                    trace.lineage.target_protocol_version,
                    trace.lineage.reason_code_version,
                    trace.lineage.selection_rng_state_hash,
                )
                if value
            }
        ),
        "class_name": ["unknown"],
        "termination_reason": sorted({trace.termination_reason for trace in traces}),
        "transition": [step.transition_id or "" for trace in traces for step in trace.steps],
    }


def _write_dictionaries(group: Any, dictionaries: dict[str, list[str]]) -> None:
    for name, values in dictionaries.items():
        _write_string_array(group, name, values)


def _write_targets(
    group: Any,
    traces: list[RolloutTrace],
    dictionaries: dict[str, list[str]],
    *,
    target_protocol_version: str,
) -> None:
    target_name_by_row = {
        trace.lineage.target_row_id if trace.lineage.target_row_id is not None else 0: trace.lineage.target_id
        or "synthetic-target"
        for trace in traces
    }
    target_ids = sorted(target_name_by_row)
    if not target_ids:
        target_ids = [0]
    _write_array(group, "target_row_id", np.asarray(target_ids, dtype=np.int64))
    _write_array(
        group,
        "target_id",
        np.asarray(
            [
                _dict_id(dictionaries["target"], target_name_by_row.get(target_row_id, "synthetic-target"))
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(group, "target_valid_mask", np.ones((len(target_ids),), dtype=np.bool_))
    _write_array(
        group,
        "target_invalid_reason_bitset",
        np.full((len(target_ids),), 1 << INVALID_REASON_CODES["VALID"], dtype=np.uint32),
    )
    _write_string_array(group, "target_protocol_version", [target_protocol_version])
    _write_array(group, "crop_vertices", np.empty((0, 3), dtype=np.float32))
    _write_array(group, "crop_vertex_offsets", np.asarray([0], dtype=np.int64))
    _write_array(group, "crop_faces", np.empty((0, 3), dtype=np.int32))
    _write_array(group, "crop_face_offsets", np.asarray([0], dtype=np.int64))


def _write_mesh_refs(group: Any) -> None:
    _write_array(group, "scene_mesh_row_id", np.empty((0,), dtype=np.int64))
    _write_string_array(group, "mesh_uri", [])
    _write_string_array(group, "mesh_sha256", [])


def _flatten_traces(traces: list[RolloutTrace], dictionaries: dict[str, list[str]]) -> dict[str, np.ndarray]:
    rollout_rows: dict[str, list[Any]] = {
        "rollout_row_id": [],
        "rollout_id": [],
        "chain_id": [],
        "root_pose_world": [],
        "scene_id": [],
        "snippet_id": [],
        "target_row_id": [],
        "policy_id": [],
        "horizon": [],
        "branch_factor": [],
        "beam_width": [],
        "temperature": [],
        "random_seed": [],
        "termination_reason": [],
        "final_cumulative_target_rri": [],
        "final_cumulative_scene_rri": [],
        "split_id": [],
        "candidate_config_id": [],
        "oracle_config_id": [],
        "rollout_config_id": [],
        "model_checkpoint_id": [],
        "mesh_version_id": [],
        "source_cache_version_id": [],
        "source_offline_store_manifest_hash_id": [],
        "split_manifest_hash_id": [],
        "branch_schedule_id": [],
        "target_protocol_version_id": [],
        "reason_code_version_id": [],
        "selection_rng_state_hash_id": [],
    }
    step_rows: dict[str, list[Any]] = {
        "step_row_id": [],
        "rollout_row_id": [],
        "step_index": [],
        "selected_candidate_row_id": [],
        "selected_shell_index": [],
        "selected_compact_valid_index": [],
        "num_candidates": [],
        "num_valid_candidates": [],
        "cumulative_target_rri": [],
        "cumulative_scene_rri": [],
        "transition_id": [],
    }
    candidate_rows: dict[str, list[Any]] = _empty_candidate_rows()

    candidate_row_id = 0
    step_row_id = 0
    for rollout_row_id, trace in enumerate(traces):
        rollout_rows["rollout_row_id"].append(rollout_row_id)
        rollout_rows["rollout_id"].append(_dict_id(dictionaries["rollout"], trace.lineage.rollout_id))
        rollout_rows["chain_id"].append(trace.lineage.chain_id)
        rollout_rows["root_pose_world"].append(
            trace.root_pose_world.detach().cpu().to(dtype=torch.float32).reshape(-1).numpy()
        )
        rollout_rows["scene_id"].append(_dict_id(dictionaries["scene"], trace.lineage.scene_id or ""))
        rollout_rows["snippet_id"].append(_dict_id(dictionaries["snippet"], trace.lineage.snippet_id or ""))
        rollout_rows["target_row_id"].append(
            trace.lineage.target_row_id if trace.lineage.target_row_id is not None else 0
        )
        rollout_rows["policy_id"].append(_dict_id(dictionaries["policy"], trace.selection_policy))
        rollout_rows["horizon"].append(trace.horizon)
        rollout_rows["branch_factor"].append(trace.branch_factor)
        rollout_rows["beam_width"].append(-1 if trace.beam_width is None else trace.beam_width)
        rollout_rows["temperature"].append(_first_temperature(trace))
        rollout_rows["random_seed"].append(-1 if trace.lineage.random_seed is None else trace.lineage.random_seed)
        rollout_rows["termination_reason"].append(
            _dict_id(dictionaries["termination_reason"], trace.termination_reason)
        )
        rollout_rows["final_cumulative_target_rri"].append(_nan_if_none(trace.final_cumulative_rri))
        rollout_rows["final_cumulative_scene_rri"].append(_nan_if_none(trace.final_cumulative_rri))
        rollout_rows["split_id"].append(_dict_id(dictionaries["split"], trace.lineage.split or "synthetic"))
        rollout_rows["candidate_config_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.candidate_config_hash or "")
        )
        rollout_rows["oracle_config_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.oracle_config_hash or "")
        )
        rollout_rows["rollout_config_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.rollout_config_hash or "")
        )
        rollout_rows["model_checkpoint_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.model_checkpoint_hash or "")
        )
        rollout_rows["mesh_version_id"].append(_dict_id(dictionaries["config"], trace.lineage.mesh_version or ""))
        rollout_rows["source_cache_version_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.source_cache_version or "")
        )
        rollout_rows["source_offline_store_manifest_hash_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.source_offline_store_manifest_hash or "")
        )
        rollout_rows["split_manifest_hash_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.split_manifest_hash or "")
        )
        rollout_rows["branch_schedule_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.branch_schedule_id or "")
        )
        rollout_rows["target_protocol_version_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.target_protocol_version or "")
        )
        rollout_rows["reason_code_version_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.reason_code_version or "")
        )
        rollout_rows["selection_rng_state_hash_id"].append(
            _dict_id(dictionaries["config"], trace.lineage.selection_rng_state_hash or "")
        )

        for step in trace.steps:
            this_step_row_id = step_row_id
            step_row_id += 1
            selected_candidate_row_id = candidate_row_id + int(step.selected_shell_index)
            step_rows["step_row_id"].append(this_step_row_id)
            step_rows["rollout_row_id"].append(rollout_row_id)
            step_rows["step_index"].append(step.step_index)
            step_rows["selected_candidate_row_id"].append(selected_candidate_row_id)
            step_rows["selected_shell_index"].append(step.selected_shell_index)
            step_rows["selected_compact_valid_index"].append(step.selected_valid_index)
            step_rows["num_candidates"].append(int(step.candidate_valid.shape[0]))
            step_rows["num_valid_candidates"].append(int(step.candidate_valid.sum().item()))
            step_rows["cumulative_target_rri"].append(_nan_if_none(step.cumulative_rri))
            step_rows["cumulative_scene_rri"].append(_nan_if_none(step.cumulative_rri))
            step_rows["transition_id"].append(_dict_id(dictionaries["transition"], step.transition_id or ""))

            root_pose = trace.root_pose_world.reshape(-1)
            for shell_index in range(int(step.candidate_valid.shape[0])):
                _append_candidate_row(
                    candidate_rows,
                    step=step,
                    candidate_row_id=candidate_row_id,
                    step_row_id=this_step_row_id,
                    rollout_row_id=rollout_row_id,
                    shell_index=shell_index,
                    root_pose=root_pose,
                    dictionaries=dictionaries,
                )
                candidate_row_id += 1

    table = {
        **_to_numpy_table("rollout_", rollout_rows),
        **_to_numpy_table("step_", step_rows),
        **_candidate_rows_to_numpy(candidate_rows),
    }
    return table


def _empty_candidate_rows() -> dict[str, list[Any]]:
    return {
        "candidate_row_id": [],
        "step_row_id": [],
        "rollout_row_id": [],
        "step_index": [],
        "shell_index": [],
        "compact_valid_index": [],
        "pose_world_cam": [],
        "pose_relative_root": [],
        "candidate_valid_mask": [],
        "actor_action_mask": [],
        "oracle_label_mask": [],
        "q_train_mask": [],
        "padded_mask": [],
        "selected_mask": [],
        "heavy_diag_available_mask": [],
        "strategy_id": [],
        "mixture_id": [],
        "sampler_probability": [],
        "score_source_id": [],
        "invalid_reason_bitset": [],
        "primary_invalid_reason": [],
        "scene_rri": [],
        "target_rri": [],
        "selection_logits": [],
        "selection_probabilities": [],
        "selection_log_probabilities": [],
        "selection_entropy": [],
    }


def _append_candidate_row(
    rows: dict[str, list[Any]],
    *,
    step: Any,
    candidate_row_id: int,
    step_row_id: int,
    rollout_row_id: int,
    shell_index: int,
    root_pose: Any,
    dictionaries: dict[str, list[str]],
) -> None:
    is_valid = bool(step.candidate_valid[shell_index].item())
    is_selected = int(step.selected_shell_index) == int(shell_index)
    target_rri = _metric_value(step, ("target_rri", "oracle_target_rri"), shell_index)
    scene_rri = _metric_value(step, ("scene_rri", "oracle_scene_rri", "rri"), shell_index)
    if not is_valid:
        target_rri = float("nan")
        scene_rri = float("nan")
    oracle_label = bool(is_valid and np.isfinite(target_rri))
    q_train = bool(is_valid and oracle_label)
    pose = step.candidate_poses_world_cam[shell_index].detach().cpu().numpy().astype(np.float32)
    rows["candidate_row_id"].append(candidate_row_id)
    rows["step_row_id"].append(step_row_id)
    rows["rollout_row_id"].append(rollout_row_id)
    rows["step_index"].append(step.step_index)
    rows["shell_index"].append(shell_index)
    rows["compact_valid_index"].append(_compact_valid_index(step, shell_index))
    rows["pose_world_cam"].append(pose)
    rows["pose_relative_root"].append(_relative_pose_to_root(pose_world_cam=pose, root_pose_world=root_pose))
    rows["candidate_valid_mask"].append(is_valid)
    rows["actor_action_mask"].append(is_valid)
    rows["oracle_label_mask"].append(oracle_label)
    rows["q_train_mask"].append(q_train)
    rows["padded_mask"].append(False)
    rows["selected_mask"].append(is_selected)
    rows["heavy_diag_available_mask"].append(bool(is_selected and step.selected_point_cloud_world is not None))
    rows["strategy_id"].append(-1)
    rows["mixture_id"].append(-1)
    rows["sampler_probability"].append(float("nan"))
    rows["score_source_id"].append(
        _dict_id(dictionaries["score_source"], step.score_source or step.selection_score_label)
    )
    rows["invalid_reason_bitset"].append(_array_value(step.candidate_invalid_reason_bitset, shell_index, default=0))
    rows["primary_invalid_reason"].append(_array_value(step.candidate_primary_invalid_reason, shell_index, default=0))
    rows["scene_rri"].append(scene_rri)
    rows["target_rri"].append(target_rri)
    rows["selection_logits"].append(_array_value(step.selection_logits, shell_index, default=np.nan))
    rows["selection_probabilities"].append(_array_value(step.selection_probabilities, shell_index, default=0.0))
    rows["selection_log_probabilities"].append(
        _array_value(step.selection_log_probabilities, shell_index, default=-np.inf)
    )
    rows["selection_entropy"].append(_nan_if_none(step.selection_entropy))


def _write_rollout_tables(groups: dict[str, Any], table: dict[str, np.ndarray]) -> None:
    for name, values in table.items():
        if name.startswith("rollout_"):
            _write_array(groups["rollouts"], name.removeprefix("rollout_"), values)
            _write_array(groups["lineage"], name.removeprefix("rollout_"), values)
        elif name.startswith("step_"):
            _write_array(groups["steps"], name.removeprefix("step_"), values)
        elif name.startswith("candidate_"):
            _write_array(groups["candidates"], name.removeprefix("candidate_"), values)
    _write_array(groups["splits"], "rollout_row_id", table["rollout_rollout_row_id"])
    _write_array(groups["splits"], "rollout_split_id", table["rollout_split_id"].astype(np.int16))


def _build_q_h_arrays(table: dict[str, np.ndarray], *, horizon: int, gamma: float) -> dict[str, np.ndarray]:
    step_ids = table["step_step_row_id"].astype(np.int64)
    candidate_step_ids = table["candidate_step_row_id"].astype(np.int64)
    max_candidates = 0
    for step_id in step_ids:
        max_candidates = max(max_candidates, int((candidate_step_ids == step_id).sum()))
    state_count = int(step_ids.shape[0])
    h_max = max(int(horizon), 1)

    q = {
        "state_step_row_id": step_ids,
        "candidate_row_id": np.full((state_count, max_candidates), -1, dtype=np.int64),
        "valid_action_mask": np.zeros((state_count, max_candidates), dtype=np.bool_),
        "q_train_mask": np.zeros((state_count, max_candidates), dtype=np.bool_),
        "target_row_id": np.zeros((state_count,), dtype=np.int64),
        "selected_candidate_index": np.full((state_count,), -1, dtype=np.int32),
        "one_step_target_rri": np.full((state_count, max_candidates), np.nan, dtype=np.float32),
        "one_step_scene_rri": np.full((state_count, max_candidates), np.nan, dtype=np.float32),
        "bootstrap_next_step_row_id": np.full((state_count, max_candidates), -1, dtype=np.int64),
        "terminal_mask": np.ones((state_count, max_candidates), dtype=np.bool_),
        "invalid_reason_bitset": np.zeros((state_count, max_candidates), dtype=np.uint32),
        "q_target_target_rri": np.full((state_count, h_max, max_candidates), np.nan, dtype=np.float32),
        "q_target_scene_rri": np.full((state_count, h_max, max_candidates), np.nan, dtype=np.float32),
        "q_target_available_mask": np.zeros((state_count, h_max, max_candidates), dtype=np.bool_),
        "discount": np.asarray([float(gamma) ** power for power in range(h_max)], dtype=np.float32),
        "td_selected_candidate_row_id": np.full((state_count,), -1, dtype=np.int64),
        "td_reward_target_rri": np.full((state_count,), np.nan, dtype=np.float32),
        "td_next_step_row_id": np.full((state_count,), -1, dtype=np.int64),
        "td_terminal_mask": np.ones((state_count,), dtype=np.bool_),
        "td_discount": np.full((state_count,), float(gamma), dtype=np.float32),
    }

    next_step_by_rollout: dict[tuple[int, int], int] = {}
    for row, rollout_id in enumerate(table["step_rollout_row_id"]):
        step_index = int(table["step_step_index"][row])
        next_step_by_rollout[(int(rollout_id), step_index)] = int(table["step_step_row_id"][row])

    for row, step_id in enumerate(step_ids):
        indices = np.nonzero(candidate_step_ids == step_id)[0]
        selected_candidate_row_id = int(table["step_selected_candidate_row_id"][row])
        rollout_id = int(table["step_rollout_row_id"][row])
        rollout_matches = np.nonzero(table["rollout_rollout_row_id"] == rollout_id)[0]
        if rollout_matches.size == 1:
            q["target_row_id"][row] = int(table["rollout_target_row_id"][int(rollout_matches[0])])
        selected_local_index = -1
        for local_index, candidate_index in enumerate(indices):
            q["candidate_row_id"][row, local_index] = int(table["candidate_candidate_row_id"][candidate_index])
            q["valid_action_mask"][row, local_index] = bool(table["candidate_actor_action_mask"][candidate_index])
            q["q_train_mask"][row, local_index] = bool(table["candidate_q_train_mask"][candidate_index])
            q["one_step_target_rri"][row, local_index] = float(table["candidate_target_rri"][candidate_index])
            q["one_step_scene_rri"][row, local_index] = float(table["candidate_scene_rri"][candidate_index])
            q["invalid_reason_bitset"][row, local_index] = int(
                table["candidate_invalid_reason_bitset"][candidate_index]
            )
            if int(table["candidate_candidate_row_id"][candidate_index]) == selected_candidate_row_id:
                selected_local_index = local_index

        q["selected_candidate_index"][row] = selected_local_index
        q["td_selected_candidate_row_id"][row] = selected_candidate_row_id
        if selected_local_index >= 0:
            q["td_reward_target_rri"][row] = q["one_step_target_rri"][row, selected_local_index]
            next_step = next_step_by_rollout.get((rollout_id, int(table["step_step_index"][row]) + 1), -1)
            q["td_next_step_row_id"][row] = next_step
            q["td_terminal_mask"][row] = next_step < 0
            if next_step >= 0:
                q["terminal_mask"][row, selected_local_index] = False
                q["bootstrap_next_step_row_id"][row, selected_local_index] = next_step

    q["q_train_mask"] &= q["valid_action_mask"]
    q["one_step_target_rri"][~q["valid_action_mask"]] = np.nan
    q["one_step_scene_rri"][~q["valid_action_mask"]] = np.nan
    return q


def _write_q_h(group: Any, q_h: dict[str, np.ndarray], *, return_semantics: str, target_protocol_version: str) -> None:
    for name, values in q_h.items():
        array = _write_array(group, name, values)
        if name.startswith("q_target") or name in {"discount", "td_reward_target_rri"}:
            array.attrs.update(
                {
                    "return_semantics": return_semantics,
                    "target_protocol_version": target_protocol_version,
                    "dense_all_action_oracle_q_materialized": False,
                }
            )


def _candidate_rows_to_numpy(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    dtypes = {
        "candidate_row_id": np.int64,
        "step_row_id": np.int64,
        "rollout_row_id": np.int64,
        "step_index": np.int16,
        "shell_index": np.int32,
        "compact_valid_index": np.int32,
        "pose_world_cam": np.float32,
        "pose_relative_root": np.float32,
        "candidate_valid_mask": np.bool_,
        "actor_action_mask": np.bool_,
        "oracle_label_mask": np.bool_,
        "q_train_mask": np.bool_,
        "padded_mask": np.bool_,
        "selected_mask": np.bool_,
        "heavy_diag_available_mask": np.bool_,
        "strategy_id": np.int32,
        "mixture_id": np.int32,
        "sampler_probability": np.float32,
        "score_source_id": np.int32,
        "invalid_reason_bitset": np.uint32,
        "primary_invalid_reason": np.uint16,
        "scene_rri": np.float32,
        "target_rri": np.float32,
        "selection_logits": np.float32,
        "selection_probabilities": np.float32,
        "selection_log_probabilities": np.float32,
        "selection_entropy": np.float32,
    }
    return {f"candidate_{name}": np.asarray(values, dtype=dtypes[name]) for name, values in rows.items()}


def _to_numpy_table(prefix: str, rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, values in rows.items():
        dtype: Any
        if name.endswith("_id") or name in {"chain_id", "horizon", "branch_factor", "beam_width", "random_seed"}:
            dtype = np.int64
        elif name in {
            "step_index",
            "selected_shell_index",
            "selected_compact_valid_index",
            "num_candidates",
            "num_valid_candidates",
        }:
            dtype = np.int32
        else:
            dtype = np.float32
        arrays[f"{prefix}{name}"] = np.asarray(values, dtype=dtype)
    return arrays


def _write_array(group: Any, name: str, values: np.ndarray) -> Any:
    array = np.asarray(values)
    chunks = _default_chunks(array)
    zarr_array = group.create_array(name, shape=array.shape, chunks=chunks, dtype=array.dtype, overwrite=True)
    zarr_array[...] = array
    return zarr_array


def _write_string_array(group: Any, name: str, values: list[str]) -> None:
    encoded = np.frombuffer(json.dumps(values, ensure_ascii=True).encode("utf-8"), dtype=np.uint8)
    _write_array(group, name, encoded)


def _default_chunks(array: np.ndarray) -> tuple[int, ...] | None:
    if array.ndim == 0:
        return None
    if array.ndim == 1:
        return (min(max(int(array.shape[0]), 1), 1024),)
    return (1, *array.shape[1:])


def _dict_id(values: list[str], value: str) -> int:
    try:
        return values.index(value)
    except ValueError:
        return -1


def _first_temperature(trace: RolloutTrace) -> float:
    for step in trace.steps:
        if step.selection_temperature is not None:
            return float(step.selection_temperature)
    return float("nan")


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _compact_valid_index(step: Any, shell_index: int) -> int:
    if not bool(step.candidate_valid[shell_index].item()):
        return -1
    valid_indices = np.nonzero(step.candidate_valid.detach().cpu().numpy().astype(bool))[0]
    matches = np.nonzero(valid_indices == int(shell_index))[0]
    if matches.size != 1:
        return -1
    return int(matches[0])


def _metric_value(step: Any, metric_names: tuple[str, ...], shell_index: int) -> float:
    for metric_name in metric_names:
        values = step.metric_vectors.get(metric_name)
        if values is not None:
            return float(values[shell_index].detach().cpu().item())
    return float("nan")


def _relative_pose_to_root(*, pose_world_cam: np.ndarray, root_pose_world: torch.Tensor) -> np.ndarray:
    root = PoseTW(root_pose_world.detach().cpu().to(dtype=torch.float32).reshape(-1))
    candidate = PoseTW(torch.as_tensor(pose_world_cam, dtype=torch.float32).reshape(-1))
    return root.inverse().compose(candidate).tensor().detach().cpu().numpy().astype(np.float32).reshape(-1)


def _missing_lineage_token(value: Any) -> bool:
    return value is None or str(value) in {"", "synthetic"}


def _encoded_values(root: Any, *, dictionary_name: str, array_path: str) -> list[str]:
    try:
        encoded = np.asarray(root[array_path])
    except KeyError:
        return []
    dictionary = _read_string_array(root, f"dictionaries/{dictionary_name}")
    values: list[str] = []
    for index in encoded.reshape(-1):
        index_int = int(index)
        if index_int < 0 or index_int >= len(dictionary):
            values.append("")
        else:
            values.append(dictionary[index_int])
    return values


def _read_string_array(root: Any, path: str) -> list[str]:
    try:
        encoded = np.asarray(root[path])
    except KeyError:
        return []
    return json.loads(encoded.tobytes().decode("utf-8"))


def _array_value(values: Any | None, index: int, *, default: float | int) -> float | int:
    if values is None:
        return default
    return values[index].detach().cpu().item()


__all__ = [
    "DEFAULT_RETURN_SEMANTICS",
    "ROLLOUT_ZARR_SCHEMA_ID",
    "ROLLOUT_ZARR_SCHEMA_VERSION",
    "RolloutZarrStoreConfig",
    "RolloutZarrStoreReader",
    "RolloutZarrStoreWriter",
    "RolloutZarrValidationResult",
    "RolloutZarrWriteResult",
    "validate_rollout_zarr_store",
    "write_rollout_zarr_store",
]
