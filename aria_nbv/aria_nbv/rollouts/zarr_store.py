"""Standalone Zarr replay store for finite-candidate rollout traces.

This module is the implementation-contract owner for `rollouts.zarr`. A store
contains compact row tables for rollout chains, steps, full-shell candidates,
shared VIN source rows, lineage, target records, masks, and reason codes. The
padded `Q_H` tensors used by finite-candidate value learning are derived by the
reader from factual `steps/` and `candidates/` tables. The store deliberately
does not mutate or migrate the strict VIN offline store; rollout replay is a
separate artifact with source manifest, split, target, candidate-mixture,
policy, and oracle config hashes.

`q_train_mask` is true only when a row is non-padded, actor-selectable,
target-valid, GT-label-valid, and has a finite target-RRI label. Invalid
candidates keep their full-shell row but carry false masks and NaN labels.
Target RRI and scene RRI are distinct diagnostics; target labels must not be
silently filled from scene scores or low-quality invalid rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import zarr
from efm3d.aria.pose import PoseTW
from pydantic import Field, field_validator
from zarr.storage import LocalStore

from ..configs import PathConfig
from ..utils import BaseConfig
from ..utils.config_paths import resolve_cache_artifact_dir
from .manifest import (
    ROLLOUT_MANIFEST_FILENAME,
    ROLLOUT_MANIFEST_VERSION,
    RolloutStoreManifestContext,
    manifest_sha256,
    read_rollout_store_manifest,
    utc_timestamp,
    write_rollout_store_manifest,
)
from .trace import (
    INVALID_REASON_CODES,
    INVALID_REASON_VERSION,
    RolloutLineage,
    RolloutZarrRecord,
    _candidate_invalid_reasons,
    _full_shell_or_default,
    _policy_name,
    _termination_reason,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..pose_generation.counterfactuals import CounterfactualStepResult, CounterfactualTrajectory

ROLLOUT_ZARR_SCHEMA_ID = "aria_nbv.rollout_zarr_q_invalidity"
"""Schema id stored as a root attribute on rollout replay stores."""

ROLLOUT_ZARR_SCHEMA_VERSION = "0.4-manifested-shards"
"""Manifest-backed clean table-owner rollout replay schema version."""

DEFAULT_RETURN_SEMANTICS = "cumulative_target_rri"
"""Default return target family for initial ``Q_H`` replay views."""


@dataclass(frozen=True, slots=True)
class _TableField:
    """One fixed-width Zarr table field."""

    name: str
    dtype: Any


@dataclass(frozen=True, slots=True)
class _TableSchema:
    """Compact schema owner for one fixed-width Zarr table."""

    name: str
    fields: tuple[_TableField, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Return field names in write order."""

        return tuple(field.name for field in self.fields)

    @property
    def dtypes(self) -> dict[str, Any]:
        """Return the field dtype map used for NumPy materialization."""

        return {field.name: field.dtype for field in self.fields}


SOURCE_TABLE = _TableSchema(
    "sources",
    (
        _TableField("source_row_id", np.int64),
        _TableField("sample_index", np.int64),
        _TableField("sample_key_id", np.int32),
        _TableField("scene_id", np.int32),
        _TableField("snippet_id", np.int32),
        _TableField("split_id", np.int32),
        _TableField("source_cache_version_id", np.int32),
        _TableField("source_offline_store_manifest_hash_id", np.int32),
        _TableField("split_manifest_hash_id", np.int32),
        _TableField("source_shard_id", np.int32),
        _TableField("source_shard_row", np.int64),
    ),
)
"""Canonical `sources/` table schema."""

ROLLOUT_TABLE = _TableSchema(
    "rollouts",
    (
        _TableField("rollout_row_id", np.int64),
        _TableField("rollout_id", np.int32),
        _TableField("chain_id", np.int32),
        _TableField("source_row_id", np.int64),
        _TableField("root_pose_world", np.float32),
        _TableField("scene_id", np.int32),
        _TableField("snippet_id", np.int32),
        _TableField("target_row_id", np.int64),
        _TableField("policy_id", np.int32),
        _TableField("horizon", np.int16),
        _TableField("branch_factor", np.int16),
        _TableField("beam_width", np.int16),
        _TableField("temperature", np.float32),
        _TableField("random_seed", np.int64),
        _TableField("termination_reason", np.int32),
        _TableField("final_cumulative_target_rri", np.float32),
        _TableField("final_cumulative_scene_rri", np.float32),
        _TableField("split_id", np.int32),
    ),
)
"""Canonical `rollouts/` table fields and dtypes."""

LINEAGE_TABLE = _TableSchema(
    "lineage",
    (
        _TableField("rollout_row_id", np.int64),
        _TableField("candidate_config_id", np.int32),
        _TableField("oracle_config_id", np.int32),
        _TableField("rollout_config_id", np.int32),
        _TableField("model_checkpoint_id", np.int32),
        _TableField("mesh_version_id", np.int32),
        _TableField("branch_schedule_id", np.int32),
        _TableField("target_protocol_version_id", np.int32),
        _TableField("target_crop_policy_id", np.int32),
        _TableField("reason_code_version_id", np.int32),
        _TableField("selection_rng_state_hash_id", np.int32),
    ),
)
"""Canonical `lineage/` table fields and dtypes."""

STEP_TABLE = _TableSchema(
    "steps",
    (
        _TableField("step_row_id", np.int64),
        _TableField("rollout_row_id", np.int64),
        _TableField("step_index", np.int16),
        _TableField("selected_candidate_row_id", np.int64),
        _TableField("selected_shell_index", np.int32),
        _TableField("selected_compact_valid_index", np.int32),
        _TableField("num_candidates", np.int32),
        _TableField("num_valid_candidates", np.int32),
        _TableField("cumulative_target_rri", np.float32),
        _TableField("cumulative_scene_rri", np.float32),
        _TableField("transition_id", np.int32),
    ),
)
"""Canonical `steps/` table fields and dtypes."""

CANDIDATE_TABLE = _TableSchema(
    "candidates",
    (
        _TableField("candidate_row_id", np.int64),
        _TableField("step_row_id", np.int64),
        _TableField("rollout_row_id", np.int64),
        _TableField("step_index", np.int16),
        _TableField("shell_index", np.int32),
        _TableField("compact_valid_index", np.int32),
        _TableField("pose_world_cam", np.float32),
        _TableField("pose_relative_root", np.float32),
        _TableField("candidate_valid_mask", np.bool_),
        _TableField("actor_action_mask", np.bool_),
        _TableField("oracle_label_mask", np.bool_),
        _TableField("q_train_mask", np.bool_),
        _TableField("padded_mask", np.bool_),
        _TableField("selected_mask", np.bool_),
        _TableField("heavy_diag_available_mask", np.bool_),
        _TableField("strategy_id", np.int32),
        _TableField("mixture_id", np.int32),
        _TableField("sampler_probability", np.float32),
        _TableField("score_source_id", np.int32),
        _TableField("invalid_reason_bitset", np.uint32),
        _TableField("primary_invalid_reason", np.uint16),
        _TableField("scene_rri", np.float32),
        _TableField("target_rri", np.float32),
        _TableField("selection_logits", np.float32),
        _TableField("selection_probabilities", np.float32),
        _TableField("selection_log_probabilities", np.float32),
        _TableField("selection_entropy", np.float32),
    ),
)
"""Canonical `candidates/` table fields and dtypes."""


@dataclass(slots=True)
class RolloutZarrWriteResult:
    """Summary of one rollout Zarr write.

    Counts refer to materialized row tables, not source VIN samples. One source
    sample can contribute multiple targets, rollout recipes, beam chains,
    steps, and full-shell candidate rows.
    """

    store_dir: Path
    num_rollouts: int
    num_steps: int
    num_candidates: int
    manifest_path: Path
    manifest_sha256: str


@dataclass(slots=True)
class RolloutZarrValidationResult:
    """Validation summary for a rollout Zarr store.

    `errors` contains schema, linkage, mask, and lineage violations. Validation
    fails if candidate strategy ids, mixture ids, target protocol metadata,
    source hashes, or explicit target-RRI labels are missing.
    """

    store_dir: Path
    num_rollouts: int
    num_steps: int
    num_candidates: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return ``True`` when no validation errors were found."""

        return not self.errors


@dataclass(slots=True)
class _RolloutTables:
    """Materialized rollout store row tables before Zarr persistence."""

    sources: dict[str, np.ndarray]
    rollouts: dict[str, np.ndarray]
    lineage: dict[str, np.ndarray]
    steps: dict[str, np.ndarray]
    candidates: dict[str, np.ndarray]


class RolloutZarrStoreConfig(BaseConfig):
    """Filesystem and target metadata for one standalone rollout replay store."""

    paths: PathConfig = Field(default_factory=PathConfig)
    store_dir: Path = Field(default_factory=lambda: PathConfig().offline_cache_dir / "rollouts.zarr")
    return_semantics: str = DEFAULT_RETURN_SEMANTICS
    discount_gamma: float = Field(default=1.0, ge=0.0)
    target_protocol_version: str = "v1-observed"
    reason_code_version: str = INVALID_REASON_VERSION
    field_retention_policy: str = "compact"
    source_offline_store_version: str = "unknown-source-version"
    split_manifest_hash: str = "unknown-split-manifest"

    _resolve_store_dir = field_validator("store_dir", mode="before")(resolve_cache_artifact_dir)


class RolloutZarrStoreReader:
    """Open and validate standalone rollout replay stores."""

    def __init__(self, store_dir: Path | str) -> None:
        self.store_dir = Path(store_dir).expanduser().resolve()
        self.root = zarr.open_group(store=LocalStore(str(self.store_dir), read_only=True), mode="r")

    def array(self, path: str) -> np.ndarray:
        """Read an array by slash-separated Zarr path."""

        return np.asarray(self.root[path])

    def validate(self) -> RolloutZarrValidationResult:
        """Validate row linkage, masks, and initial ``Q_H`` target availability."""

        return validate_rollout_zarr_store(self.store_dir)

    def manifest(self) -> dict[str, Any]:
        """Read root attrs and the top-level sidecar manifest without payload arrays."""

        return {
            "root_attrs": dict(self.root.attrs),
            "manifest": read_rollout_store_manifest(self.store_dir),
        }

    def q_h_view(self, *, discount_gamma: float | None = None, horizon: int | None = None) -> dict[str, np.ndarray]:
        """Build the padded finite-candidate ``Q_H`` view from factual store tables.

        The view is intentionally derived rather than persisted. This keeps the
        store schema factual while still giving training and inspection code the
        dense candidate-query tensors they need.
        """

        gamma = float(self.root.attrs.get("discount_gamma", 1.0) if discount_gamma is None else discount_gamma)
        if horizon is None:
            horizon_values = np.asarray(self.root["rollouts/horizon"])
            horizon = int(horizon_values.max()) if horizon_values.size else 1
        return _build_q_h_arrays(_read_tables_from_root(self.root), horizon=int(horizon), gamma=gamma)


def write_rollout_zarr_store(
    store_dir: Path | str,
    records: list[RolloutZarrRecord],
    *,
    return_semantics: str = DEFAULT_RETURN_SEMANTICS,
    discount_gamma: float = 1.0,
    target_protocol_version: str = "v1-observed",
    reason_code_version: str = INVALID_REASON_VERSION,
    field_retention_policy: str = "compact",
    source_offline_store_version: str = "unknown-source-version",
    split_manifest_hash: str = "unknown-split-manifest",
    manifest_context: RolloutStoreManifestContext | None = None,
) -> RolloutZarrWriteResult:
    """Write rollout records into a standalone ``rollouts.zarr`` store."""

    return _RolloutZarrWriteSession(
        store_dir=store_dir,
        records=records,
        return_semantics=return_semantics,
        discount_gamma=discount_gamma,
        target_protocol_version=target_protocol_version,
        reason_code_version=reason_code_version,
        field_retention_policy=field_retention_policy,
        source_offline_store_version=source_offline_store_version,
        split_manifest_hash=split_manifest_hash,
        manifest_context=manifest_context,
    ).write()


class _RolloutZarrWriteSession:
    """Own one write of a rollout Zarr store and its derived row tables."""

    def __init__(
        self,
        *,
        store_dir: Path | str,
        records: list[RolloutZarrRecord],
        return_semantics: str,
        discount_gamma: float,
        target_protocol_version: str,
        reason_code_version: str,
        field_retention_policy: str,
        source_offline_store_version: str,
        split_manifest_hash: str,
        manifest_context: RolloutStoreManifestContext | None,
    ) -> None:
        self.output_dir = Path(store_dir).expanduser().resolve()
        self.records = records
        self.return_semantics = return_semantics
        self.discount_gamma = float(discount_gamma)
        self.target_protocol_version = target_protocol_version
        self.reason_code_version = reason_code_version
        self.field_retention_policy = field_retention_policy
        self.source_offline_store_version = source_offline_store_version
        self.split_manifest_hash = split_manifest_hash
        self.manifest_context = manifest_context or RolloutStoreManifestContext.programmatic()

    def write(self) -> RolloutZarrWriteResult:
        """Materialize the configured rollout traces to disk."""

        created_at_utc = utc_timestamp()
        dictionaries = _build_dictionaries(self.records)
        table = _flatten_records(self.records, dictionaries)
        root_metadata = _root_metadata_payload(
            records=self.records,
            tables=table,
            return_semantics=self.return_semantics,
            discount_gamma=self.discount_gamma,
            target_protocol_version=self.target_protocol_version,
            reason_code_version=self.reason_code_version,
            field_retention_policy=self.field_retention_policy,
            source_offline_store_version=self.source_offline_store_version,
            split_manifest_hash=self.split_manifest_hash,
            created_at_utc=created_at_utc,
            manifest_sha256="",
        )
        manifest_payload = _build_manifest_payload(
            records=self.records,
            tables=table,
            dictionaries=dictionaries,
            context=self.manifest_context,
            root_attrs=root_metadata,
            created_at_utc=created_at_utc,
        )
        manifest_digest = manifest_sha256(manifest_payload)
        root_metadata["manifest_sha256"] = manifest_digest

        root = zarr.open_group(str(self.output_dir), mode="w")
        root.attrs.update(root_metadata)
        groups = {name: root.create_group(name, overwrite=True) for name in _required_groups()}

        _write_dictionaries(groups["dictionaries"], dictionaries)
        _write_metadata_group(groups["metadata"], field_retention_policy=self.field_retention_policy)
        _write_targets(
            groups["targets"],
            self.records,
            dictionaries,
            target_protocol_version=self.target_protocol_version,
        )

        _write_rollout_tables(groups, table)
        written_manifest_digest = write_rollout_store_manifest(self.output_dir, manifest_payload)
        if written_manifest_digest != manifest_digest:
            raise RuntimeError("Rollout manifest digest changed while writing.")

        return RolloutZarrWriteResult(
            store_dir=self.output_dir,
            num_rollouts=int(table.rollouts["rollout_row_id"].shape[0]),
            num_steps=int(table.steps["step_row_id"].shape[0]),
            num_candidates=int(table.candidates["candidate_row_id"].shape[0]),
            manifest_path=self.output_dir / ROLLOUT_MANIFEST_FILENAME,
            manifest_sha256=manifest_digest,
        )


def validate_rollout_zarr_store(store_dir: Path | str) -> RolloutZarrValidationResult:
    """Validate a standalone rollout replay store and return all discovered errors."""

    return _RolloutZarrValidator(store_dir).validate()


class _RolloutZarrValidator:
    """Validate one rollout store without mixing checks into the public facade."""

    def __init__(self, store_dir: Path | str) -> None:
        self.store_dir = Path(store_dir).expanduser().resolve()
        self.root = zarr.open_group(
            store=LocalStore(str(self.store_dir), read_only=True),
            mode="r",
        )
        self.errors: list[str] = []

    def validate(self) -> RolloutZarrValidationResult:
        """Validate row linkage, masks, target validity, and lineage."""

        self._validate_root_contract()
        if self.errors:
            return RolloutZarrValidationResult(self.store_dir, 0, 0, 0, self.errors)

        candidate_row_id = np.asarray(self.root["candidates/candidate_row_id"])
        self._validate_q_h(candidate_row_id)
        self._validate_candidates(candidate_row_id)
        self._validate_sources()
        self._validate_targets()
        self._validate_required_lineage()

        return RolloutZarrValidationResult(
            store_dir=self.store_dir,
            num_rollouts=int(np.asarray(self.root["rollouts/rollout_row_id"]).shape[0]),
            num_steps=int(np.asarray(self.root["steps/step_row_id"]).shape[0]),
            num_candidates=int(candidate_row_id.shape[0]),
            errors=self.errors,
        )

    def _validate_root_contract(self) -> None:
        if self.root.attrs.get("schema_version") != ROLLOUT_ZARR_SCHEMA_VERSION:
            self.errors.append(
                f"Unsupported rollout Zarr schema_version={self.root.attrs.get('schema_version')!r}; "
                f"expected {ROLLOUT_ZARR_SCHEMA_VERSION!r}."
            )
        self._validate_manifest_contract()
        for group_name in _required_groups():
            if group_name not in self.root:
                self.errors.append(f"Missing required group {group_name!r}.")

    def _validate_manifest_contract(self) -> None:
        manifest_path_attr = self.root.attrs.get("manifest_path")
        if manifest_path_attr != ROLLOUT_MANIFEST_FILENAME:
            self.errors.append(
                f"Rollout store root attr 'manifest_path' must be {ROLLOUT_MANIFEST_FILENAME!r}, "
                f"got {manifest_path_attr!r}."
            )
            return
        manifest_path = self.store_dir / ROLLOUT_MANIFEST_FILENAME
        if not manifest_path.exists():
            self.errors.append(f"Missing required top-level rollout manifest {manifest_path.name!r}.")
            return
        try:
            payload = read_rollout_store_manifest(self.store_dir)
        except (OSError, json.JSONDecodeError) as exc:
            self.errors.append(f"Failed to read rollout manifest: {exc}.")
            return
        expected_hash = self.root.attrs.get("manifest_sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            self.errors.append("Rollout store root attr 'manifest_sha256' is missing.")
        elif manifest_sha256(payload) != expected_hash:
            self.errors.append("Rollout store manifest hash does not match root attr 'manifest_sha256'.")
        if payload.get("manifest_version") != ROLLOUT_MANIFEST_VERSION:
            self.errors.append(
                f"Unsupported rollout manifest_version={payload.get('manifest_version')!r}; "
                f"expected {ROLLOUT_MANIFEST_VERSION!r}."
            )
        if payload.get("schema_version") != ROLLOUT_ZARR_SCHEMA_VERSION:
            self.errors.append("Rollout manifest schema_version does not match the current rollout Zarr schema.")

    def _validate_q_h(self, candidate_row_id: np.ndarray) -> None:
        q_h = _build_q_h_arrays(_read_tables_from_root(self.root), horizon=_stored_horizon(self.root), gamma=1.0)
        q_candidate_row_id = q_h["candidate_row_id"]
        q_train_mask = q_h["q_train_mask"]
        valid_action_mask = q_h["valid_action_mask"]
        one_step_target_rri = q_h["one_step_target_rri"]

        real_q_ids = q_candidate_row_id[q_candidate_row_id >= 0]
        if not np.isin(real_q_ids, candidate_row_id).all():
            self.errors.append("Q_H candidate_row_id contains ids not present in candidates/candidate_row_id.")
        if np.any(q_train_mask & (~valid_action_mask)):
            self.errors.append("Q_H q_train_mask is true for invalid or padded candidates.")
        if np.any(q_train_mask & (~np.isfinite(one_step_target_rri))):
            self.errors.append("Q_H q_train_mask is true without a finite explicit target-RRI label.")

    def _validate_candidates(self, candidate_row_id: np.ndarray) -> None:
        selected_mask = np.asarray(self.root["candidates/selected_mask"])
        actor_action_mask = np.asarray(self.root["candidates/actor_action_mask"])
        if np.any(selected_mask & (~actor_action_mask)):
            self.errors.append("Selected candidates must be actor-selectable.")

        rollout_split_id = np.asarray(self.root["rollouts/split_id"])
        if np.unique(rollout_split_id).shape[0] > 1:
            self.errors.append("A rollout shard must contain exactly one split.")

        for name, array in self.root["candidates"].arrays():
            if int(array.shape[0]) != int(candidate_row_id.shape[0]):
                self.errors.append(
                    f"Candidate table field {name!r} has {array.shape[0]} rows, expected {candidate_row_id.shape[0]}."
                )

    def _validate_sources(self) -> None:
        source_row_id = np.asarray(self.root["sources/source_row_id"])
        rollout_source_row_id = np.asarray(self.root["rollouts/source_row_id"])
        if not np.isin(rollout_source_row_id, source_row_id).all():
            self.errors.append("Rollout source_row_id contains ids not present in sources/source_row_id.")
        if np.unique(source_row_id).shape[0] != source_row_id.shape[0]:
            self.errors.append("sources/source_row_id must be unique within one rollout shard.")

    def _validate_targets(self) -> None:
        target_row_id = np.asarray(self.root["targets/target_row_id"])
        rollout_target_row_id = np.asarray(self.root["rollouts/target_row_id"])
        if not np.isin(rollout_target_row_id, target_row_id).all():
            self.errors.append("Rollout target_row_id contains ids not present in targets/target_row_id.")
        if "root_pose_world" not in self.root["rollouts"]:
            self.errors.append("Missing required rollout root_pose_world field.")
        else:
            root_pose_world = np.asarray(self.root["rollouts/root_pose_world"])
            if root_pose_world.shape != (int(np.asarray(self.root["rollouts/rollout_row_id"]).shape[0]), 12):
                self.errors.append("rollouts/root_pose_world must have shape (num_rollouts, 12).")
            elif not np.isfinite(root_pose_world).all():
                self.errors.append("rollouts/root_pose_world contains non-finite values.")

        q_state_target_row_id = _build_q_h_arrays(
            _read_tables_from_root(self.root), horizon=_stored_horizon(self.root), gamma=1.0
        )["target_row_id"]
        step_rollout_row_id = np.asarray(self.root["steps/rollout_row_id"])
        rollout_row_id = np.asarray(self.root["rollouts/rollout_row_id"])
        target_by_rollout = {
            int(row_id): int(target) for row_id, target in zip(rollout_row_id, rollout_target_row_id, strict=True)
        }
        expected_state_target = np.asarray(
            [target_by_rollout.get(int(row_id), -1) for row_id in step_rollout_row_id], dtype=np.int64
        )
        if q_state_target_row_id.shape == expected_state_target.shape and not np.array_equal(
            q_state_target_row_id, expected_state_target
        ):
            self.errors.append("Q_H target_row_id does not match the parent rollout target_row_id.")
        elif q_state_target_row_id.shape != expected_state_target.shape:
            self.errors.append("Q_H target_row_id shape does not match the steps table.")

    def _validate_required_lineage(self) -> None:
        target_row_id = np.asarray(self.root["targets/target_row_id"])
        q_h = _build_q_h_arrays(_read_tables_from_root(self.root), horizon=_stored_horizon(self.root), gamma=1.0)
        q_state_target_row_id = q_h["target_row_id"]
        q_train_mask = q_h["q_train_mask"]
        target_valid_by_id = {
            int(row_id): bool(valid and gt_valid)
            for row_id, valid, gt_valid in zip(
                target_row_id,
                np.asarray(self.root["targets/target_valid_mask"]),
                np.asarray(self.root["targets/gt_label_valid_mask"]),
                strict=True,
            )
        }
        q_target_valid = np.asarray([target_valid_by_id.get(int(row_id), False) for row_id in q_state_target_row_id])
        if q_train_mask.shape[0] == q_target_valid.shape[0] and np.any(q_train_mask & (~q_target_valid[:, None])):
            self.errors.append("Q_H q_train_mask is true for a target without valid actor-visible and GT label state.")
        for attr_name in ("source_offline_store_version", "split_manifest_hash", "target_protocol_version"):
            if _missing_lineage_token(self.root.attrs.get(attr_name)):
                self.errors.append(f"Rollout store is missing required root attr {attr_name!r}.")
        required_lineage = (
            "rollout_row_id",
            "candidate_config_id",
            "oracle_config_id",
            "rollout_config_id",
            "target_protocol_version_id",
            "target_crop_policy_id",
            "reason_code_version_id",
        )
        for name in required_lineage:
            if name not in self.root["lineage"] or np.any(np.asarray(self.root[f"lineage/{name}"]) < 0):
                self.errors.append(f"Rollout store is missing required lineage field {name!r}.")
        rollout_row_id = np.asarray(self.root["rollouts/rollout_row_id"])
        if "rollout_row_id" in self.root["lineage"] and not np.array_equal(
            np.asarray(self.root["lineage/rollout_row_id"]),
            rollout_row_id,
        ):
            self.errors.append("Lineage rollout_row_id must align with rollouts/rollout_row_id.")
        for name in (
            "candidate_config_id",
            "oracle_config_id",
            "rollout_config_id",
            "target_crop_policy_id",
        ):
            values = _encoded_values(self.root, dictionary_name="config", array_path=f"lineage/{name}")
            if any(_missing_lineage_token(value) for value in values):
                self.errors.append(f"Rollout store has empty lineage field {name!r}.")
        expected_config_values = {
            "target_protocol_version_id": str(self.root.attrs.get("target_protocol_version", "")),
            "reason_code_version_id": str(self.root.attrs.get("reason_code_version", "")),
        }
        for name, expected in expected_config_values.items():
            values = _encoded_values(self.root, dictionary_name="config", array_path=f"lineage/{name}")
            if any(value != expected for value in values):
                self.errors.append(f"Rollout store lineage field {name!r} does not match root metadata.")
        for name in ("source_offline_store_manifest_hash_id", "split_manifest_hash_id", "source_cache_version_id"):
            values = _encoded_values(self.root, dictionary_name="config", array_path=f"sources/{name}")
            if any(_missing_lineage_token(value) for value in values):
                self.errors.append(f"Rollout store has empty source field {name!r}.")
        actor_rows = np.asarray(self.root["candidates/actor_action_mask"])
        if np.any(actor_rows & (np.asarray(self.root["candidates/strategy_id"]) < 0)):
            self.errors.append("Actor-selectable candidates require non-placeholder strategy_id.")
        if np.any(actor_rows & (np.asarray(self.root["candidates/mixture_id"]) < 0)):
            self.errors.append("Actor-selectable candidates require non-placeholder mixture_id.")
        if np.any(actor_rows & (~np.isfinite(np.asarray(self.root["candidates/sampler_probability"])))):
            self.errors.append("Actor-selectable candidates require finite sampler_probability.")


def _required_groups() -> tuple[str, ...]:
    return (
        "metadata",
        "dictionaries",
        "sources",
        "lineage",
        "targets",
        "rollouts",
        "steps",
        "candidates",
    )


def _root_metadata_payload(
    *,
    records: list[RolloutZarrRecord],
    tables: _RolloutTables,
    return_semantics: str,
    discount_gamma: float,
    target_protocol_version: str,
    reason_code_version: str,
    field_retention_policy: str,
    source_offline_store_version: str,
    split_manifest_hash: str,
    created_at_utc: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Return compact root attrs for one rollout store."""

    split_values = {
        record.lineage_for_chain(chain_id).split or "unknown"
        for record in records
        for chain_id, _trajectory in enumerate(record.result.trajectories)
    }
    return {
        "schema_id": ROLLOUT_ZARR_SCHEMA_ID,
        "schema_version": ROLLOUT_ZARR_SCHEMA_VERSION,
        "zarr_format": 3,
        "created_at_utc": created_at_utc,
        "manifest_path": ROLLOUT_MANIFEST_FILENAME,
        "manifest_sha256": manifest_sha256,
        "manifest_version": ROLLOUT_MANIFEST_VERSION,
        "source_offline_store_version": source_offline_store_version,
        "split_manifest_hash": split_manifest_hash,
        "source_split": next(iter(split_values)) if len(split_values) == 1 else "mixed",
        "reason_code_version": reason_code_version,
        "target_protocol_version": target_protocol_version,
        "return_semantics": return_semantics,
        "discount_gamma": float(discount_gamma),
        "field_retention_policy": field_retention_policy,
        "num_sources": int(tables.sources["source_row_id"].shape[0]),
        "num_targets": int(len(_unique_targets(records))),
        "num_rollouts": int(tables.rollouts["rollout_row_id"].shape[0]),
        "num_steps": int(tables.steps["step_row_id"].shape[0]),
        "num_candidates": int(tables.candidates["candidate_row_id"].shape[0]),
    }


def _build_manifest_payload(
    *,
    records: list[RolloutZarrRecord],
    tables: _RolloutTables,
    dictionaries: dict[str, list[str]],
    context: RolloutStoreManifestContext,
    root_attrs: dict[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    """Build the human-readable top-level rollout-store manifest."""

    return {
        "manifest_version": ROLLOUT_MANIFEST_VERSION,
        "schema_id": ROLLOUT_ZARR_SCHEMA_ID,
        "schema_version": ROLLOUT_ZARR_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "store_kind": "standalone_rollout_zarr_shard",
        "root_attrs": {key: value for key, value in root_attrs.items() if key != "manifest_sha256"},
        "counts": {
            "sources": int(tables.sources["source_row_id"].shape[0]),
            "targets": int(len(_unique_targets(records))),
            "rollouts": int(tables.rollouts["rollout_row_id"].shape[0]),
            "steps": int(tables.steps["step_row_id"].shape[0]),
            "candidates": int(tables.candidates["candidate_row_id"].shape[0]),
        },
        "source_coverage": _source_coverage(records),
        "config_hashes": _manifest_config_hashes(records),
        "dictionary_sizes": {name: len(values) for name, values in sorted(dictionaries.items())},
        "generation": context.to_jsonable(),
    }


def _source_coverage(records: list[RolloutZarrRecord]) -> dict[str, Any]:
    """Summarize source rows without reading Zarr payload arrays."""

    rows: dict[int, dict[str, Any]] = {}
    for record in records:
        lineage = record.lineage
        source_row_id = -1 if lineage.source_row_id is None else int(lineage.source_row_id)
        rows[source_row_id] = {
            "source_row_id": source_row_id,
            "source_sample_index": lineage.source_sample_index,
            "source_sample_key": lineage.source_sample_key,
            "scene_id": lineage.scene_id,
            "snippet_id": lineage.snippet_id,
            "split": lineage.split,
            "source_shard_id": lineage.source_shard_id,
            "source_shard_row": lineage.source_shard_row,
        }
    scene_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for row in rows.values():
        scene = str(row["scene_id"] or "unknown")
        split = str(row["split"] or "unknown")
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1
    return {
        "num_source_rows": len(rows),
        "scene_counts": dict(sorted(scene_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "sources": [rows[key] for key in sorted(rows)],
    }


def _manifest_config_hashes(records: list[RolloutZarrRecord]) -> dict[str, list[str]]:
    """Collect unique config/protocol hashes stored in rollout lineages."""

    values: dict[str, set[str]] = {
        "candidate": set(),
        "oracle": set(),
        "rollout": set(),
        "model_checkpoint": set(),
        "source_manifest": set(),
        "split_manifest": set(),
        "target_crop_policy": set(),
        "target_protocol": set(),
    }
    for record in records:
        lineage = record.lineage
        _add_manifest_hash(values["candidate"], lineage.candidate_config_hash)
        _add_manifest_hash(values["oracle"], lineage.oracle_config_hash)
        _add_manifest_hash(values["rollout"], lineage.rollout_config_hash)
        _add_manifest_hash(values["model_checkpoint"], lineage.model_checkpoint_hash)
        _add_manifest_hash(values["source_manifest"], lineage.source_offline_store_manifest_hash)
        _add_manifest_hash(values["split_manifest"], lineage.split_manifest_hash)
        _add_manifest_hash(values["target_crop_policy"], lineage.target_crop_policy)
        _add_manifest_hash(values["target_protocol"], lineage.target_protocol_version)
    return {name: sorted(items) for name, items in values.items()}


def _add_manifest_hash(target: set[str], value: str | None) -> None:
    if value:
        target.add(value)


def _unique_targets(records: list[RolloutZarrRecord]) -> set[int]:
    """Return unique target row ids represented by rollout records."""

    return {int(record.lineage.target_row_id) for record in records if record.lineage.target_row_id is not None}


def _write_metadata_group(group: zarr.Group, *, field_retention_policy: str) -> None:
    reason_names = [name for name, _bit in sorted(INVALID_REASON_CODES.items(), key=lambda item: item[1])]
    reason_bits = [bit for _name, bit in sorted(INVALID_REASON_CODES.items(), key=lambda item: item[1])]
    _write_array(group, "reason_code_bits", np.asarray(reason_bits, dtype=np.uint16))
    _write_string_array(group, "reason_code_names", reason_names)
    _write_string_array(group, "field_retention_policy", [field_retention_policy])


def _build_dictionaries(records: list[RolloutZarrRecord]) -> dict[str, list[str]]:
    items = list(_record_items(records))
    policy_values = {_policy_name(record.result.selection_policy) for record in records}
    policy_values.update(
        step.selection_policy
        for record in records
        for trajectory in record.result.trajectories
        for step in trajectory.steps
    )
    policy_values.update(
        lineage.target_selection_policy
        for _record, _trajectory, lineage in items
        if lineage.target_selection_policy is not None
    )
    target_values = {lineage.target_id or "unknown-target" for _record, _trajectory, lineage in items}
    target_values.update(
        lineage.matched_gt_target_id
        for _record, _trajectory, lineage in items
        if lineage.matched_gt_target_id is not None
    )
    source_key_values = {lineage.source_sample_key or "" for _record, _trajectory, lineage in items}
    source_shard_values = {lineage.source_shard_id or "" for _record, _trajectory, lineage in items}
    score_source_values = {
        step.selection_score_label
        for record in records
        for trajectory in record.result.trajectories
        for step in trajectory.steps
    }
    split_values = {lineage.split or "unknown" for _record, _trajectory, lineage in items}
    target_match_status_values = {lineage.gt_match_status or "not_requested" for _record, _trajectory, lineage in items}
    return {
        "scene": sorted({lineage.scene_id or "" for _record, _trajectory, lineage in items}),
        "snippet": sorted({lineage.snippet_id or "" for _record, _trajectory, lineage in items}),
        "rollout": [lineage.rollout_id for _record, _trajectory, lineage in items],
        "target": sorted(target_values),
        "source_key": sorted(source_key_values),
        "source_shard": sorted(source_shard_values),
        "target_source": sorted({lineage.target_source or "" for _record, _trajectory, lineage in items}),
        "policy": sorted(policy_values),
        "score_source": sorted(score_source_values),
        "split": sorted(split_values),
        "config": sorted(
            {
                value
                for _record, _trajectory, lineage in items
                for value in (
                    lineage.candidate_config_hash,
                    lineage.oracle_config_hash,
                    lineage.rollout_config_hash,
                    lineage.model_checkpoint_hash,
                    lineage.mesh_version,
                    lineage.source_cache_version,
                    lineage.source_offline_store_manifest_hash,
                    lineage.split_manifest_hash,
                    lineage.branch_schedule_id,
                    lineage.target_protocol_version,
                    lineage.target_crop_policy,
                    lineage.target_reason_code_version,
                    lineage.reason_code_version,
                    lineage.selection_rng_state_hash,
                )
                if value
            }
        ),
        "class_name": sorted({lineage.target_class_name or "unknown" for _record, _trajectory, lineage in items}),
        "target_match_status": sorted(target_match_status_values),
        "termination_reason": sorted(
            {
                _termination_reason(record.result, trajectory)
                for record in records
                for trajectory in record.result.trajectories
            }
        ),
        "transition": sorted(
            {
                f"{lineage.rollout_id}:step={step.step_index}:shell={step.selected_shell_index}"
                for _record, trajectory, lineage in items
                for step in trajectory.steps
            }
            | {""}
        ),
    }


def _write_dictionaries(group: zarr.Group, dictionaries: dict[str, list[str]]) -> None:
    for name, values in dictionaries.items():
        _write_string_array(group, name, values)


def _write_targets(
    group: zarr.Group,
    records: list[RolloutZarrRecord],
    dictionaries: dict[str, list[str]],
    *,
    target_protocol_version: str,
) -> None:
    target_rows = _target_rows_from_records(records)
    target_ids = sorted(target_rows)
    if not target_ids:
        target_ids = [0]
        target_rows[0] = {}
    _write_array(group, "target_row_id", np.asarray(target_ids, dtype=np.int64))
    _write_array(
        group,
        "target_id",
        np.asarray(
            [
                _dict_id(
                    dictionaries["target"],
                    str(target_rows[target_row_id].get("target_id") or "unknown-target"),
                )
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "target_selection_policy_id",
        np.asarray(
            [
                _dict_id(dictionaries["policy"], str(target_rows[target_row_id].get("target_selection_policy") or ""))
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "target_selection_rank",
        np.asarray(
            [
                _int_or_default(target_rows[target_row_id].get("target_selection_rank"), default=-1)
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "target_selection_score",
        np.asarray(
            [_float_or_nan(target_rows[target_row_id].get("target_selection_score")) for target_row_id in target_ids],
            dtype=np.float32,
        ),
    )
    _write_array(
        group,
        "target_selection_probability",
        np.asarray(
            [
                _float_or_nan(target_rows[target_row_id].get("target_selection_probability"))
                for target_row_id in target_ids
            ],
            dtype=np.float32,
        ),
    )
    _write_array(
        group,
        "target_selection_temperature",
        np.asarray(
            [
                _float_or_nan(target_rows[target_row_id].get("target_selection_temperature"))
                for target_row_id in target_ids
            ],
            dtype=np.float32,
        ),
    )
    _write_array(
        group,
        "target_source_id",
        np.asarray(
            [
                _dict_id(dictionaries["target_source"], str(target_rows[target_row_id].get("target_source") or ""))
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "target_source_index",
        np.asarray(
            [
                _int_or_default(target_rows[target_row_id].get("target_source_index"), default=-1)
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "target_sem_id",
        np.asarray(
            [
                _int_or_default(target_rows[target_row_id].get("target_sem_id"), default=-1)
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "target_inst_id",
        np.asarray(
            [
                _int_or_default(target_rows[target_row_id].get("target_inst_id"), default=-1)
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "target_class_name_id",
        np.asarray(
            [
                _dict_id(dictionaries["class_name"], str(target_rows[target_row_id].get("target_class_name") or ""))
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "target_confidence",
        np.asarray(
            [_float_or_nan(target_rows[target_row_id].get("target_confidence")) for target_row_id in target_ids],
            dtype=np.float32,
        ),
    )
    _write_array(
        group,
        "target_center_world",
        np.asarray(
            [
                _fixed_float_vector(target_rows[target_row_id].get("target_center_world"), length=3)
                for target_row_id in target_ids
            ],
            dtype=np.float32,
        ),
    )
    _write_array(
        group,
        "target_extents",
        np.asarray(
            [
                _fixed_float_vector(target_rows[target_row_id].get("target_extents"), length=3)
                for target_row_id in target_ids
            ],
            dtype=np.float32,
        ),
    )
    _write_array(
        group,
        "target_pose_world_object",
        np.asarray(
            [
                _fixed_float_vector(target_rows[target_row_id].get("target_pose_world_object"), length=12)
                for target_row_id in target_ids
            ],
            dtype=np.float32,
        ),
    )
    _write_array(
        group,
        "target_relative_pose_reference_object",
        np.asarray(
            [
                _fixed_float_vector(
                    target_rows[target_row_id].get("target_relative_pose_reference_object"),
                    length=12,
                )
                for target_row_id in target_ids
            ],
            dtype=np.float32,
        ),
    )
    target_reason = np.asarray(
        [
            _int_or_default(
                target_rows[target_row_id].get("target_invalid_reason_bitset"),
                default=1 << INVALID_REASON_CODES["VALID"],
            )
            for target_row_id in target_ids
        ],
        dtype=np.uint32,
    )
    _write_array(group, "target_valid_mask", target_reason == np.uint32(1 << INVALID_REASON_CODES["VALID"]))
    _write_array(
        group,
        "target_invalid_reason_bitset",
        target_reason,
    )
    _write_array(
        group,
        "target_primary_invalid_reason",
        np.asarray(
            [
                _int_or_default(
                    target_rows[target_row_id].get("target_primary_invalid_reason"),
                    default=INVALID_REASON_CODES["VALID"],
                )
                for target_row_id in target_ids
            ],
            dtype=np.uint16,
        ),
    )
    _write_array(
        group,
        "target_reason_code_version_id",
        np.asarray(
            [
                _dict_id(
                    dictionaries["config"], str(target_rows[target_row_id].get("target_reason_code_version") or "")
                )
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "matched_gt_target_row_id",
        np.asarray(
            [
                _int_or_default(target_rows[target_row_id].get("matched_gt_target_row_id"), default=-1)
                for target_row_id in target_ids
            ],
            dtype=np.int64,
        ),
    )
    _write_array(
        group,
        "matched_gt_target_id",
        np.asarray(
            [
                _dict_id(dictionaries["target"], str(target_rows[target_row_id].get("matched_gt_target_id") or ""))
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "gt_match_iou",
        np.asarray(
            [_float_or_nan(target_rows[target_row_id].get("gt_match_iou")) for target_row_id in target_ids],
            dtype=np.float32,
        ),
    )
    _write_array(
        group,
        "gt_match_score",
        np.asarray(
            [_float_or_nan(target_rows[target_row_id].get("gt_match_score")) for target_row_id in target_ids],
            dtype=np.float32,
        ),
    )
    _write_array(
        group,
        "gt_match_status_id",
        np.asarray(
            [
                _dict_id(
                    dictionaries["target_match_status"],
                    str(target_rows[target_row_id].get("gt_match_status") or "not_requested"),
                )
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_array(
        group,
        "gt_label_valid_mask",
        np.asarray(
            [
                str(target_rows[target_row_id].get("gt_match_status") or "") in {"matched", "v0_gt_input"}
                and _int_or_default(target_rows[target_row_id].get("matched_gt_target_row_id"), default=-1) >= 0
                for target_row_id in target_ids
            ],
            dtype=np.bool_,
        ),
    )
    _write_array(
        group,
        "target_crop_policy_id",
        np.asarray(
            [
                _dict_id(dictionaries["config"], str(target_rows[target_row_id].get("target_crop_policy") or ""))
                for target_row_id in target_ids
            ],
            dtype=np.int32,
        ),
    )
    _write_string_array(group, "target_protocol_version", [target_protocol_version])


def _target_rows_from_records(records: list[RolloutZarrRecord]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for _record, _trajectory, lineage in _record_items(records):
        row_id = lineage.target_row_id if lineage.target_row_id is not None else 0
        existing = rows.setdefault(int(row_id), {})
        values = {
            "target_id": lineage.target_id or "unknown-target",
            "target_selection_policy": lineage.target_selection_policy,
            "target_selection_rank": lineage.target_selection_rank,
            "target_selection_score": lineage.target_selection_score,
            "target_selection_probability": lineage.target_selection_probability,
            "target_selection_temperature": lineage.target_selection_temperature,
            "target_source": lineage.target_source,
            "target_source_index": lineage.target_source_index,
            "target_sem_id": lineage.target_sem_id,
            "target_inst_id": lineage.target_inst_id,
            "target_class_name": lineage.target_class_name,
            "target_confidence": lineage.target_confidence,
            "target_center_world": lineage.target_center_world,
            "target_extents": lineage.target_extents,
            "target_pose_world_object": lineage.target_pose_world_object,
            "target_relative_pose_reference_object": lineage.target_relative_pose_reference_object,
            "target_invalid_reason_bitset": lineage.target_invalid_reason_bitset,
            "target_primary_invalid_reason": lineage.target_primary_invalid_reason,
            "target_reason_code_version": lineage.target_reason_code_version,
            "matched_gt_target_row_id": lineage.matched_gt_target_row_id,
            "matched_gt_target_id": lineage.matched_gt_target_id,
            "gt_match_iou": lineage.gt_match_iou,
            "gt_match_score": lineage.gt_match_score,
            "gt_match_status": lineage.gt_match_status,
            "target_crop_policy": lineage.target_crop_policy,
        }
        for name, value in values.items():
            if value is not None or name not in existing:
                existing[name] = value
    return rows


def _flatten_records(records: list[RolloutZarrRecord], dictionaries: dict[str, list[str]]) -> _RolloutTables:
    source_rows: dict[str, list[Any]] = _empty_rows(SOURCE_TABLE)
    rollout_rows: dict[str, list[Any]] = _empty_rows(ROLLOUT_TABLE)
    lineage_rows: dict[str, list[Any]] = _empty_rows(LINEAGE_TABLE)
    step_rows: dict[str, list[Any]] = _empty_rows(STEP_TABLE)
    candidate_rows: dict[str, list[Any]] = _empty_candidate_rows()

    candidate_row_id = 0
    step_row_id = 0
    seen_source_row_ids: set[int] = set()
    rollout_row_id = 0
    for record, trajectory, lineage in _record_items(records):
        final_target_rri = _trajectory_cumulative_metric(trajectory, ("target_rri", "rri"))
        if final_target_rri is None:
            final_target_rri = trajectory.cumulative_rri
        final_scene_rri = _trajectory_cumulative_metric(trajectory, ("scene_rri",))
        source_row_id = _lineage_source_row_id(lineage)
        if source_row_id not in seen_source_row_ids:
            seen_source_row_ids.add(source_row_id)
            _append_source_row(source_rows, lineage=lineage, source_row_id=source_row_id, dictionaries=dictionaries)
        rollout_rows["rollout_row_id"].append(rollout_row_id)
        rollout_rows["rollout_id"].append(_dict_id(dictionaries["rollout"], lineage.rollout_id))
        rollout_rows["chain_id"].append(lineage.chain_id)
        rollout_rows["source_row_id"].append(source_row_id)
        rollout_rows["root_pose_world"].append(
            record.result.root_pose_world.tensor().detach().cpu().to(dtype=torch.float32).reshape(-1).numpy()
        )
        rollout_rows["scene_id"].append(_dict_id(dictionaries["scene"], lineage.scene_id or ""))
        rollout_rows["snippet_id"].append(_dict_id(dictionaries["snippet"], lineage.snippet_id or ""))
        rollout_rows["target_row_id"].append(lineage.target_row_id if lineage.target_row_id is not None else 0)
        rollout_rows["policy_id"].append(_dict_id(dictionaries["policy"], _policy_name(record.result.selection_policy)))
        rollout_rows["horizon"].append(record.result.horizon)
        rollout_rows["branch_factor"].append(record.result.branch_factor)
        rollout_rows["beam_width"].append(-1 if record.result.beam_width is None else record.result.beam_width)
        rollout_rows["temperature"].append(_first_temperature(trajectory))
        rollout_rows["random_seed"].append(-1 if lineage.random_seed is None else lineage.random_seed)
        rollout_rows["termination_reason"].append(
            _dict_id(dictionaries["termination_reason"], _termination_reason(record.result, trajectory))
        )
        rollout_rows["final_cumulative_target_rri"].append(_nan_if_none(final_target_rri))
        rollout_rows["final_cumulative_scene_rri"].append(_nan_if_none(final_scene_rri))
        rollout_rows["split_id"].append(_dict_id(dictionaries["split"], lineage.split or "unknown"))

        lineage_rows["rollout_row_id"].append(rollout_row_id)
        lineage_rows["candidate_config_id"].append(
            _dict_id(dictionaries["config"], lineage.candidate_config_hash or "")
        )
        lineage_rows["oracle_config_id"].append(_dict_id(dictionaries["config"], lineage.oracle_config_hash or ""))
        lineage_rows["rollout_config_id"].append(_dict_id(dictionaries["config"], lineage.rollout_config_hash or ""))
        lineage_rows["model_checkpoint_id"].append(
            _dict_id(dictionaries["config"], lineage.model_checkpoint_hash or "")
        )
        lineage_rows["mesh_version_id"].append(_dict_id(dictionaries["config"], lineage.mesh_version or ""))
        lineage_rows["branch_schedule_id"].append(_dict_id(dictionaries["config"], lineage.branch_schedule_id or ""))
        lineage_rows["target_protocol_version_id"].append(
            _dict_id(dictionaries["config"], lineage.target_protocol_version or "")
        )
        lineage_rows["target_crop_policy_id"].append(_dict_id(dictionaries["config"], lineage.target_crop_policy or ""))
        lineage_rows["reason_code_version_id"].append(
            _dict_id(dictionaries["config"], lineage.reason_code_version or "")
        )
        lineage_rows["selection_rng_state_hash_id"].append(
            _dict_id(dictionaries["config"], lineage.selection_rng_state_hash or "")
        )

        running_target_rri: float | None = None
        running_scene_rri: float | None = None
        root_pose = record.result.root_pose_world.tensor().detach().cpu().reshape(-1)
        for step in trajectory.steps:
            candidate_valid = _candidate_valid(step)
            running_target_rri = _accumulate_selected_metric(running_target_rri, step, ("target_rri", "rri"))
            running_scene_rri = _accumulate_selected_metric(running_scene_rri, step, ("scene_rri",))
            this_step_row_id = step_row_id
            step_row_id += 1
            selected_candidate_row_id = candidate_row_id + int(step.selected_shell_index)
            step_rows["step_row_id"].append(this_step_row_id)
            step_rows["rollout_row_id"].append(rollout_row_id)
            step_rows["step_index"].append(step.step_index)
            step_rows["selected_candidate_row_id"].append(selected_candidate_row_id)
            step_rows["selected_shell_index"].append(step.selected_shell_index)
            step_rows["selected_compact_valid_index"].append(step.selected_valid_index)
            step_rows["num_candidates"].append(int(candidate_valid.shape[0]))
            step_rows["num_valid_candidates"].append(int(candidate_valid.sum().item()))
            step_rows["cumulative_target_rri"].append(_nan_if_none(running_target_rri))
            step_rows["cumulative_scene_rri"].append(_nan_if_none(running_scene_rri))
            transition_id = f"{lineage.rollout_id}:step={step.step_index}:shell={step.selected_shell_index}"
            step_rows["transition_id"].append(_dict_id(dictionaries["transition"], transition_id))

            for shell_index in range(int(candidate_valid.shape[0])):
                _append_candidate_row(
                    candidate_rows,
                    step=step,
                    candidate_valid=candidate_valid,
                    candidate_row_id=candidate_row_id,
                    step_row_id=this_step_row_id,
                    rollout_row_id=rollout_row_id,
                    shell_index=shell_index,
                    root_pose=root_pose,
                    dictionaries=dictionaries,
                    target_label_valid=_lineage_target_label_valid(lineage),
                )
                candidate_row_id += 1
        rollout_row_id += 1

    return _RolloutTables(
        sources=_rows_to_numpy_table(source_rows, SOURCE_TABLE),
        rollouts=_rows_to_numpy_table(rollout_rows, ROLLOUT_TABLE),
        lineage=_rows_to_numpy_table(lineage_rows, LINEAGE_TABLE),
        steps=_rows_to_numpy_table(step_rows, STEP_TABLE),
        candidates=_rows_to_numpy_table(candidate_rows, CANDIDATE_TABLE),
    )


def _append_source_row(
    rows: dict[str, list[Any]],
    *,
    lineage: RolloutLineage,
    source_row_id: int,
    dictionaries: dict[str, list[str]],
) -> None:
    rows["source_row_id"].append(source_row_id)
    rows["sample_index"].append(
        source_row_id if lineage.source_sample_index is None else int(lineage.source_sample_index)
    )
    rows["sample_key_id"].append(_dict_id(dictionaries["source_key"], lineage.source_sample_key or ""))
    rows["scene_id"].append(_dict_id(dictionaries["scene"], lineage.scene_id or ""))
    rows["snippet_id"].append(_dict_id(dictionaries["snippet"], lineage.snippet_id or ""))
    rows["split_id"].append(_dict_id(dictionaries["split"], lineage.split or "unknown"))
    rows["source_cache_version_id"].append(_dict_id(dictionaries["config"], lineage.source_cache_version or ""))
    rows["source_offline_store_manifest_hash_id"].append(
        _dict_id(dictionaries["config"], lineage.source_offline_store_manifest_hash or "")
    )
    rows["split_manifest_hash_id"].append(_dict_id(dictionaries["config"], lineage.split_manifest_hash or ""))
    rows["source_shard_id"].append(_dict_id(dictionaries["source_shard"], lineage.source_shard_id or ""))
    rows["source_shard_row"].append(-1 if lineage.source_shard_row is None else int(lineage.source_shard_row))


def _lineage_source_row_id(lineage: RolloutLineage) -> int:
    if lineage.source_row_id is not None:
        return int(lineage.source_row_id)
    if lineage.source_sample_index is not None:
        return int(lineage.source_sample_index)
    return 0


def _empty_rows(schema: _TableSchema) -> dict[str, list[Any]]:
    return {name: [] for name in schema.names}


def _empty_candidate_rows() -> dict[str, list[Any]]:
    return _empty_rows(CANDIDATE_TABLE)


def _append_candidate_row(
    rows: dict[str, list[Any]],
    *,
    step: CounterfactualStepResult,
    candidate_valid: torch.Tensor,
    candidate_row_id: int,
    step_row_id: int,
    rollout_row_id: int,
    shell_index: int,
    root_pose: torch.Tensor,
    dictionaries: dict[str, list[str]],
    target_label_valid: bool,
) -> None:
    is_valid = bool(candidate_valid[shell_index].item())
    is_selected = int(step.selected_shell_index) == int(shell_index)
    target_rri = _metric_value(step, ("target_rri", "oracle_target_rri"), shell_index)
    scene_rri = _metric_value(step, ("scene_rri", "oracle_scene_rri"), shell_index)
    if not is_valid:
        target_rri = float("nan")
        scene_rri = float("nan")
    oracle_label = bool(is_valid and np.isfinite(target_rri))
    q_train = bool(is_valid and oracle_label and target_label_valid)
    pose = step.candidates.shell_poses.tensor()[shell_index].detach().cpu().numpy().astype(np.float32)
    rows["candidate_row_id"].append(candidate_row_id)
    rows["step_row_id"].append(step_row_id)
    rows["rollout_row_id"].append(rollout_row_id)
    rows["step_index"].append(step.step_index)
    rows["shell_index"].append(shell_index)
    rows["compact_valid_index"].append(_compact_valid_index(candidate_valid, shell_index))
    rows["pose_world_cam"].append(pose)
    rows["pose_relative_root"].append(_relative_pose_to_root(pose_world_cam=pose, root_pose_world=root_pose))
    rows["candidate_valid_mask"].append(is_valid)
    rows["actor_action_mask"].append(is_valid)
    rows["oracle_label_mask"].append(oracle_label)
    rows["q_train_mask"].append(q_train)
    rows["padded_mask"].append(False)
    rows["selected_mask"].append(is_selected)
    rows["heavy_diag_available_mask"].append(bool(is_selected and step.selected_point_cloud_world is not None))
    rows["strategy_id"].append(_full_shell_value(step.candidates.strategy_id, shell_index, candidate_valid, default=-1))
    rows["mixture_id"].append(_full_shell_value(step.candidates.mixture_id, shell_index, candidate_valid, default=-1))
    rows["sampler_probability"].append(
        _full_shell_value(step.candidates.sampler_probability, shell_index, candidate_valid, default=np.nan)
    )
    rows["score_source_id"].append(_dict_id(dictionaries["score_source"], step.selection_score_label))
    reason_bitset, primary_reason = _candidate_invalid_reasons(step.candidates)
    rows["invalid_reason_bitset"].append(int(reason_bitset[shell_index].item()))
    rows["primary_invalid_reason"].append(int(primary_reason[shell_index].item()))
    rows["scene_rri"].append(scene_rri)
    rows["target_rri"].append(target_rri)
    rows["selection_logits"].append(
        _valid_vector_value(step.selection_logits, shell_index, candidate_valid, default=np.nan)
    )
    rows["selection_probabilities"].append(
        _valid_vector_value(step.selection_probabilities, shell_index, candidate_valid, default=0.0)
    )
    rows["selection_log_probabilities"].append(
        _valid_vector_value(step.selection_log_probabilities, shell_index, candidate_valid, default=-np.inf)
    )
    rows["selection_entropy"].append(_nan_if_none(step.selection_entropy))


def _write_rollout_tables(groups: dict[str, zarr.Group], tables: _RolloutTables) -> None:
    for name, values in tables.sources.items():
        _write_array(groups["sources"], name, values)
    for name, values in tables.rollouts.items():
        _write_array(groups["rollouts"], name, values)
    for name, values in tables.lineage.items():
        _write_array(groups["lineage"], name, values)
    for name, values in tables.steps.items():
        _write_array(groups["steps"], name, values)
    for name, values in tables.candidates.items():
        _write_array(groups["candidates"], name, values)


def _build_q_h_arrays(tables: _RolloutTables, *, horizon: int, gamma: float) -> dict[str, np.ndarray]:
    steps = tables.steps
    candidates = tables.candidates
    rollouts = tables.rollouts
    step_ids = steps["step_row_id"].astype(np.int64)
    candidate_step_ids = candidates["step_row_id"].astype(np.int64)
    max_candidates = _max_candidates_per_step(steps, candidates)
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
        "discount": np.asarray([float(gamma) ** power for power in range(h_max)], dtype=np.float32),
        "td_selected_candidate_row_id": np.full((state_count,), -1, dtype=np.int64),
        "td_reward_target_rri": np.full((state_count,), np.nan, dtype=np.float32),
        "td_next_step_row_id": np.full((state_count,), -1, dtype=np.int64),
        "td_terminal_mask": np.ones((state_count,), dtype=np.bool_),
        "td_discount": np.full((state_count,), float(gamma), dtype=np.float32),
    }

    next_step_by_rollout: dict[tuple[int, int], int] = {}
    for row, rollout_id in enumerate(steps["rollout_row_id"]):
        step_index = int(steps["step_index"][row])
        next_step_by_rollout[(int(rollout_id), step_index)] = int(steps["step_row_id"][row])

    for row, step_id in enumerate(step_ids):
        indices = np.nonzero(candidate_step_ids == step_id)[0]
        selected_candidate_row_id = int(steps["selected_candidate_row_id"][row])
        rollout_id = int(steps["rollout_row_id"][row])
        rollout_matches = np.nonzero(rollouts["rollout_row_id"] == rollout_id)[0]
        if rollout_matches.size == 1:
            q["target_row_id"][row] = int(rollouts["target_row_id"][int(rollout_matches[0])])
        selected_local_index = -1
        for local_index, candidate_index in enumerate(indices):
            q["candidate_row_id"][row, local_index] = int(candidates["candidate_row_id"][candidate_index])
            q["valid_action_mask"][row, local_index] = bool(candidates["actor_action_mask"][candidate_index])
            q["q_train_mask"][row, local_index] = bool(candidates["q_train_mask"][candidate_index])
            q["one_step_target_rri"][row, local_index] = float(candidates["target_rri"][candidate_index])
            q["one_step_scene_rri"][row, local_index] = float(candidates["scene_rri"][candidate_index])
            q["invalid_reason_bitset"][row, local_index] = int(candidates["invalid_reason_bitset"][candidate_index])
            if int(candidates["candidate_row_id"][candidate_index]) == selected_candidate_row_id:
                selected_local_index = local_index

        q["selected_candidate_index"][row] = selected_local_index
        q["td_selected_candidate_row_id"][row] = selected_candidate_row_id
        if selected_local_index >= 0:
            q["td_reward_target_rri"][row] = q["one_step_target_rri"][row, selected_local_index]
            next_step = next_step_by_rollout.get((rollout_id, int(steps["step_index"][row]) + 1), -1)
            q["td_next_step_row_id"][row] = next_step
            q["td_terminal_mask"][row] = next_step < 0
            if next_step >= 0:
                q["terminal_mask"][row, selected_local_index] = False
                q["bootstrap_next_step_row_id"][row, selected_local_index] = next_step

    q["q_train_mask"] &= q["valid_action_mask"]
    q["one_step_target_rri"][~q["valid_action_mask"]] = np.nan
    q["one_step_scene_rri"][~q["valid_action_mask"]] = np.nan
    return q


def _rows_to_numpy_table(rows: dict[str, list[Any]], schema: _TableSchema) -> dict[str, np.ndarray]:
    expected = set(schema.names)
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        extra = sorted(set(rows) - expected)
        raise ValueError(f"Row table fields do not match schema; missing={missing}, extra={extra}.")
    return {name: np.asarray(rows[name], dtype=dtype) for name, dtype in schema.dtypes.items()}


def _read_tables_from_root(root: Any) -> _RolloutTables:
    return _RolloutTables(
        sources=_read_group_table(root, SOURCE_TABLE),
        rollouts=_read_group_table(root, ROLLOUT_TABLE),
        lineage=_read_group_table(root, LINEAGE_TABLE),
        steps=_read_group_table(root, STEP_TABLE),
        candidates=_read_group_table(root, CANDIDATE_TABLE),
    )


def _read_group_table(root: Any, schema: _TableSchema) -> dict[str, np.ndarray]:
    return {field.name: np.asarray(root[f"{schema.name}/{field.name}"]) for field in schema.fields}


def _stored_horizon(root: Any) -> int:
    values = np.asarray(root["rollouts/horizon"])
    return int(values.max()) if values.size else 1


def _max_candidates_per_step(steps: dict[str, np.ndarray], candidates: dict[str, np.ndarray]) -> int:
    candidate_step_ids = candidates["step_row_id"].astype(np.int64)
    return max((int((candidate_step_ids == int(step_id)).sum()) for step_id in steps["step_row_id"]), default=0)


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> zarr.Array:
    array = np.asarray(values)
    chunks = _default_chunks(array)
    zarr_array = group.create_array(name, shape=array.shape, chunks=chunks, dtype=array.dtype, overwrite=True)
    zarr_array[...] = array
    return zarr_array


def _write_string_array(group: zarr.Group, name: str, values: list[str]) -> None:
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


def _record_items(
    records: list[RolloutZarrRecord],
) -> Iterator[tuple[RolloutZarrRecord, CounterfactualTrajectory, RolloutLineage]]:
    for record in records:
        for chain_id, trajectory in enumerate(record.result.trajectories):
            yield record, trajectory, record.lineage_for_chain(chain_id)


def _first_temperature(trajectory: CounterfactualTrajectory) -> float:
    for step in trajectory.steps:
        if step.selection_temperature is not None:
            return float(step.selection_temperature)
    return float("nan")


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _trajectory_cumulative_metric(trajectory: CounterfactualTrajectory, metric_names: tuple[str, ...]) -> float | None:
    cumulative: float | None = None
    for step in trajectory.steps:
        cumulative = _accumulate_selected_metric(cumulative, step, metric_names)
    return cumulative


def _accumulate_selected_metric(
    current: float | None,
    step: CounterfactualStepResult,
    metric_names: tuple[str, ...],
) -> float | None:
    for metric_name in metric_names:
        value = step.selected_metrics.get(metric_name)
        if value is not None and np.isfinite(float(value)):
            return float(value) if current is None else float(current + float(value))
    return current


def _float_or_nan(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _fixed_float_vector(value: Any, *, length: int) -> np.ndarray:
    if value is None:
        return np.full((length,), np.nan, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] != length:
        return np.full((length,), np.nan, dtype=np.float32)
    return array


def _int_or_default(value: Any, *, default: int) -> int:
    return int(default) if value is None else int(value)


def _candidate_valid(step: CounterfactualStepResult) -> torch.Tensor:
    return step.candidates.mask_valid.detach().cpu().to(dtype=torch.bool).reshape(-1)


def _compact_valid_index(candidate_valid: torch.Tensor, shell_index: int) -> int:
    if not bool(candidate_valid[shell_index].item()):
        return -1
    valid_indices = np.nonzero(candidate_valid.detach().cpu().numpy().astype(bool))[0]
    matches = np.nonzero(valid_indices == int(shell_index))[0]
    if matches.size != 1:
        return -1
    return int(matches[0])


def _metric_value(step: CounterfactualStepResult, metric_names: tuple[str, ...], shell_index: int) -> float:
    for metric_name in metric_names:
        values = step.metric_vectors.get(metric_name)
        if values is not None:
            return float(_valid_vector_value(values, shell_index, _candidate_valid(step), default=np.nan))
    return float("nan")


def _full_shell_value(
    values: torch.Tensor | None,
    shell_index: int,
    candidate_valid: torch.Tensor,
    *,
    default: float | int,
) -> float | int:
    full = _full_shell_or_default(values, candidate_valid, fill_value=default)
    return full[shell_index].detach().cpu().item()


def _valid_vector_value(
    values: torch.Tensor | None,
    shell_index: int,
    candidate_valid: torch.Tensor,
    *,
    default: float | int,
) -> float | int:
    if values is None:
        return default
    vector = values.detach().cpu().reshape(-1)
    valid_mask = candidate_valid.detach().cpu().to(dtype=torch.bool).reshape(-1)
    if vector.numel() == valid_mask.numel():
        return vector[shell_index].item()
    if vector.numel() != int(valid_mask.sum().item()):
        raise ValueError(
            f"Expected either {valid_mask.numel()} full-shell values or {int(valid_mask.sum().item())} "
            f"valid-candidate values, got {vector.numel()}."
        )
    if not bool(valid_mask[shell_index].item()):
        return default
    return vector[_compact_valid_index(valid_mask, shell_index)].item()


def _lineage_target_label_valid(lineage: RolloutLineage) -> bool:
    target_bitset = lineage.target_invalid_reason_bitset
    target_valid = target_bitset is None or int(target_bitset) == (1 << INVALID_REASON_CODES["VALID"])
    gt_status = lineage.gt_match_status
    gt_valid = gt_status in {"matched", "v0_gt_input"} and (
        lineage.matched_gt_target_row_id is not None and int(lineage.matched_gt_target_row_id) >= 0
    )
    return bool(target_valid and gt_valid)


def _relative_pose_to_root(*, pose_world_cam: np.ndarray, root_pose_world: torch.Tensor) -> np.ndarray:
    root = PoseTW(root_pose_world.detach().cpu().to(dtype=torch.float32).reshape(-1))
    candidate = PoseTW(torch.as_tensor(pose_world_cam, dtype=torch.float32).reshape(-1))
    return root.inverse().compose(candidate).tensor().detach().cpu().numpy().astype(np.float32).reshape(-1)


def _missing_lineage_token(value: Any) -> bool:
    return value is None or str(value) == ""


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


__all__ = [
    "DEFAULT_RETURN_SEMANTICS",
    "ROLLOUT_ZARR_SCHEMA_ID",
    "ROLLOUT_ZARR_SCHEMA_VERSION",
    "RolloutZarrStoreConfig",
    "RolloutZarrStoreReader",
    "RolloutZarrValidationResult",
    "RolloutZarrWriteResult",
    "validate_rollout_zarr_store",
    "write_rollout_zarr_store",
]
