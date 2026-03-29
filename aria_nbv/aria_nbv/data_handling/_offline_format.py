"""Typed manifest and index records for the VIN offline dataset format.

The new offline dataset format is an immutable indexed-shard layout optimized
for multi-worker random access. This module defines the normalized metadata
records shared by the writer, migration tools, and runtime dataset reader:

- the top-level dataset manifest,
- per-shard block descriptors, and
- sample-index records used for global random access and split membership.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary.

    Args:
        path: JSON file to read.

    Returns:
        Parsed JSON dictionary.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary to JSON with stable formatting.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable dictionary to persist.
    """

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(slots=True)
class VinOfflineBlockSpec:
    """Descriptor for one stored block inside a shard.

    Attributes:
        name: Logical block name, for example ``"vin.points_world"``.
        kind: Storage kind such as ``"fixed_npy"`` or ``"pickle_records"``.
        paths: Relative file paths that materialize the block.
        dtype: NumPy dtype name for numeric blocks.
        shape: Full stored array shape for numeric blocks.
        optional: Whether the block may be absent for some datasets.
    """

    name: str
    """Logical block name."""

    kind: str
    """Storage kind used to decode the block."""

    paths: list[str]
    """Relative file paths that materialize the block."""

    dtype: str | None = None
    """NumPy dtype name for numeric blocks."""

    shape: list[int] | None = None
    """Full stored array shape for numeric blocks."""

    optional: bool = False
    """Whether the block may be absent in a valid dataset."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the block spec into a JSON-ready dictionary.

        Returns:
            JSON-serializable block-spec dictionary.
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VinOfflineBlockSpec":
        """Build a block spec from a JSON dictionary.

        Args:
            payload: Parsed block payload.

        Returns:
            Deserialized block descriptor.
        """

        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            paths=[str(item) for item in payload["paths"]],
            dtype=payload.get("dtype"),
            shape=[int(item) for item in payload["shape"]] if payload.get("shape") is not None else None,
            optional=bool(payload.get("optional", False)),
        )


@dataclass(slots=True)
class VinOfflineShardSpec:
    """Descriptor for one immutable dataset shard.

    Attributes:
        shard_id: Stable shard identifier, for example ``"shard-000003"``.
        relative_dir: Relative directory containing the shard files.
        row_start: Global row offset covered by the shard.
        num_rows: Number of samples stored in the shard.
        blocks: Stored block descriptors keyed by logical block name.
    """

    shard_id: str
    """Stable shard identifier."""

    relative_dir: str
    """Relative directory that contains shard artifacts."""

    row_start: int
    """Global row offset covered by this shard."""

    num_rows: int
    """Number of samples stored in this shard."""

    blocks: dict[str, VinOfflineBlockSpec] = field(default_factory=dict)
    """Stored block descriptors keyed by logical block name."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the shard spec into a JSON-ready dictionary.

        Returns:
            JSON-serializable shard-spec dictionary.
        """

        return {
            "shard_id": self.shard_id,
            "relative_dir": self.relative_dir,
            "row_start": self.row_start,
            "num_rows": self.num_rows,
            "blocks": {name: spec.to_dict() for name, spec in self.blocks.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VinOfflineShardSpec":
        """Build a shard descriptor from a JSON dictionary.

        Args:
            payload: Parsed shard payload.

        Returns:
            Deserialized shard descriptor.
        """

        return cls(
            shard_id=str(payload["shard_id"]),
            relative_dir=str(payload["relative_dir"]),
            row_start=int(payload["row_start"]),
            num_rows=int(payload["num_rows"]),
            blocks={
                str(name): VinOfflineBlockSpec.from_dict(spec) for name, spec in dict(payload.get("blocks", {})).items()
            },
        )


@dataclass(slots=True)
class VinOfflineMaterializedBlocks:
    """Materialized block flags for a VIN offline dataset."""

    backbone: bool
    """Whether backbone outputs are materialized."""

    depths: bool
    """Whether candidate depth maps are materialized."""

    candidate_pcs: bool
    """Whether candidate point clouds are materialized."""

    counterfactuals: bool
    """Whether future counterfactual trajectory blocks are materialized."""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VinOfflineMaterializedBlocks":
        """Build materialized-block flags from JSON.

        Args:
            payload: Parsed block-flag payload.

        Returns:
            Deserialized materialized-block record.
        """

        return cls(
            backbone=bool(payload.get("backbone", False)),
            depths=bool(payload.get("depths", False)),
            candidate_pcs=bool(payload.get("candidate_pcs", False)),
            counterfactuals=bool(payload.get("counterfactuals", False)),
        )


@dataclass(slots=True)
class VinOfflineCounterfactuals:
    """Counterfactual trajectory metadata reserved for future extensions."""

    enabled: bool = False
    """Whether counterfactual trajectories are materialized."""

    k: int | None = None
    """Optional number of counterfactual trajectories per snippet."""

    horizon: int | None = None
    """Optional rollout horizon in steps."""

    selection_policy: str | None = None
    """Optional policy used to choose counterfactual trajectories."""

    materialized_modalities: list[str] = field(default_factory=list)
    """Optional modalities materialized for each counterfactual trajectory."""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VinOfflineCounterfactuals":
        """Build counterfactual metadata from JSON.

        Args:
            payload: Parsed counterfactual payload.

        Returns:
            Deserialized counterfactual metadata.
        """

        return cls(
            enabled=bool(payload.get("enabled", False)),
            k=int(payload["k"]) if payload.get("k") is not None else None,
            horizon=int(payload["horizon"]) if payload.get("horizon") is not None else None,
            selection_policy=payload.get("selection_policy"),
            materialized_modalities=[str(item) for item in payload.get("materialized_modalities", [])],
        )


@dataclass(slots=True)
class VinOfflineManifest:
    """Top-level manifest for one immutable VIN offline dataset.

    Attributes:
        version: Dataset-format version.
        created_at: UTC creation timestamp.
        source: Raw dataset provenance and configuration snapshot.
        oracle: Oracle-label pipeline provenance and storage policy.
        vin: VIN-specific materialization settings.
        materialized_blocks: Flags for optional stored blocks.
        counterfactuals: Reserved future counterfactual trajectory metadata.
        stats: Aggregate dataset statistics.
        provenance: Legacy source provenance and migration hints.
        shards: Immutable shard descriptors.
    """

    version: int
    """Dataset-format version."""

    created_at: str
    """UTC creation timestamp."""

    source: dict[str, Any]
    """Raw dataset provenance and configuration snapshot."""

    oracle: dict[str, Any]
    """Oracle-label pipeline provenance and storage policy."""

    vin: dict[str, Any]
    """VIN-specific padding and collapse settings."""

    materialized_blocks: VinOfflineMaterializedBlocks
    """Flags describing which optional blocks are materialized."""

    counterfactuals: VinOfflineCounterfactuals = field(default_factory=VinOfflineCounterfactuals)
    """Reserved future counterfactual trajectory metadata."""

    stats: dict[str, Any] = field(default_factory=dict)
    """Aggregate dataset statistics."""

    provenance: dict[str, Any] = field(default_factory=dict)
    """Legacy source provenance and migration hints."""

    shards: list[VinOfflineShardSpec] = field(default_factory=list)
    """Immutable shard descriptors."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest into a JSON-ready dictionary.

        Returns:
            JSON-serializable manifest dictionary.
        """

        payload = asdict(self)
        payload["materialized_blocks"] = asdict(self.materialized_blocks)
        payload["counterfactuals"] = asdict(self.counterfactuals)
        payload["shards"] = [
            {
                "shard_id": shard.shard_id,
                "relative_dir": shard.relative_dir,
                "row_start": shard.row_start,
                "num_rows": shard.num_rows,
                "blocks": {name: asdict(spec) for name, spec in shard.blocks.items()},
            }
            for shard in self.shards
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VinOfflineManifest":
        """Deserialize a manifest from a dictionary.

        Args:
            payload: Parsed manifest payload.

        Returns:
            Deserialized manifest object.
        """

        return cls(
            version=int(payload["version"]),
            created_at=str(payload["created_at"]),
            source=dict(payload.get("source", {})),
            oracle=dict(payload.get("oracle", {})),
            vin=dict(payload.get("vin", {})),
            materialized_blocks=VinOfflineMaterializedBlocks.from_dict(
                dict(payload.get("materialized_blocks", {})),
            ),
            counterfactuals=VinOfflineCounterfactuals.from_dict(dict(payload.get("counterfactuals", {}))),
            stats=dict(payload.get("stats", {})),
            provenance=dict(payload.get("provenance", {})),
            shards=[VinOfflineShardSpec.from_dict(item) for item in payload.get("shards", [])],
        )

    def write(self, path: Path) -> None:
        """Persist the manifest to disk.

        Args:
            path: Destination manifest path.
        """

        _write_json(path, self.to_dict())

    @classmethod
    def read(cls, path: Path) -> "VinOfflineManifest":
        """Load a manifest from disk.

        Args:
            path: Manifest JSON path.

        Returns:
            Deserialized manifest.
        """

        return cls.from_dict(_load_json(path))


@dataclass(slots=True)
class VinOfflineIndexRecord:
    """Global sample-index entry for VIN offline random access.

    Attributes:
        sample_index: Global zero-based sample index.
        sample_key: Stable dataset sample key.
        scene_id: ASE scene identifier.
        snippet_id: ASE snippet identifier.
        split: Canonical split membership: ``all``, ``train``, or ``val``.
        shard_id: Shard that stores the sample.
        row: Zero-based row offset inside the shard.
        legacy_oracle_key: Optional legacy oracle-cache key used during migration.
        legacy_oracle_path: Optional legacy oracle-cache payload path.
        legacy_vin_key: Optional legacy VIN-cache key used during migration.
        legacy_vin_path: Optional legacy VIN-cache payload path.
    """

    sample_index: int
    """Global zero-based sample index."""

    sample_key: str
    """Stable dataset sample key."""

    scene_id: str
    """ASE scene identifier."""

    snippet_id: str
    """ASE snippet identifier."""

    split: str
    """Canonical split membership."""

    shard_id: str
    """Shard that stores the sample."""

    row: int
    """Zero-based row offset inside the shard."""

    legacy_oracle_key: str | None = None
    """Optional legacy oracle-cache key used during migration."""

    legacy_oracle_path: str | None = None
    """Optional legacy oracle-cache payload path."""

    legacy_vin_key: str | None = None
    """Optional legacy VIN-cache key used during migration."""

    legacy_vin_path: str | None = None
    """Optional legacy VIN-cache payload path."""

    def to_json(self) -> str:
        """Serialize the index record to one JSON line.

        Returns:
            JSON representation of the record.
        """

        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VinOfflineIndexRecord":
        """Build an index record from a parsed dictionary.

        Args:
            payload: Parsed JSON dictionary.

        Returns:
            Deserialized index record.
        """

        return cls(
            sample_index=int(payload["sample_index"]),
            sample_key=str(payload["sample_key"]),
            scene_id=str(payload["scene_id"]),
            snippet_id=str(payload["snippet_id"]),
            split=str(payload["split"]),
            shard_id=str(payload["shard_id"]),
            row=int(payload["row"]),
            legacy_oracle_key=payload.get("legacy_oracle_key"),
            legacy_oracle_path=payload.get("legacy_oracle_path"),
            legacy_vin_key=payload.get("legacy_vin_key"),
            legacy_vin_path=payload.get("legacy_vin_path"),
        )


def read_sample_index(path: Path) -> list[VinOfflineIndexRecord]:
    """Read the global sample index.

    Args:
        path: ``sample_index.jsonl`` path.

    Returns:
        Parsed sample-index records.
    """

    records: list[VinOfflineIndexRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(VinOfflineIndexRecord.from_dict(json.loads(line)))
    return records


def write_sample_index(path: Path, records: list[VinOfflineIndexRecord]) -> None:
    """Write the global sample index.

    Args:
        path: Destination ``sample_index.jsonl`` path.
        records: Global sample-index records to persist.
    """

    payload = "\n".join(record.to_json() for record in records)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


__all__ = [
    "VinOfflineBlockSpec",
    "VinOfflineCounterfactuals",
    "VinOfflineIndexRecord",
    "VinOfflineManifest",
    "VinOfflineMaterializedBlocks",
    "VinOfflineShardSpec",
    "read_sample_index",
    "write_sample_index",
]
