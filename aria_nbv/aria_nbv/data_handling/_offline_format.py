"""Typed manifest and index records for the VIN offline dataset format.

The new offline dataset format is an immutable indexed-shard layout optimized
for multi-worker random access. This module defines the normalized metadata
records shared by the writer, migration tools, and runtime dataset reader:

- the top-level dataset manifest,
- per-shard block descriptors, and
- sample-index records used for global random access and split membership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import msgspec

OfflineRecord = TypeVar("OfflineRecord")
_JSON_ENCODER = msgspec.json.Encoder()


def _read_typed_json(path: Path, *, record_type: type[OfflineRecord]) -> OfflineRecord:
    """Read one typed JSON record from disk.

    Args:
        path: JSON file to read.
        record_type: Target dataclass type.

    Returns:
        Decoded record instance.
    """

    return msgspec.json.decode(path.read_bytes(), type=record_type)


def _write_json(path: Path, payload: Any) -> None:
    """Write one JSON-serializable payload to disk.

    Args:
        path: Destination JSON path.
        payload: Typed payload to persist.
    """

    path.write_bytes(_JSON_ENCODER.encode(payload))


@dataclass(slots=True)
class VinOfflineBlockSpec:
    """Descriptor for one stored block inside a shard.

    Attributes:
        name: Logical block name, for example ``"vin.points_world"``.
        kind: Storage kind such as ``"zarr_array"`` or ``"pickle_records"``.
        paths: Relative array names or file paths that materialize the block.
        dtype: NumPy dtype name for numeric blocks.
        shape: Full stored array shape for numeric blocks.
        optional: Whether the block may be absent for some datasets.
    """

    name: str
    """Logical block name."""

    kind: str
    """Storage kind used to decode the block."""

    paths: list[str]
    """Relative array names or file paths that materialize the block."""

    dtype: str | None = None
    """NumPy dtype name for numeric blocks."""

    shape: list[int] | None = None
    """Full stored array shape for numeric blocks."""

    optional: bool = False
    """Whether the block may be absent in a valid dataset."""


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

    def write(self, path: Path) -> None:
        """Persist the manifest to disk.

        Args:
            path: Destination manifest path.
        """

        _write_json(path, self)

    @classmethod
    def read(cls, path: Path) -> "VinOfflineManifest":
        """Load a manifest from disk.

        Args:
            path: Manifest JSON path.

        Returns:
            Deserialized manifest.
        """

        return _read_typed_json(path, record_type=cls)


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


def read_sample_index(path: Path) -> list[VinOfflineIndexRecord]:
    """Read the global sample index.

    Args:
        path: ``sample_index.jsonl`` path.

    Returns:
        Parsed sample-index records.
    """

    records: list[VinOfflineIndexRecord] = []
    for line in path.read_bytes().splitlines():
        if line.strip():
            records.append(msgspec.json.decode(line, type=VinOfflineIndexRecord))
    return records


def write_sample_index(path: Path, records: list[VinOfflineIndexRecord]) -> None:
    """Write the global sample index.

    Args:
        path: Destination ``sample_index.jsonl`` path.
        records: Global sample-index records to persist.
    """

    payload = b"\n".join(_JSON_ENCODER.encode(record) for record in records)
    if payload:
        payload += b"\n"
    path.write_bytes(payload)


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
