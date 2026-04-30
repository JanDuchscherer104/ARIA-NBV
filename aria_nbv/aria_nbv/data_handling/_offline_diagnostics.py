"""Diagnostics for immutable VIN offline stores.

The helpers in this module inspect the current ``VinOfflineDataset`` storage
format directly. They replace the old oracle-cache Streamlit statistics path
without reviving deprecated cache datasets or snippet providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._offline_store import VinOfflineStoreConfig, VinOfflineStoreReader


@dataclass(slots=True)
class NumericSummary:
    """Finite-value summary for one numeric diagnostic series."""

    count: int
    """Number of observed finite values."""

    minimum: float | None
    """Minimum finite value, or ``None`` when no finite values exist."""

    mean: float | None
    """Mean finite value, or ``None`` when no finite values exist."""

    maximum: float | None
    """Maximum finite value, or ``None`` when no finite values exist."""


@dataclass(slots=True)
class VinOfflineDatasetStats:
    """Store-level diagnostics for an immutable VIN offline dataset."""

    store_dir: str
    """Absolute store directory inspected for diagnostics."""

    version: int
    """Offline dataset format version from ``manifest.json``."""

    num_samples: int
    """Number of rows listed in ``sample_index.jsonl``."""

    sampled_samples: int
    """Number of rows scanned for per-sample statistics."""

    split_counts: dict[str, int]
    """Sample counts keyed by split name."""

    num_scenes: int
    """Number of unique scene IDs in the sample index."""

    num_snippets: int
    """Number of unique ``(scene_id, snippet_id)`` pairs."""

    materialized_blocks: dict[str, bool]
    """Optional block flags from the manifest."""

    candidate_count: NumericSummary
    """Distribution of valid candidate counts per sample."""

    rri: NumericSummary
    """Distribution of finite oracle RRI values across sampled candidates."""

    vin_points: NumericSummary
    """Distribution of VIN point lengths across sampled snippets."""

    numeric_bytes: int
    """Approximate bytes occupied by manifest-declared numeric shard blocks."""

    block_shapes: dict[str, list[int]] = field(default_factory=dict)
    """Stored numeric block shapes keyed by logical block name."""

    block_diagnostics: list["VinOfflineBlockDiagnostic"] = field(default_factory=list)
    """Manifest-declared block diagnostics for Streamlit and CLI tables."""

    sample_summaries: list["VinOfflineSampleDiagnostic"] = field(default_factory=list)
    """Per-row sanity summaries for sampled records."""

    candidate_count_values: list[float] = field(default_factory=list)
    """Sampled candidate-count values used for histograms."""

    rri_values: list[float] = field(default_factory=list)
    """Sampled finite oracle RRI values used for histograms."""

    vin_point_values: list[float] = field(default_factory=list)
    """Sampled VIN point lengths used for histograms."""


@dataclass(slots=True)
class VinOfflineBlockDiagnostic:
    """Render-ready manifest summary for one stored offline block."""

    shard_id: str
    """Shard that declares the block."""

    name: str
    """Logical block name."""

    kind: str
    """Storage kind such as ``zarr_array`` or ``msgpack_indexed_records``."""

    dtype: str | None
    """Stored NumPy dtype for numeric blocks."""

    shape: list[int] | None
    """Stored array shape or record count."""

    optional: bool
    """Whether the block is optional."""

    estimated_bytes: int | None
    """Estimated numeric bytes, or ``None`` for non-numeric blocks."""


@dataclass(slots=True)
class VinOfflineSampleDiagnostic:
    """Per-row sanity summary for one sampled VIN offline record."""

    sample_index: int
    """Global sample index."""

    sample_key: str
    """Stable sample key."""

    scene_id: str
    """ASE scene identifier."""

    snippet_id: str
    """ASE snippet identifier."""

    split: str
    """Dataset split membership."""

    shard_id: str
    """Shard that stores the row."""

    row: int
    """Shard-local row index."""

    candidate_count: int
    """Valid candidate count for the row."""

    rri: NumericSummary
    """Finite RRI summary for valid candidates."""

    vin_points: NumericSummary
    """VIN point-length summary for the row."""


def _finite_values(values: np.ndarray) -> list[float]:
    """Return finite values from a numeric array as Python floats."""

    flat = np.asarray(values).reshape(-1)
    finite = flat[np.isfinite(flat)]
    return [float(value) for value in finite]


def _summary(values: list[float]) -> NumericSummary:
    """Summarize finite numeric values."""

    if not values:
        return NumericSummary(count=0, minimum=None, mean=None, maximum=None)
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return NumericSummary(count=0, minimum=None, mean=None, maximum=None)
    return NumericSummary(
        count=int(finite.size),
        minimum=float(np.min(finite)),
        mean=float(np.mean(finite)),
        maximum=float(np.max(finite)),
    )


def _collect_block_diagnostics(
    reader: VinOfflineStoreReader,
) -> tuple[int, dict[str, list[int]], list[VinOfflineBlockDiagnostic]]:
    """Collect manifest-declared block shapes and byte estimates."""

    total = 0
    block_shapes: dict[str, list[int]] = {}
    diagnostics: list[VinOfflineBlockDiagnostic] = []
    for shard in reader.manifest.shards:
        for block_name, spec in shard.blocks.items():
            shape = [int(dim) for dim in spec.shape] if spec.shape is not None else None
            estimated_bytes: int | None = None
            if spec.kind == "zarr_array" and shape is not None and spec.dtype is not None:
                block_shapes.setdefault(block_name, shape)
                estimated_bytes = int(np.prod(shape, dtype=np.int64)) * int(np.dtype(spec.dtype).itemsize)
                total += estimated_bytes
            diagnostics.append(
                VinOfflineBlockDiagnostic(
                    shard_id=shard.shard_id,
                    name=block_name,
                    kind=spec.kind,
                    dtype=spec.dtype,
                    shape=shape,
                    optional=bool(spec.optional),
                    estimated_bytes=estimated_bytes,
                ),
            )
    diagnostics.sort(key=lambda item: (item.shard_id, item.name))
    return total, block_shapes, diagnostics


def collect_vin_offline_dataset_stats(
    store: VinOfflineStoreConfig,
    *,
    max_samples: int | None = 512,
) -> VinOfflineDatasetStats:
    """Collect coverage, shape, RRI, and memory diagnostics for a VIN store.

    Args:
        store: Immutable VIN offline store to inspect.
        max_samples: Optional cap on rows scanned for per-sample tensor stats.
            ``None`` scans the whole store.

    Returns:
        Store-level diagnostics suitable for tests, CLI output, or Streamlit.
    """

    reader = VinOfflineStoreReader(store)
    records = reader.sample_index
    scan_records = records if max_samples is None else records[: max(0, int(max_samples))]

    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1

    candidate_counts: list[float] = []
    rri_values: list[float] = []
    vin_lengths: list[float] = []
    sample_summaries: list[VinOfflineSampleDiagnostic] = []
    for record in scan_records:
        candidate_count = int(reader.read_numeric_block(record, "oracle.candidate_count").reshape(()))
        candidate_counts.append(float(candidate_count))
        rri = np.asarray(reader.read_numeric_block(record, "oracle.rri")).reshape(-1)[:candidate_count]
        row_rri_values = _finite_values(rri)
        rri_values.extend(row_rri_values)
        lengths = np.asarray(reader.read_numeric_block(record, "vin.lengths")).reshape(-1)
        row_vin_lengths = _finite_values(lengths)
        vin_lengths.extend(row_vin_lengths)
        sample_summaries.append(
            VinOfflineSampleDiagnostic(
                sample_index=int(record.sample_index),
                sample_key=record.sample_key,
                scene_id=record.scene_id,
                snippet_id=record.snippet_id,
                split=record.split,
                shard_id=record.shard_id,
                row=int(record.row),
                candidate_count=candidate_count,
                rri=_summary(row_rri_values),
                vin_points=_summary(row_vin_lengths),
            ),
        )

    numeric_bytes, block_shapes, block_diagnostics = _collect_block_diagnostics(reader)
    return VinOfflineDatasetStats(
        store_dir=store.store_dir.expanduser().resolve().as_posix(),
        version=int(reader.manifest.version),
        num_samples=len(records),
        sampled_samples=len(scan_records),
        split_counts=split_counts,
        num_scenes=len({record.scene_id for record in records}),
        num_snippets=len({(record.scene_id, record.snippet_id) for record in records}),
        materialized_blocks={
            "backbone": bool(reader.manifest.materialized_blocks.backbone),
            "depths": bool(reader.manifest.materialized_blocks.depths),
            "candidate_pcs": bool(reader.manifest.materialized_blocks.candidate_pcs),
            "counterfactuals": bool(reader.manifest.materialized_blocks.counterfactuals),
        },
        candidate_count=_summary(candidate_counts),
        rri=_summary(rri_values),
        vin_points=_summary(vin_lengths),
        numeric_bytes=int(numeric_bytes),
        block_shapes=block_shapes,
        block_diagnostics=block_diagnostics,
        sample_summaries=sample_summaries,
        candidate_count_values=candidate_counts,
        rri_values=rri_values,
        vin_point_values=vin_lengths,
    )


__all__ = [
    "NumericSummary",
    "VinOfflineBlockDiagnostic",
    "VinOfflineDatasetStats",
    "VinOfflineSampleDiagnostic",
    "collect_vin_offline_dataset_stats",
]
