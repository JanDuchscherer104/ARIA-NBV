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


def _estimate_numeric_bytes(reader: VinOfflineStoreReader) -> tuple[int, dict[str, list[int]]]:
    """Estimate bytes occupied by numeric blocks declared in the manifest."""

    total = 0
    block_shapes: dict[str, list[int]] = {}
    for shard in reader.manifest.shards:
        for block_name, spec in shard.blocks.items():
            if spec.kind != "zarr_array" or spec.shape is None or spec.dtype is None:
                continue
            shape = [int(dim) for dim in spec.shape]
            block_shapes.setdefault(block_name, shape)
            total += int(np.prod(shape, dtype=np.int64)) * int(np.dtype(spec.dtype).itemsize)
    return total, block_shapes


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
    for record in scan_records:
        candidate_count = int(reader.read_numeric_block(record, "oracle.candidate_count").reshape(()))
        candidate_counts.append(float(candidate_count))
        rri = np.asarray(reader.read_numeric_block(record, "oracle.rri")).reshape(-1)[:candidate_count]
        rri_values.extend(float(value) for value in rri[np.isfinite(rri)])
        lengths = np.asarray(reader.read_numeric_block(record, "vin.lengths")).reshape(-1)
        vin_lengths.extend(float(value) for value in lengths[np.isfinite(lengths)])

    numeric_bytes, block_shapes = _estimate_numeric_bytes(reader)
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
    )


__all__ = [
    "NumericSummary",
    "VinOfflineDatasetStats",
    "collect_vin_offline_dataset_stats",
]
