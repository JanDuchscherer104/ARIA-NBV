"""Tests for rollout dataset writer lineage helpers."""

# ruff: noqa: S101

from aria_nbv.rollouts.dataset_writer import _RolloutSourceLineageBuilder


def test_split_manifest_hash_tracks_source_rows_and_order() -> None:
    rows = [
        {
            "order": 0,
            "sample_index": 1,
            "sample_key": "a",
            "scene_id": "scene-a",
            "snippet_id": "snippet-a",
            "split": "train",
            "shard_id": "shard-0",
            "row": 0,
        },
        {
            "order": 1,
            "sample_index": 2,
            "sample_key": "b",
            "scene_id": "scene-b",
            "snippet_id": "snippet-b",
            "split": "train",
            "shard_id": "shard-0",
            "row": 1,
        },
    ]

    base = _RolloutSourceLineageBuilder.build_split_manifest_hash(
        source_manifest_hash="source", split="train", records=rows
    )
    reordered = _RolloutSourceLineageBuilder.build_split_manifest_hash(
        source_manifest_hash="source", split="train", records=list(reversed(rows))
    )
    changed_source = _RolloutSourceLineageBuilder.build_split_manifest_hash(
        source_manifest_hash="other", split="train", records=rows
    )

    assert base != reordered
    assert base != changed_source
