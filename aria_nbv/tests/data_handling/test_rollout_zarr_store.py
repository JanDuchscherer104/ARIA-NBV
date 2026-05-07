"""Smoke tests for the standalone rollout Zarr replay store."""

# ruff: noqa: S101

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("efm3d")

from efm3d.aria.pose import PoseTW

from aria_nbv.data_handling import (
    RolloutZarrStoreReader,
    validate_rollout_zarr_store,
    write_rollout_zarr_store,
)
from aria_nbv.pose_generation import INVALID_REASON_VERSION, build_synthetic_rollout_traces


def _json_list(reader: RolloutZarrStoreReader, path: str) -> list[str]:
    return json.loads(bytes(reader.array(path).tolist()).decode("utf-8"))


def test_rollout_zarr_store_writes_reads_and_validates_synthetic_traces(tmp_path) -> None:
    traces = build_synthetic_rollout_traces(horizon=2, num_samples=8, seed=7)
    result = write_rollout_zarr_store(
        tmp_path / "rollouts.zarr",
        traces,
        discount_gamma=0.95,
        target_protocol_version="synthetic",
    )

    assert result.num_rollouts == 3
    assert result.num_steps == 6
    assert result.num_candidates > 0
    assert result.q_h_state_count == result.num_steps

    validation = validate_rollout_zarr_store(result.store_dir)
    assert validation.ok, validation.errors

    reader = RolloutZarrStoreReader(result.store_dir)
    assert reader.root.attrs["schema_id"] == "aria_nbv.rollout_zarr_q_invalidity"
    assert reader.root.attrs["return_semantics"] == "cumulative_target_rri"

    candidate_valid = reader.array("candidates/candidate_valid_mask")
    selection_probabilities = reader.array("candidates/selection_probabilities")
    assert np.all(selection_probabilities[~candidate_valid] == 0.0)
    assert np.all(reader.array("candidates/selected_mask") <= candidate_valid)

    q_candidate_row_id = reader.array("q_h/candidate_row_id")
    valid_action_mask = reader.array("q_h/valid_action_mask")
    q_train_mask = reader.array("q_h/q_train_mask")
    q_target = reader.array("q_h/q_target_target_rri")
    q_target_available = reader.array("q_h/q_target_available_mask")

    assert q_candidate_row_id.shape == valid_action_mask.shape
    assert np.all(q_train_mask <= valid_action_mask)
    assert np.isnan(q_target[~q_target_available]).all()
    assert np.all(~valid_action_mask[q_candidate_row_id < 0])
    assert np.allclose(reader.array("q_h/discount")[:2], np.asarray([1.0, 0.95], dtype=np.float32))


def test_rollout_zarr_selected_action_td_fields_align_with_step_rows(tmp_path) -> None:
    traces = build_synthetic_rollout_traces(horizon=2, num_samples=6, seed=3)
    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", traces)
    reader = RolloutZarrStoreReader(result.store_dir)

    step_selected = reader.array("steps/selected_candidate_row_id")
    td_selected = reader.array("q_h/td_selected_candidate_row_id")
    td_next = reader.array("q_h/td_next_step_row_id")
    td_terminal = reader.array("q_h/td_terminal_mask")

    assert np.array_equal(td_selected, step_selected)
    assert td_next.shape == td_terminal.shape == td_selected.shape
    assert np.any(~td_terminal)
    assert np.any(td_terminal)


def test_rollout_zarr_requires_explicit_target_rri_for_q_training(tmp_path) -> None:
    traces = build_synthetic_rollout_traces(horizon=1, num_samples=6, seed=11)[:1]
    for step in traces[0].steps:
        step.metric_vectors = {}
        step.selected_metrics = {}

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", traces)
    reader = RolloutZarrStoreReader(result.store_dir)

    assert np.isfinite(reader.array("candidates/selection_logits")).any()
    assert np.isnan(reader.array("candidates/target_rri")).all()
    assert not reader.array("candidates/q_train_mask").any()
    assert np.isnan(reader.array("q_h/one_step_target_rri")).all()
    assert not reader.array("q_h/q_train_mask").any()


def test_rollout_zarr_masks_invalid_candidate_oracle_labels(tmp_path) -> None:
    traces = build_synthetic_rollout_traces(horizon=1, num_samples=6, seed=12)[:1]
    step = traces[0].steps[0]
    invalid_shell_index = 0 if int(step.selected_shell_index) != 0 else 1
    step.candidate_valid[invalid_shell_index] = False
    step.metric_vectors["target_rri"] = torch.arange(step.candidate_valid.shape[0], dtype=torch.float32)

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", traces)
    reader = RolloutZarrStoreReader(result.store_dir)

    candidate_valid = reader.array("candidates/candidate_valid_mask")
    assert not candidate_valid[invalid_shell_index]
    assert np.isnan(reader.array("candidates/target_rri")[invalid_shell_index])
    assert not reader.array("candidates/q_train_mask")[invalid_shell_index]
    assert np.isnan(reader.array("q_h/one_step_target_rri")[0, invalid_shell_index])
    assert not reader.array("q_h/q_train_mask")[0, invalid_shell_index]


def test_rollout_zarr_preserves_multi_target_identity_in_qh_view(tmp_path) -> None:
    traces = build_synthetic_rollout_traces(horizon=1, num_samples=6, seed=13)[:2]
    traces[0].lineage.target_row_id = 7
    traces[0].lineage.target_id = "target-a"
    traces[1].lineage.target_row_id = 9
    traces[1].lineage.target_id = "target-b"

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", traces)
    reader = RolloutZarrStoreReader(result.store_dir)

    target_rows = reader.array("targets/target_row_id")
    target_names = _json_list(reader, "dictionaries/target")
    target_name_ids = reader.array("targets/target_id")

    assert set(target_rows.tolist()) == {7, 9}
    assert {target_names[int(index)] for index in target_name_ids.tolist()} == {"target-a", "target-b"}
    assert set(reader.array("rollouts/target_row_id").tolist()) == {7, 9}
    assert set(reader.array("q_h/target_row_id").tolist()) == {7, 9}


def test_rollout_zarr_relative_pose_root_is_pose_transform(tmp_path) -> None:
    traces = build_synthetic_rollout_traces(horizon=1, num_samples=6, seed=17)[:1]
    traces[0].root_pose_world = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
        dtype=torch.float32,
    )

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", traces)
    reader = RolloutZarrStoreReader(result.store_dir)

    pose = reader.array("candidates/pose_world_cam")[0]
    relative = reader.array("candidates/pose_relative_root")[0]
    expected = PoseTW(traces[0].root_pose_world).inverse().compose(PoseTW(torch.as_tensor(pose))).tensor().numpy()

    assert np.allclose(relative, expected, atol=1e-5)
    assert not np.allclose(relative, pose - traces[0].root_pose_world.numpy(), atol=1e-5)


def test_rollout_zarr_records_per_rollout_lineage_and_split(tmp_path) -> None:
    traces = build_synthetic_rollout_traces(horizon=1, num_samples=6, seed=19)[:1]
    lineage = traces[0].lineage
    lineage.split = "train"
    lineage.candidate_config_hash = "candidate-cfg"
    lineage.oracle_config_hash = "oracle-cfg"
    lineage.rollout_config_hash = "rollout-cfg"
    lineage.model_checkpoint_hash = "model-ckpt"
    lineage.source_cache_version = "source-cache-v2"
    lineage.source_offline_store_manifest_hash = "source-manifest"
    lineage.split_manifest_hash = "split-manifest"
    lineage.branch_schedule_id = "branch-schedule"
    lineage.target_protocol_version = "v1-observed"
    lineage.reason_code_version = INVALID_REASON_VERSION
    lineage.selection_rng_state_hash = "rng-state"

    result = write_rollout_zarr_store(
        tmp_path / "rollouts.zarr",
        traces,
        target_protocol_version="v1-observed",
        source_offline_store_version="vin-offline-v1",
        split_manifest_hash="split-manifest",
    )
    reader = RolloutZarrStoreReader(result.store_dir)

    split_names = _json_list(reader, "dictionaries/split")
    assert split_names[int(reader.array("splits/rollout_split_id")[0])] == "train"
    assert split_names[int(reader.array("rollouts/split_id")[0])] == "train"
    for name in (
        "candidate_config_id",
        "oracle_config_id",
        "rollout_config_id",
        "model_checkpoint_id",
        "source_cache_version_id",
        "source_offline_store_manifest_hash_id",
        "split_manifest_hash_id",
        "branch_schedule_id",
        "target_protocol_version_id",
        "reason_code_version_id",
        "selection_rng_state_hash_id",
    ):
        assert int(reader.array(f"lineage/{name}")[0]) >= 0
    validation = validate_rollout_zarr_store(result.store_dir)
    assert validation.ok, validation.errors


def test_rollout_zarr_rejects_non_synthetic_store_with_synthetic_lineage(tmp_path) -> None:
    traces = build_synthetic_rollout_traces(horizon=1, num_samples=6, seed=23)[:1]

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", traces, target_protocol_version="v1-observed")

    validation = validate_rollout_zarr_store(result.store_dir)
    assert not validation.ok
    assert any("source_offline_store_version" in error for error in validation.errors)
    assert any("target_protocol_version_id" in error for error in validation.errors)
