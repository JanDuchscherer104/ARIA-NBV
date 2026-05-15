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
    TARGET_INVALID_REASON_VERSION,
)
from aria_nbv.pose_generation.target_counterfactuals import TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1
from aria_nbv.rollouts import (
    INVALID_REASON_VERSION,
    ROLLOUT_MANIFEST_FILENAME,
    ROLLOUT_ZARR_SCHEMA_VERSION,
    RolloutStoreManifestContext,
    RolloutZarrStoreConfig,
    RolloutZarrStoreReader,
    validate_rollout_zarr_store,
    write_rollout_zarr_store,
)
from aria_nbv.rollouts.info_cli import main as rollouts_info_main
from tests.rollout_fixtures import build_rollout_records


def _json_list(reader: RolloutZarrStoreReader, path: str) -> list[str]:
    return json.loads(bytes(reader.array(path).tolist()).decode("utf-8"))


def _steps(record):
    return record.result.trajectories[0].steps


def test_rollout_zarr_store_writes_reads_and_validates_records(tmp_path) -> None:
    records = build_rollout_records(horizon=2, num_samples=8, seed=7)
    result = write_rollout_zarr_store(
        tmp_path / "rollouts.zarr",
        records,
        discount_gamma=0.95,
        target_protocol_version="v1-observed",
        source_offline_store_version="7",
        split_manifest_hash="fixture-split-manifest",
    )

    assert result.num_rollouts == 3
    assert result.num_steps == 6
    assert result.num_candidates > 0

    validation = validate_rollout_zarr_store(result.store_dir)
    assert validation.ok, validation.errors

    reader = RolloutZarrStoreReader(result.store_dir)
    assert reader.root.attrs["schema_id"] == "aria_nbv.rollout_zarr_q_invalidity"
    assert reader.root.attrs["schema_version"] == ROLLOUT_ZARR_SCHEMA_VERSION
    assert reader.root.attrs["return_semantics"] == "cumulative_target_rri"
    assert reader.root.attrs["manifest_path"] == ROLLOUT_MANIFEST_FILENAME
    assert result.manifest_path.exists()
    assert result.manifest_sha256 == reader.root.attrs["manifest_sha256"]
    assert "writer_config" not in reader.root.attrs
    assert "generation_manifest_json" not in reader.root["metadata"]
    manifest_bundle = reader.manifest()
    manifest = manifest_bundle["manifest"]
    assert manifest["schema_version"] == ROLLOUT_ZARR_SCHEMA_VERSION
    assert manifest["counts"]["rollouts"] == result.num_rollouts
    assert manifest["counts"]["steps"] == result.num_steps
    assert manifest["counts"]["candidates"] == result.num_candidates
    assert manifest["generation"]["invocation"]["mode"] == "programmatic"
    assert manifest["source_coverage"]["scene_counts"] == {"fixture_box": 3}
    assert "splits" not in reader.root
    assert "sources" in reader.root
    assert "q_h" not in reader.root
    assert set(reader.array("sources/source_row_id").tolist()) == {0, 1, 2}
    assert set(reader.array("rollouts/source_row_id").tolist()) == {0, 1, 2}
    root_pose = reader.array("rollouts/root_pose_world")
    assert root_pose.shape == (result.num_rollouts, 12)
    assert np.isfinite(root_pose).all()

    candidate_valid = reader.array("candidates/candidate_valid_mask")
    selection_probabilities = reader.array("candidates/selection_probabilities")
    assert np.all(selection_probabilities[~candidate_valid] == 0.0)
    assert np.all(reader.array("candidates/selected_mask") <= candidate_valid)

    assert "q_h" not in reader.root
    q_h = reader.q_h_view()
    q_candidate_row_id = q_h["candidate_row_id"]
    valid_action_mask = q_h["valid_action_mask"]
    q_train_mask = q_h["q_train_mask"]

    assert q_candidate_row_id.shape == valid_action_mask.shape
    assert np.all(q_train_mask <= valid_action_mask)
    assert "q_target_target_rri" not in q_h
    assert np.all(~valid_action_mask[q_candidate_row_id < 0])
    assert np.allclose(q_h["discount"][:2], np.asarray([1.0, 0.95], dtype=np.float32))


def test_rollout_zarr_selected_action_td_fields_align_with_step_rows(tmp_path) -> None:
    records = build_rollout_records(horizon=2, num_samples=6, seed=3)
    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records)
    reader = RolloutZarrStoreReader(result.store_dir)
    q_h = reader.q_h_view()

    step_selected = reader.array("steps/selected_candidate_row_id")
    td_selected = q_h["td_selected_candidate_row_id"]
    td_next = q_h["td_next_step_row_id"]
    td_terminal = q_h["td_terminal_mask"]

    assert np.array_equal(td_selected, step_selected)
    assert td_next.shape == td_terminal.shape == td_selected.shape
    assert np.any(~td_terminal)
    assert np.any(td_terminal)


def test_rollout_zarr_requires_explicit_target_rri_for_q_training(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=11)[:1]
    for step in _steps(records[0]):
        step.metric_vectors = {}
        step.selected_metrics = {}

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records)
    reader = RolloutZarrStoreReader(result.store_dir)
    q_h = reader.q_h_view()

    assert np.isfinite(reader.array("candidates/selection_logits")).any()
    assert np.isnan(reader.array("candidates/target_rri")).all()
    assert not reader.array("candidates/q_train_mask").any()
    assert np.isnan(q_h["one_step_target_rri"]).all()
    assert not q_h["q_train_mask"].any()


def test_rollout_zarr_never_backfills_scene_rri_from_generic_rri(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=10)[:1]
    for step in _steps(records[0]):
        generic = torch.arange(step.candidates.mask_valid.shape[0], dtype=torch.float32)
        step.metric_vectors = {"rri": generic, "target_rri": generic}
        step.selected_metrics = {"rri": 1.0, "target_rri": 1.0}

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records)
    reader = RolloutZarrStoreReader(result.store_dir)
    q_h = reader.q_h_view()

    assert np.isfinite(reader.array("candidates/target_rri")).any()
    assert np.isnan(reader.array("candidates/scene_rri")).all()
    assert np.isnan(q_h["one_step_scene_rri"]).all()


def test_rollout_zarr_masks_invalid_candidate_oracle_labels(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=12)[:1]
    step = _steps(records[0])[0]
    invalid_shell_index = 0 if int(step.selected_shell_index) != 0 else 1
    step.candidates.mask_valid[invalid_shell_index] = False
    step.metric_vectors["target_rri"] = torch.arange(step.candidates.mask_valid.shape[0], dtype=torch.float32)

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records)
    reader = RolloutZarrStoreReader(result.store_dir)
    q_h = reader.q_h_view()

    candidate_valid = reader.array("candidates/candidate_valid_mask")
    assert not candidate_valid[invalid_shell_index]
    assert np.isnan(reader.array("candidates/target_rri")[invalid_shell_index])
    assert not reader.array("candidates/q_train_mask")[invalid_shell_index]
    assert np.isnan(q_h["one_step_target_rri"][0, invalid_shell_index])
    assert not q_h["q_train_mask"][0, invalid_shell_index]


def test_rollout_zarr_preserves_multi_target_identity_in_qh_view(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=13)[:2]
    records[0].lineage.target_row_id = 7
    records[0].lineage.target_id = "target-a"
    records[0].lineage.target_selection_policy = "greedy_top_k"
    records[0].lineage.target_selection_rank = 0
    records[0].lineage.target_selection_score = 0.75
    records[0].lineage.target_invalid_reason_bitset = 1
    records[0].lineage.target_primary_invalid_reason = 0
    records[0].lineage.target_reason_code_version = TARGET_INVALID_REASON_VERSION
    records[0].lineage.matched_gt_target_row_id = 70
    records[0].lineage.matched_gt_target_id = "gt-target-a"
    records[0].lineage.gt_match_iou = 0.8
    records[0].lineage.gt_match_score = 0.8
    records[0].lineage.gt_match_status = "matched"
    records[1].lineage.target_row_id = 9
    records[1].lineage.target_id = "target-b"
    records[1].lineage.target_selection_policy = "greedy_top_k"
    records[1].lineage.target_selection_rank = 1
    records[1].lineage.target_selection_score = 0.5
    records[1].lineage.target_invalid_reason_bitset = 1
    records[1].lineage.target_primary_invalid_reason = 0
    records[1].lineage.target_reason_code_version = TARGET_INVALID_REASON_VERSION
    records[1].lineage.matched_gt_target_row_id = None
    records[1].lineage.matched_gt_target_id = None
    records[1].lineage.gt_match_status = "unmatched_gt"

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records)
    reader = RolloutZarrStoreReader(result.store_dir)
    q_h = reader.q_h_view()

    target_rows = reader.array("targets/target_row_id")
    target_names = _json_list(reader, "dictionaries/target")
    target_name_ids = reader.array("targets/target_id")

    assert set(target_rows.tolist()) == {7, 9}
    assert {target_names[int(index)] for index in target_name_ids.tolist()} == {"target-a", "target-b"}
    assert set(reader.array("rollouts/target_row_id").tolist()) == {7, 9}
    assert set(q_h["target_row_id"].tolist()) == {7, 9}
    assert reader.array("targets/target_selection_rank").tolist() == [0, 1]
    assert np.allclose(reader.array("targets/target_selection_score"), np.asarray([0.75, 0.5], dtype=np.float32))
    assert reader.array("targets/matched_gt_target_row_id").tolist() == [70, -1]
    assert reader.array("targets/gt_label_valid_mask").tolist() == [True, False]
    config_names = _json_list(reader, "dictionaries/config")
    reason_version_ids = reader.array("targets/target_reason_code_version_id")
    assert {config_names[int(index)] for index in reason_version_ids.tolist()} == {TARGET_INVALID_REASON_VERSION}
    match_status = _json_list(reader, "dictionaries/target_match_status")
    status_ids = reader.array("targets/gt_match_status_id")
    assert [match_status[int(index)] for index in status_ids.tolist()] == ["matched", "unmatched_gt"]


def test_rollout_zarr_relative_pose_root_is_pose_transform(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=17)[:1]
    records[0].result.root_pose_world = PoseTW(
        torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
            dtype=torch.float32,
        )
    )

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records)
    reader = RolloutZarrStoreReader(result.store_dir)

    pose = reader.array("candidates/pose_world_cam")[0]
    relative = reader.array("candidates/pose_relative_root")[0]
    stored_root = reader.array("rollouts/root_pose_world")[0]
    root_pose = records[0].result.root_pose_world.tensor().numpy()
    expected = records[0].result.root_pose_world.inverse().compose(PoseTW(torch.as_tensor(pose))).tensor().numpy()

    assert np.allclose(stored_root, root_pose, atol=1e-5)
    assert np.allclose(relative, expected, atol=1e-5)
    assert not np.allclose(relative, pose - root_pose, atol=1e-5)


def test_rollout_zarr_records_per_rollout_lineage_and_split(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=19)[:1]
    lineage = records[0].lineage
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
    lineage.target_crop_policy = TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1
    lineage.reason_code_version = INVALID_REASON_VERSION
    lineage.selection_rng_state_hash = "rng-state"
    lineage.target_row_id = 5
    lineage.target_id = "target"
    lineage.target_selection_policy = "greedy_top_k"
    lineage.target_invalid_reason_bitset = 1
    lineage.target_primary_invalid_reason = 0
    lineage.target_reason_code_version = TARGET_INVALID_REASON_VERSION
    lineage.matched_gt_target_row_id = 50
    lineage.matched_gt_target_id = "gt-target"
    lineage.gt_match_status = "matched"
    for step in _steps(records[0]):
        n = int(step.candidates.mask_valid.shape[0])
        step.candidates.strategy_id = torch.arange(n, dtype=torch.int64) % 4
        step.candidates.mixture_id = torch.arange(n, dtype=torch.int64) % 2
        step.candidates.sampler_probability = torch.full((n,), 1.0 / float(n), dtype=torch.float32)

    result = write_rollout_zarr_store(
        tmp_path / "rollouts.zarr",
        records,
        target_protocol_version="v1-observed",
        source_offline_store_version="vin-offline-v1",
        split_manifest_hash="split-manifest",
    )
    reader = RolloutZarrStoreReader(result.store_dir)

    split_names = _json_list(reader, "dictionaries/split")
    assert "splits" not in reader.root
    assert split_names[int(reader.array("rollouts/split_id")[0])] == "train"
    assert np.array_equal(reader.array("rollouts/source_row_id"), reader.array("sources/source_row_id"))
    assert np.array_equal(reader.array("lineage/rollout_row_id"), reader.array("rollouts/rollout_row_id"))
    for name in (
        "candidate_config_id",
        "oracle_config_id",
        "rollout_config_id",
        "model_checkpoint_id",
        "branch_schedule_id",
        "target_protocol_version_id",
        "target_crop_policy_id",
        "reason_code_version_id",
        "selection_rng_state_hash_id",
    ):
        assert int(reader.array(f"lineage/{name}")[0]) >= 0
    for name in ("source_cache_version_id", "source_offline_store_manifest_hash_id", "split_manifest_hash_id"):
        assert int(reader.array(f"sources/{name}")[0]) >= 0
    for rollout_owned_name in ("root_pose_world", "policy_id", "target_row_id"):
        assert rollout_owned_name not in reader.root["lineage"]
    for target_owned_name in ("target_selection_policy_id", "matched_gt_target_row_id", "gt_match_status_id"):
        assert target_owned_name not in reader.root["lineage"]
        assert target_owned_name in reader.root["targets"]
    validation = validate_rollout_zarr_store(result.store_dir)
    assert validation.ok, validation.errors


def test_rollout_zarr_rejects_mixed_split_shards(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=20)[:2]
    records[0].lineage.split = "train"
    records[1].lineage.split = "val"

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records)

    validation = validate_rollout_zarr_store(result.store_dir)
    assert not validation.ok
    assert any("exactly one split" in error for error in validation.errors)


def test_rollout_zarr_preserves_candidate_mixture_provenance_for_real_stores(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=21)[:1]
    lineage = records[0].lineage
    lineage.split = "train"
    lineage.candidate_config_hash = "candidate-cfg"
    lineage.oracle_config_hash = "oracle-cfg"
    lineage.rollout_config_hash = "rollout-cfg"
    lineage.source_cache_version = "source-cache-v7"
    lineage.source_offline_store_manifest_hash = "source-manifest"
    lineage.split_manifest_hash = "split-manifest"
    lineage.target_protocol_version = "v1-observed"
    lineage.target_crop_policy = TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1
    lineage.reason_code_version = INVALID_REASON_VERSION
    lineage.selection_rng_state_hash = "rng-state"
    lineage.target_row_id = 3
    lineage.target_id = "target"
    lineage.target_selection_policy = "greedy_top_k"
    lineage.target_invalid_reason_bitset = 1
    lineage.target_primary_invalid_reason = 0
    lineage.target_reason_code_version = TARGET_INVALID_REASON_VERSION
    lineage.matched_gt_target_row_id = 30
    lineage.matched_gt_target_id = "gt-target"
    lineage.gt_match_status = "matched"
    for step in _steps(records[0]):
        n = int(step.candidates.mask_valid.shape[0])
        step.candidates.strategy_id = torch.arange(n, dtype=torch.int64) % 4
        step.candidates.mixture_id = torch.arange(n, dtype=torch.int64) % 2
        step.candidates.sampler_probability = torch.full((n,), 1.0 / float(n), dtype=torch.float32)

    result = write_rollout_zarr_store(
        tmp_path / "rollouts.zarr",
        records,
        target_protocol_version="v1-observed",
        source_offline_store_version="7",
        split_manifest_hash="split-manifest",
    )
    reader = RolloutZarrStoreReader(result.store_dir)

    actor_rows = reader.array("candidates/actor_action_mask")
    assert np.all(reader.array("candidates/strategy_id")[actor_rows] >= 0)
    assert np.all(reader.array("candidates/mixture_id")[actor_rows] >= 0)
    assert np.isfinite(reader.array("candidates/sampler_probability")[actor_rows]).all()

    validation = validate_rollout_zarr_store(result.store_dir)
    assert validation.ok, validation.errors


def test_rollout_zarr_blocks_q_training_for_target_invalid_records(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=25)[:1]
    records[0].lineage.target_protocol_version = "v1-observed"
    records[0].lineage.target_row_id = 11
    records[0].lineage.target_id = "target-invalid"
    records[0].lineage.target_invalid_reason_bitset = 1 << 10
    records[0].lineage.target_primary_invalid_reason = 10
    records[0].lineage.gt_match_status = "unmatched_gt"

    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records, target_protocol_version="v1-observed")
    reader = RolloutZarrStoreReader(result.store_dir)
    q_h = reader.q_h_view()

    assert reader.array("candidates/oracle_label_mask").any()
    assert not reader.array("candidates/q_train_mask").any()
    assert not q_h["q_train_mask"].any()


def test_rollout_zarr_rejects_missing_root_lineage(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=23)[:1]

    result = write_rollout_zarr_store(
        tmp_path / "rollouts.zarr",
        records,
        target_protocol_version="v1-observed",
        source_offline_store_version="",
    )

    validation = validate_rollout_zarr_store(result.store_dir)
    assert not validation.ok
    assert any("source_offline_store_version" in error for error in validation.errors)


def test_rollout_zarr_validation_requires_top_level_manifest(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=29)[:1]
    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records)
    result.manifest_path.unlink()

    validation = validate_rollout_zarr_store(result.store_dir)
    assert not validation.ok
    assert any("manifest" in error for error in validation.errors)


def test_rollout_zarr_validation_rejects_manifest_hash_mismatch(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=31)[:1]
    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records)
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    payload["counts"]["rollouts"] = 999
    result.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    validation = validate_rollout_zarr_store(result.store_dir)
    assert not validation.ok
    assert any("manifest hash" in error for error in validation.errors)


def test_rollouts_info_cli_prints_manifest_without_validation(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=37)[:1]
    result = write_rollout_zarr_store(tmp_path / "rollouts.zarr", records)
    capsys.readouterr()

    rollouts_info_main(["--store", str(result.store_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["schema_version"] == ROLLOUT_ZARR_SCHEMA_VERSION
    assert payload["manifest"]["source_coverage"]["num_source_rows"] == 1
    assert "validation" not in payload


def test_rollout_manifest_preserves_cli_toml_and_resolved_config(tmp_path) -> None:
    records = build_rollout_records(horizon=1, num_samples=6, seed=41)[:1]
    config_path = tmp_path / "build_rollouts.toml"
    raw_toml = "max_samples = 1\n"
    config_path.write_text(raw_toml, encoding="utf-8")
    writer_config = RolloutZarrStoreConfig(store_dir=tmp_path / "rollouts.zarr")

    result = write_rollout_zarr_store(
        tmp_path / "rollouts.zarr",
        records,
        manifest_context=RolloutStoreManifestContext.from_cli(
            writer_config=writer_config,
            argv=["nbv-build-rollouts", "--config-path", str(config_path)],
            config_path=config_path,
        ),
    )
    manifest = RolloutZarrStoreReader(result.store_dir).manifest()["manifest"]

    invocation = manifest["generation"]["invocation"]
    assert invocation["mode"] == "cli"
    assert invocation["raw_toml_text"] == raw_toml
    assert invocation["raw_toml_sha256"]
    assert manifest["generation"]["writer_config"]["store_dir"].endswith("rollouts.zarr")
    assert manifest["generation"]["runtime"]["git"]["available"] in {True, False}
