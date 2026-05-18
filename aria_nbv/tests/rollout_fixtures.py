"""Test-only rollout Zarr record fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import trimesh  # type: ignore[import-untyped]
from efm3d.aria import CameraTW, PoseTW

from aria_nbv.data_handling import TARGET_INVALID_REASON_VERSION
from aria_nbv.pose_generation.candidate_generation import CandidateViewGeneratorConfig
from aria_nbv.pose_generation.counterfactuals import (
    CounterfactualCandidateEvaluation,
    CounterfactualMetricBundle,
    CounterfactualPoseGenerator,
    CounterfactualPoseGeneratorConfig,
    CounterfactualRolloutResult,
    CounterfactualSelectionPolicy,
    CounterfactualTrajectory,
)
from aria_nbv.pose_generation.target_counterfactuals import TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1
from aria_nbv.pose_generation.types import SamplingStrategy
from aria_nbv.rollouts import INVALID_REASON_VERSION, RolloutLineage, RolloutZarrRecord
from aria_nbv.utils.fingerprints import stable_config_hash

if TYPE_CHECKING:
    from typing import Any

    from aria_nbv.utils import BaseConfig


def build_rollout_records(
    *,
    horizon: int = 2,
    num_samples: int = 8,
    seed: int = 0,
) -> list[RolloutZarrRecord]:
    """Build real-looking fixture rollout records for store tests."""

    records: list[RolloutZarrRecord] = []
    for source_row_id, policy in enumerate(
        (
            CounterfactualSelectionPolicy.ORACLE_GREEDY,
            CounterfactualSelectionPolicy.RANDOM_VALID,
            CounterfactualSelectionPolicy.TEMPERATURE_SOFTMAX,
        )
    ):
        cfg = CounterfactualPoseGeneratorConfig(
            candidate_config=CandidateViewGeneratorConfig(
                num_samples=num_samples,
                min_radius=0.5,
                max_radius=0.5,
                ensure_collision_free=False,
                ensure_free_space=False,
                min_distance_to_mesh=0.0,
                sampling_strategy=SamplingStrategy.UNIFORM_SPHERE,
                view_max_azimuth_deg=0.0,
                view_max_elevation_deg=0.0,
                device="cpu",
                seed=seed,
                verbosity=0,
                is_debug=True,
            ),
            horizon=horizon,
            branch_factor=1,
            selection_policy=policy,
            selection_temperature=1.0,
            seed=seed,
            verbosity=0,
        )
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        result = CounterfactualPoseGenerator(cfg).generate(
            reference_pose=_identity_pose(),
            gt_mesh=mesh,
            mesh_verts=torch.as_tensor(mesh.vertices, dtype=torch.float32),
            mesh_faces=torch.as_tensor(mesh.faces, dtype=torch.int64),
            camera_calib_template=_dummy_camera(),
            occupancy_extent=torch.tensor([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=torch.float32),
            score_candidates=_fixture_scores,
        )
        _attach_fixture_candidate_provenance(result)
        records.append(
            RolloutZarrRecord(
                result=result,
                rollout_id_prefix=f"fixture-{policy.value}",
                lineage=RolloutLineage(
                    scene_id="fixture_box",
                    snippet_id="smoke",
                    mesh_version="fixture-mesh-v1",
                    candidate_config_hash=_config_hash(cfg.candidate_config),
                    oracle_config_hash="fixture-oracle",
                    rollout_config_hash=_config_hash(cfg),
                    branch_schedule_id=cfg.branch_schedule_id,
                    random_seed=seed,
                    source_cache_version="7",
                    source_row_id=source_row_id,
                    source_sample_index=source_row_id,
                    source_sample_key=f"fixture:smoke:{source_row_id}",
                    split="train",
                    source_shard_id="vin-shard-000000",
                    source_shard_row=source_row_id,
                    source_offline_store_manifest_hash="fixture-source-manifest",
                    split_manifest_hash="fixture-split-manifest",
                    selection_rng_state_hash="fixture-rng",
                    target_protocol_version="v1-observed",
                    target_crop_policy=TARGET_CROP_POLICY_GT_OBB_ORIENTED_ANY_VERTEX_V1,
                    reason_code_version=INVALID_REASON_VERSION,
                    target_row_id=source_row_id,
                    target_id=f"fixture-target-{source_row_id}",
                    target_selection_policy="fixture_top_k",
                    target_selection_rank=source_row_id,
                    target_selection_score=1.0 - 0.1 * source_row_id,
                    target_invalid_reason_bitset=1,
                    target_primary_invalid_reason=0,
                    target_reason_code_version=TARGET_INVALID_REASON_VERSION,
                    matched_gt_target_row_id=100 + source_row_id,
                    matched_gt_target_id=f"fixture-gt-target-{source_row_id}",
                    gt_match_iou=0.9,
                    gt_match_score=0.9,
                    gt_match_status="matched",
                ),
            )
        )
    return records


def _attach_fixture_candidate_provenance(result: CounterfactualRolloutResult) -> None:
    for trajectory in result.trajectories:
        for step in trajectory.steps:
            mask = step.candidates.mask_valid.detach().cpu().reshape(-1)
            n = int(mask.shape[0])
            step.candidates.strategy_id = torch.arange(n, dtype=torch.int64) % 4
            step.candidates.mixture_id = torch.arange(n, dtype=torch.int64) % 2
            step.candidates.sampler_probability = torch.full((n,), 1.0 / float(max(n, 1)), dtype=torch.float32)
            step.selected_depth_m = torch.full((240, 240), 1.0 + float(step.step_index), dtype=torch.float32)
            step.selected_depth_valid_mask = torch.ones((240, 240), dtype=torch.bool)
            step.selected_depth_focal_px = (120.0, 120.0)
            step.selected_depth_principal_point_px = (120.0, 120.0)
            step.selected_depth_image_size_hw = (240, 240)


def _config_hash(config: BaseConfig) -> str:
    return stable_config_hash(config)


def _identity_pose() -> PoseTW:
    return PoseTW(
        torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
    )


def _dummy_camera() -> CameraTW:
    return CameraTW.from_surreal(
        width=torch.tensor([64.0]),
        height=torch.tensor([64.0]),
        type_str="Pinhole",
        params=torch.tensor([[60.0, 60.0, 32.0, 32.0]]),
        gain=torch.zeros(1),
        exposure_s=torch.zeros(1),
        valid_radius=torch.tensor([64.0]),
        T_camera_rig=PoseTW.from_matrix3x4(torch.eye(3, 4).unsqueeze(0)),
    )


def _fixture_scores(
    result: Any,
    trajectory: CounterfactualTrajectory,
    step_index: int,
) -> CounterfactualCandidateEvaluation:
    del trajectory
    valid_poses = result.poses_world_cam()
    centers = valid_poses.t.reshape(-1, 3)
    scores = torch.linspace(0.1, 0.1 * centers.shape[0], centers.shape[0], device=centers.device)
    scores = scores + float(step_index)
    return CounterfactualCandidateEvaluation(
        scores=scores,
        score_label="oracle_rri",
        metrics=CounterfactualMetricBundle(rri=scores, scene_rri=scores, target_rri=scores),
    )
