"""Worker entrypoint for Streamlit stage subprocess execution."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ..data_handling import AseEfmDatasetConfig
from ..pipelines import OracleRriLabelerConfig
from ..pose_generation.types import CandidateSamplingResult
from ..rendering.candidate_pointclouds import CandidatePointCloudBuilderConfig, CandidatePointClouds


def _iterate_sample(dataset_payload: dict[str, Any], sample_idx: int) -> Any:
    dataset_cfg = AseEfmDatasetConfig.model_validate(dataset_payload)
    dataset = dataset_cfg.setup_target()
    iterator = iter(dataset)
    sample = None
    for _ in range(int(sample_idx) + 1):
        sample = next(iterator)
    return sample


def _resolved_labeler(labeler_payload: dict[str, Any]) -> OracleRriLabelerConfig:
    cfg = OracleRriLabelerConfig.model_validate(labeler_payload)
    cfg.__pydantic_fields_set__.discard("device")
    cfg.generator.__pydantic_fields_set__.difference_update({"collision_backend", "device"})
    cfg.depth.__pydantic_fields_set__.difference_update({"backend", "device"})
    cfg.depth.pytorch3d.__pydantic_fields_set__.discard("device")
    cfg.depth.mojo.__pydantic_fields_set__.discard("device")
    cfg.pointcloud.__pydantic_fields_set__.discard("backend")
    cfg.oracle.__pydantic_fields_set__.discard("backend")
    return cfg.resolved(require_available=True)


def _generate_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    sample = _iterate_sample(payload["dataset_cfg"], int(payload["sample_idx"]))
    labeler_cfg = _resolved_labeler(payload["labeler_cfg"])
    candidates = labeler_cfg.generator.setup_target().generate_from_typed_sample(sample)
    return {"candidates": candidates.to_serializable()}


def _render_depths_and_pcs(payload: dict[str, Any]) -> dict[str, Any]:
    sample = _iterate_sample(payload["dataset_cfg"], int(payload["sample_idx"]))
    labeler_cfg = _resolved_labeler(payload["labeler_cfg"])
    candidates = CandidateSamplingResult.from_serializable(
        payload["candidates"],
        device=labeler_cfg.depth.device,
    )
    depths = labeler_cfg.depth.setup_target().render(sample=sample, candidates=candidates)
    pointcloud_cfg = CandidatePointCloudBuilderConfig.model_validate(
        labeler_cfg.pointcloud.model_dump(mode="python", round_trip=True),
    )
    pointcloud_cfg.backprojection_stride = int(payload["stride"])
    pcs = pointcloud_cfg.setup_target().build(sample, depths)
    return {
        "depths": depths.to_serializable(),
        "pcs": pcs.to_serializable(),
    }


def _score_rri(payload: dict[str, Any]) -> dict[str, Any]:
    sample = _iterate_sample(payload["dataset_cfg"], int(payload["sample_idx"]))
    if sample.mesh_verts is None or sample.mesh_faces is None:
        raise ValueError("Oracle RRI scoring requires mesh tensors on the sample.")
    labeler_cfg = _resolved_labeler(payload["labeler_cfg"])
    pcs = CandidatePointClouds.from_serializable(payload["pcs"], device=labeler_cfg.device)
    result = labeler_cfg.oracle.setup_target().score(
        points_t=pcs.semidense_points,
        points_q=pcs.points,
        lengths_q=pcs.lengths,
        gt_verts=sample.mesh_verts.to(device=pcs.points.device, dtype=pcs.points.dtype),
        gt_faces=sample.mesh_faces.to(device=pcs.points.device),
        extend=pcs.occupancy_bounds,
    )
    return {"rri": result.to_serializable()}


def _dispatch(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    if mode == "generate_candidates":
        return _generate_candidates(payload)
    if mode == "render_depths_and_pcs":
        return _render_depths_and_pcs(payload)
    if mode == "score_rri":
        return _score_rri(payload)
    raise ValueError(f"Unsupported stage subprocess mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = torch.load(Path(args.input), map_location="cpu", weights_only=False)
    result = _dispatch(args.mode, payload)
    torch.save(result, Path(args.output))


if __name__ == "__main__":
    main()
