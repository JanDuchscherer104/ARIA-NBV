"""Integration test for oracle cache round-trip on real data."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch


def _skip_if_missing_real_data(scene_id: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / ".configs" / "evl_inf_desktop.yaml"
    ckpt_path = repo_root / ".logs" / "ckpts" / "model_lite.pth"
    data_dir = repo_root / ".data" / "ase_efm" / scene_id
    mesh_path = repo_root / ".data" / "ase_meshes" / f"scene_ply_{scene_id}.ply"

    if not cfg_path.exists():
        pytest.skip("Missing EVL cfg file")
    if not ckpt_path.exists():
        pytest.skip("Missing EVL checkpoint")
    if not data_dir.exists():
        pytest.skip("Missing ASE sample shards")
    if not any(data_dir.glob("*.tar")):
        pytest.skip("Missing ASE tar shards for scene")
    if not mesh_path.exists():
        pytest.skip("Missing GT mesh for scene")


def test_oracle_cache_roundtrip(tmp_path: Path) -> None:  # noqa: PLR0915
    """Round-trip cache build and load for a single real snippet."""
    if importlib.util.find_spec("power_spherical") is None:
        pytest.skip("Missing power_spherical dependency")

    data_mod = importlib.import_module("aria_nbv.data")
    offline_mod = importlib.import_module("aria_nbv.data.offline_cache")
    labeler_mod = importlib.import_module("aria_nbv.pipelines.oracle_rri_labeler")
    pose_mod = importlib.import_module(
        "aria_nbv.pose_generation.candidate_generation",
    )
    depth_mod = importlib.import_module("aria_nbv.rendering.candidate_depth_renderer")
    p3d_mod = importlib.import_module("aria_nbv.rendering.pytorch3d_depth_renderer")
    utils_mod = importlib.import_module("aria_nbv.utils")
    backbone_mod = importlib.import_module("aria_nbv.vin.backbone_evl")

    AseEfmDatasetConfig = data_mod.AseEfmDatasetConfig  # noqa: N806
    OracleRriCacheConfig = offline_mod.OracleRriCacheConfig  # noqa: N806
    OracleRriCacheDatasetConfig = offline_mod.OracleRriCacheDatasetConfig  # noqa: N806
    OracleRriCacheWriterConfig = offline_mod.OracleRriCacheWriterConfig  # noqa: N806
    OracleRriLabelerConfig = labeler_mod.OracleRriLabelerConfig  # noqa: N806
    CandidateViewGeneratorConfig = pose_mod.CandidateViewGeneratorConfig  # noqa: N806
    CandidateDepthRendererConfig = depth_mod.CandidateDepthRendererConfig  # noqa: N806
    Pytorch3DDepthRendererConfig = p3d_mod.Pytorch3DDepthRendererConfig  # noqa: N806
    Verbosity = utils_mod.Verbosity  # noqa: N806
    EvlBackboneConfig = backbone_mod.EvlBackboneConfig  # noqa: N806
    # xFormers is optional in test environments; this cache test only needs CPU decode paths.
    scene_id = "81286"
    _skip_if_missing_real_data(scene_id)

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / ".configs" / "evl_inf_desktop.yaml"
    ckpt_path = repo_root / ".logs" / "ckpts" / "model_lite.pth"

    dataset_cfg = AseEfmDatasetConfig(
        scene_ids=[scene_id],
        batch_size=1,
        load_meshes=True,
        require_mesh=True,
        device="cpu",
        wds_shuffle=False,
    )

    generator_cfg = CandidateViewGeneratorConfig(
        device="cpu",
        num_samples=4,
        oversample_factor=1.0,
        max_resamples=0,
        ensure_collision_free=False,
        ensure_free_space=False,
        min_distance_to_mesh=0.1,
    )

    depth_cfg = CandidateDepthRendererConfig(
        device="cpu",
        max_candidates_final=4,
        oversample_factor=1.0,
        renderer=Pytorch3DDepthRendererConfig(device="cpu", bin_size=0),
        verbosity=Verbosity.QUIET,
    )

    labeler_cfg = OracleRriLabelerConfig(
        device=torch.device("cpu"),
        generator=generator_cfg,
        depth=depth_cfg,
        backprojection_stride=4,
        verbosity=Verbosity.QUIET,
    )

    backbone_cfg = EvlBackboneConfig(
        model_cfg=cfg_path,
        model_ckpt=ckpt_path,
        device=torch.device("cpu"),
        freeze=True,
        features_mode="heads",
    )

    cache_cfg = OracleRriCacheConfig(cache_dir=tmp_path / "oracle_cache")
    writer_cfg = OracleRriCacheWriterConfig(
        cache=cache_cfg,
        dataset=dataset_cfg,
        labeler=labeler_cfg,
        backbone=backbone_cfg,
        max_samples=1,
        include_backbone=True,
        include_depths=True,
        include_pointclouds=True,
        overwrite=True,
        verbosity=Verbosity.QUIET,
    )

    writer = writer_cfg.setup_target()
    entries = writer.run()
    assert len(entries) == 1  # noqa: S101

    cache_ds_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        load_backbone=True,
    )
    cache_ds = cache_ds_cfg.setup_target()
    sample = cache_ds[0]

    assert sample.backbone_out is not None  # noqa: S101
    assert sample.rri.rri.shape[0] == sample.depths.depths.shape[0]  # noqa: S101
    assert sample.candidate_pcs.points.shape[0] == sample.depths.depths.shape[0]  # noqa: S101

    # Re-running the writer should skip the already cached sample.
    entries_again = writer_cfg.setup_target().run()
    assert entries_again == []  # noqa: S101
    index_entries = [
        json.loads(line)
        for line in cache_cfg.index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(index_entries) == 1  # noqa: S101
    meta_payload = json.loads(cache_cfg.metadata_path.read_text(encoding="utf-8"))
    assert meta_payload["num_samples"] == 1  # noqa: S101
