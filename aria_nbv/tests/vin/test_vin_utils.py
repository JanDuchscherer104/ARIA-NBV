"""Tests for VIN diagnostics cache helpers."""

# ruff: noqa: S101

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from efm3d.aria import CameraTW, PoseTW

from oracle_rri.app.panels.vin_utils import (
    DEFAULT_BACKBONE_KEEP_FIELDS,
    _build_experiment_config,
    _has_backbone_obbs,
    _should_fetch_vin_snippet,
    _vin_oracle_batch_from_cache,
)
from oracle_rri.data.offline_cache_types import OracleRriCacheSample
from oracle_rri.rendering.candidate_depth_renderer import CandidateDepths
from oracle_rri.rendering.candidate_pointclouds import CandidatePointClouds
from oracle_rri.rri_metrics.types import RriResult
from oracle_rri.utils import Stage
from oracle_rri.vin.types import EvlBackboneOutput

try:
    from pytorch3d.renderer.cameras import PerspectiveCameras  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    PerspectiveCameras = None  # type: ignore[assignment]


def _make_camera(num: int = 1) -> CameraTW:
    width = torch.full((num,), 32.0)
    height = torch.full((num,), 32.0)
    fx = torch.full((num,), 16.0)
    fy = torch.full((num,), 16.0)
    cx = torch.full((num,), 16.0)
    cy = torch.full((num,), 16.0)
    t_cam = PoseTW.from_Rt(
        torch.eye(3).repeat(num, 1, 1),
        torch.zeros(num, 3),
    )
    return CameraTW.from_parameters(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        T_camera_rig=t_cam,
        dist_params=torch.zeros(0),
    )


def _make_backbone() -> EvlBackboneOutput:
    t_world_voxel = PoseTW.from_Rt(torch.eye(3).unsqueeze(0), torch.zeros(1, 3))
    voxel_extent = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float32)
    return EvlBackboneOutput(
        t_world_voxel=t_world_voxel,
        voxel_extent=voxel_extent,
        obbs_pr_nms="dummy",
    )


def _make_cache_sample() -> OracleRriCacheSample:
    poses = PoseTW.from_Rt(torch.eye(3).unsqueeze(0), torch.zeros(1, 3))
    ref_pose = PoseTW.from_Rt(torch.eye(3), torch.zeros(3))
    camera = _make_camera()
    if PerspectiveCameras is None:
        cams = SimpleNamespace()  # type: ignore[assignment]
    else:
        cams = PerspectiveCameras(
            R=torch.eye(3).unsqueeze(0),
            T=torch.zeros(1, 3),
            focal_length=torch.tensor([[16.0, 16.0]]),
            principal_point=torch.tensor([[16.0, 16.0]]),
            image_size=torch.tensor([[32.0, 32.0]]),
        )

    depths = CandidateDepths(
        depths=torch.zeros(1, 2, 2),
        depths_valid_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        poses=poses,
        reference_pose=ref_pose,
        candidate_indices=torch.tensor([0]),
        camera=camera,
        p3d_cameras=cams,  # type: ignore[arg-type]
    )

    pcs = CandidatePointClouds(
        points=torch.zeros(1, 1, 3),
        lengths=torch.tensor([1]),
        semidense_points=torch.zeros(1, 3),
        semidense_length=torch.tensor([1]),
        occupancy_bounds=torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
    )

    rri = RriResult(
        rri=torch.tensor([0.1]),
        pm_dist_before=torch.tensor([0.2]),
        pm_dist_after=torch.tensor([0.3]),
        pm_acc_before=torch.tensor([0.2]),
        pm_comp_before=torch.tensor([0.2]),
        pm_acc_after=torch.tensor([0.3]),
        pm_comp_after=torch.tensor([0.3]),
    )

    return OracleRriCacheSample(
        key="k",
        scene_id="scene",
        snippet_id="snippet",
        candidates=None,  # type: ignore[arg-type]
        depths=depths,
        candidate_pcs=pcs,
        rri=rri,
        backbone_out=_make_backbone(),
    )


def test_vin_utils_strip_backbone_obbs() -> None:
    cache_sample = _make_cache_sample()
    assert _has_backbone_obbs(cache_sample.backbone_out)

    batch = _vin_oracle_batch_from_cache(
        cache_sample,
        efm_snippet=None,
        drop_backbone_obbs=True,
    )
    assert batch.backbone_out is not None
    assert batch.backbone_out.obbs_pr_nms is None


def test_build_experiment_config_sets_backbone_keep_fields() -> None:
    cfg = _build_experiment_config(
        toml_path=None,
        stage=Stage.TRAIN,
        use_offline_cache=True,
        cache_dir=None,
        include_efm_snippet=True,
        include_gt_mesh=False,
    )
    source_cfg = cfg.datamodule_config.source
    assert source_cfg is not None
    assert source_cfg.cache.backbone_keep_fields == DEFAULT_BACKBONE_KEEP_FIELDS


def test_build_experiment_config_preserves_toml_keep_fields() -> None:
    toml_path = Path(__file__).resolve().parents[3] / ".configs" / "offline_only.toml"
    cfg = _build_experiment_config(
        toml_path=str(toml_path),
        stage=Stage.TRAIN,
        use_offline_cache=True,
        cache_dir=None,
        include_efm_snippet=True,
        include_gt_mesh=False,
    )
    source_cfg = cfg.datamodule_config.source
    assert source_cfg is not None
    assert source_cfg.cache.backbone_keep_fields is not None
    assert "obbs_pr_nms" not in source_cfg.cache.backbone_keep_fields


def test_should_fetch_vin_snippet() -> None:
    assert _should_fetch_vin_snippet(
        use_vin_snippet_cache=True,
        attach_snippet=False,
        require_vin_snippet=False,
    )
    assert _should_fetch_vin_snippet(
        use_vin_snippet_cache=True,
        attach_snippet=True,
        require_vin_snippet=True,
    )
    assert not _should_fetch_vin_snippet(
        use_vin_snippet_cache=False,
        attach_snippet=True,
        require_vin_snippet=True,
    )
