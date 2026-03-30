"""VIN diagnostics helpers for panels."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ...configs import PathConfig
from ...data_handling import (
    EfmSnippetView,
    VinOracleBatch,
    VinOracleOnlineDatasetConfig,
    VinSnippetView,
)
from ...data_handling._legacy_cache_api import (
    OracleRriCacheConfig,
    OracleRriCacheDatasetConfig,
    OracleRriCacheSample,
)
from ...data_handling._legacy_vin_source import VinOracleCacheDatasetConfig
from ...lightning.aria_nbv_experiment import AriaNBVExperimentConfig
from ...lightning.lit_module import VinLightningModule, VinLightningModuleConfig
from ...rri_metrics.rri_binning import RriOrdinalBinner
from ...utils import Stage
from ...vin.experimental.types import VinForwardDiagnostics
from ...vin.types import EvlBackboneOutput, VinPrediction

if TYPE_CHECKING:
    from aria_nbv.lightning.lit_module import VinLightningModule


DEFAULT_BACKBONE_KEEP_FIELDS: list[str] = [
    "t_world_voxel",
    "voxel_extent",
    "occ_pr",
    "occ_input",
    "counts",
    "cent_pr",
    "free_input",
    "pts_world",
]
"""Default backbone fields for VIN diagnostics (OBB outputs excluded)."""


def _build_experiment_config(
    *,
    toml_path: str | None,
    stage: Stage,
    use_offline_cache: bool,
    cache_dir: str | None,
    include_efm_snippet: bool,
    include_gt_mesh: bool,
) -> AriaNBVExperimentConfig:
    if toml_path:
        cfg = AriaNBVExperimentConfig.from_toml(Path(toml_path))
    else:
        cfg = AriaNBVExperimentConfig()

    cfg.run_mode = "summarize_vin"
    cfg.stage = stage
    cfg.trainer_config.use_wandb = False

    if use_offline_cache:
        # NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION: diagnostics
        # branch that swaps the datamodule to the legacy oracle cache.
        keep_fields: list[str] | None = None
        source_cfg = getattr(cfg.datamodule_config, "source", None)
        if isinstance(source_cfg, VinOracleCacheDatasetConfig):
            keep_fields = source_cfg.cache.backbone_keep_fields
        if keep_fields is None:
            keep_fields = DEFAULT_BACKBONE_KEEP_FIELDS
        paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
        cache_root = cache_dir or str(
            paths.offline_cache_dir or (paths.data_root / "oracle_rri_cache"),
        )
        cache_cfg = OracleRriCacheDatasetConfig(
            cache=OracleRriCacheConfig(cache_dir=Path(cache_root), paths=paths),
            load_backbone=True,
            include_efm_snippet=include_efm_snippet,
            include_gt_mesh=include_gt_mesh,
            backbone_keep_fields=keep_fields,
        )
        cache_source = VinOracleCacheDatasetConfig(cache=cache_cfg)
        cfg.datamodule_config.source = cache_source
    else:
        cfg.datamodule_config.num_workers = 0
        cfg.datamodule_config.source = VinOracleOnlineDatasetConfig()

    return cfg


def _load_vin_module_from_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device | str,
) -> VinLightningModule:
    payload = torch.load(checkpoint_path, map_location="cpu")
    hparams = payload.get("hyper_parameters", {})
    if isinstance(hparams, dict) and "config" in hparams and isinstance(hparams["config"], dict):
        config_payload = hparams["config"]
    elif isinstance(hparams, dict):
        config_payload = hparams
    else:
        config_payload = {}
    config = VinLightningModuleConfig(**config_payload)
    module = VinLightningModule(config=config)
    module.on_load_checkpoint(payload)
    state_dict = payload.get("state_dict")
    if state_dict is None:
        raise RuntimeError("Checkpoint missing state_dict.")
    module.load_state_dict(state_dict, strict=False)
    module.to(torch.device(device))
    module.eval()
    try:
        if getattr(module, "_binner", None) is None:
            binner = None
            if module.config.binner_path is not None:
                try:
                    binner = module._load_binner_from_config()
                except Exception:
                    binner = None
            if binner is None:
                default_binner_path = Path(".logs") / "vin" / "rri_binner.json"
                try:
                    binner = RriOrdinalBinner.load(default_binner_path)
                except Exception:
                    binner = None
            if binner is not None:
                module._binner = binner
        module._maybe_init_bin_values()
    except Exception:
        pass
    return module


def _run_vin_debug(
    module: "VinLightningModule",
    batch: VinOracleBatch,
) -> tuple[VinPrediction, VinForwardDiagnostics]:
    was_training = module.vin.training
    module.vin.eval()
    snippet_view = batch.efm_snippet_view
    if snippet_view is None:
        if batch.backbone_out is None:
            raise RuntimeError(
                "VIN debug requires efm inputs or cached backbone outputs.",
            )
        raise RuntimeError(
            "VIN debug requires a VinSnippetView or EfmSnippetView. Enable VIN snippet cache or attach EFM snippets.",
        )
    if isinstance(snippet_view, (EfmSnippetView, VinSnippetView)):
        efm = snippet_view
    elif hasattr(snippet_view, "points_world"):
        efm = snippet_view
    else:
        raise TypeError(
            f"VIN debug expects a VinSnippetView or EfmSnippetView, got {type(snippet_view)}.",
        )
    backbone_out = batch.backbone_out
    if backbone_out is not None:
        device = backbone_out.voxel_extent.device
        if next(module.vin.parameters()).device != device:
            module.vin.to(device)
        batch = batch.to(device) if hasattr(batch, "to") else batch
        backbone_out = backbone_out.to(device)
        if hasattr(batch, "p3d_cameras"):
            batch.p3d_cameras = batch.p3d_cameras.to(device)
    with torch.no_grad():
        pred, debug = module.vin.forward_with_debug(
            efm,
            candidate_poses_world_cam=batch.candidate_poses_world_cam,
            reference_pose_world_rig=batch.reference_pose_world_rig,
            p3d_cameras=batch.p3d_cameras,
            backbone_out=backbone_out,
        )
    if was_training:
        module.vin.train()
    return pred, debug


def _vin_oracle_batch_from_cache(
    cache_sample: OracleRriCacheSample,
    *,
    efm_snippet: EfmSnippetView | VinSnippetView | None,
    drop_backbone_obbs: bool = False,
) -> VinOracleBatch:
    rri = cache_sample.rri
    depths = cache_sample.depths
    backbone_out = cache_sample.backbone_out
    if drop_backbone_obbs and backbone_out is not None:
        backbone_out = _strip_backbone_obbs(backbone_out)
    return VinOracleBatch(
        efm_snippet_view=efm_snippet,
        candidate_poses_world_cam=depths.poses,
        reference_pose_world_rig=depths.reference_pose,
        rri=rri.rri,
        pm_dist_before=rri.pm_dist_before,
        pm_dist_after=rri.pm_dist_after,
        pm_acc_before=rri.pm_acc_before,
        pm_comp_before=rri.pm_comp_before,
        pm_acc_after=rri.pm_acc_after,
        pm_comp_after=rri.pm_comp_after,
        p3d_cameras=depths.p3d_cameras,
        scene_id=cache_sample.scene_id,
        snippet_id=cache_sample.snippet_id,
        backbone_out=backbone_out,
    )


def _has_backbone_obbs(backbone_out: EvlBackboneOutput | None) -> bool:
    if backbone_out is None:
        return False
    return any(
        getattr(backbone_out, name) is not None
        for name in (
            "obbs_pr_nms",
            "obb_pred",
            "obb_pred_viz",
            "obb_pred_probs_full",
            "obb_pred_probs_full_viz",
            "obb_pred_sem_id_to_name",
        )
    )


def _strip_backbone_obbs(backbone_out: EvlBackboneOutput) -> EvlBackboneOutput:
    return replace(
        backbone_out,
        obbs_pr_nms=None,
        obb_pred=None,
        obb_pred_viz=None,
        obb_pred_probs_full=None,
        obb_pred_probs_full_viz=None,
        obb_pred_sem_id_to_name=None,
    )


def _should_fetch_vin_snippet(
    *,
    use_vin_snippet_cache: bool,
    attach_snippet: bool,
    require_vin_snippet: bool,
) -> bool:
    """Return True when VIN snippet cache entries should be fetched."""
    _ = attach_snippet
    _ = require_vin_snippet
    return use_vin_snippet_cache


__all__ = [
    "_build_experiment_config",
    "_load_vin_module_from_checkpoint",
    "_run_vin_debug",
    "_should_fetch_vin_snippet",
    "_vin_oracle_batch_from_cache",
]
