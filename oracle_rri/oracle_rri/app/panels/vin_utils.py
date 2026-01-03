"""VIN diagnostics helpers for panels."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ...configs import PathConfig
from ...data import EfmSnippetView
from ...data.offline_cache import OracleRriCacheConfig, OracleRriCacheDatasetConfig
from ...data.offline_cache_types import OracleRriCacheSample
from ...lightning.aria_nbv_experiment import AriaNBVExperimentConfig
from ...lightning.lit_datamodule import VinOracleBatch
from ...lightning.lit_module import VinLightningModule, VinLightningModuleConfig
from ...rri_metrics.rri_binning import RriOrdinalBinner
from ...utils import Stage
from ...vin import VinForwardDiagnostics, VinPrediction

if TYPE_CHECKING:
    from oracle_rri.lightning.lit_module import VinLightningModule


def _build_experiment_config(
    *,
    toml_path: str | None,
    stage: Stage,
    use_offline_cache: bool,
    cache_dir: str | None,
    map_location: str,
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
        paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
        cache_root = cache_dir or str(
            paths.offline_cache_dir or (paths.data_root / "oracle_rri_cache"),
        )
        cache_cfg = OracleRriCacheDatasetConfig(
            cache=OracleRriCacheConfig(cache_dir=Path(cache_root), paths=paths),
            load_backbone=True,
            map_location=map_location,
            include_efm_snippet=include_efm_snippet,
            include_gt_mesh=include_gt_mesh,
        )
        cfg.datamodule_config.train_cache = cache_cfg
        cfg.datamodule_config.val_cache = cache_cfg
    else:
        cfg.datamodule_config.num_workers = 0
        cfg.datamodule_config.train_cache = None
        cfg.datamodule_config.val_cache = None

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
    if batch.efm_snippet_view is None:
        if batch.backbone_out is None:
            raise RuntimeError(
                "VIN debug requires efm inputs or cached backbone outputs.",
            )
        efm = {}
        backbone_out = batch.backbone_out
    else:
        efm = batch.efm_snippet_view.efm
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
    efm_snippet: EfmSnippetView | None,
) -> VinOracleBatch:
    rri = cache_sample.rri
    depths = cache_sample.depths
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
        backbone_out=cache_sample.backbone_out,
    )


__all__ = [
    "_build_experiment_config",
    "_load_vin_module_from_checkpoint",
    "_run_vin_debug",
    "_vin_oracle_batch_from_cache",
]
