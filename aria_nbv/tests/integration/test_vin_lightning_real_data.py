"""Integration test: VIN training loop via PyTorch Lightning on real ASE data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make vendored efm3d importable (mirrors other integration tests).
sys.path.append(str(Path(__file__).resolve().parents[2] / "external" / "efm3d"))

try:  # pragma: no cover - availability guard
    import pytorch_lightning as _pl  # noqa: F401
except Exception:  # pragma: no cover - availability guard
    PL_AVAILABLE = False
else:  # pragma: no cover - availability guard
    PL_AVAILABLE = True

try:  # pragma: no cover - availability guard
    import pytorch3d.renderer as _pytorch3d_renderer  # noqa: F401
except Exception:  # pragma: no cover - availability guard
    PYTORCH3D_AVAILABLE = False
else:  # pragma: no cover - availability guard
    PYTORCH3D_AVAILABLE = True

from aria_nbv.configs import PathConfig
from aria_nbv.data import AseEfmDatasetConfig
from aria_nbv.data.vin_oracle_datasets import VinOracleOnlineDatasetConfig
from aria_nbv.pipelines import OracleRriLabelerConfig
from aria_nbv.pose_generation import CandidateViewGeneratorConfig
from aria_nbv.pose_generation.types import ViewDirectionMode
from aria_nbv.rendering import (
    CandidateDepthRendererConfig,
    Pytorch3DDepthRendererConfig,
)
from aria_nbv.rri_metrics.oracle_rri import OracleRRIConfig
from aria_nbv.utils import Verbosity
from aria_nbv.vin import EvlBackboneConfig, VinModelV3Config


def _skip_if_missing_assets() -> None:
    if not PL_AVAILABLE:
        pytest.skip("pytorch-lightning not installed", allow_module_level=True)
    if not PYTORCH3D_AVAILABLE:
        pytest.skip("PyTorch3D required for oracle labeler", allow_module_level=True)

    paths = PathConfig()
    model_cfg = paths.root / ".configs" / "evl_inf_desktop.yaml"
    model_ckpt = paths.root / ".logs" / "ckpts" / "model_lite.pth"
    if not model_cfg.exists():
        pytest.skip(f"Missing EVL config: {model_cfg}", allow_module_level=True)
    if not model_ckpt.exists():
        pytest.skip(f"Missing EVL checkpoint: {model_ckpt}", allow_module_level=True)

    atek_dir = paths.resolve_atek_data_dir("efm")
    if not any(atek_dir.glob("**/*.tar")):
        pytest.skip(f"No ATEK shards found under {atek_dir}", allow_module_level=True)
    if not any(paths.ase_meshes.glob("scene_ply_*.ply")):
        pytest.skip(f"No ASE meshes found under {paths.ase_meshes}", allow_module_level=True)


_skip_if_missing_assets()


@pytest.mark.integration
def test_vin_lightning_fit_runs_real_data_smoke(tmp_path: Path):
    from aria_nbv.lightning import (
        TrainerCallbacksConfig,
        TrainerFactoryConfig,
        VinDataModuleConfig,
        VinLightningModuleConfig,
    )
    from aria_nbv.rri_metrics.rri_binning import RriOrdinalBinner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_cfg = AseEfmDatasetConfig(
        scene_ids=["81283"],
        batch_size=None,
        load_meshes=True,
        require_mesh=True,
        mesh_simplify_ratio=0.02,  # aggressive decimation for runtime
        device="cpu",
        verbosity=Verbosity.QUIET,
        is_debug=True,
    )

    generator_cfg = CandidateViewGeneratorConfig(
        num_samples=2,
        oversample_factor=1.0,
        max_resamples=0,
        min_radius=0.0,
        max_radius=0.0,
        view_direction_mode=ViewDirectionMode.FORWARD_RIG,
        min_distance_to_mesh=0.0,
        ensure_collision_free=False,
        ensure_free_space=False,
        device=str(device),
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    depth_cfg = CandidateDepthRendererConfig(
        renderer=Pytorch3DDepthRendererConfig(
            device=str(device),
            verbosity=Verbosity.QUIET,
            dtype="float32",
        ),
        max_candidates_final=2,
        oversample_factor=1.0,
        resolution_scale=0.1,
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )
    oracle_cfg = OracleRRIConfig()
    labeler_cfg = OracleRriLabelerConfig(
        generator=generator_cfg,
        depth=depth_cfg,
        oracle=oracle_cfg,
        backprojection_stride=32,
    )

    source_cfg = VinOracleOnlineDatasetConfig(
        dataset=ds_cfg,
        labeler=labeler_cfg,
        max_attempts_per_batch=10,
        verbosity=Verbosity.QUIET,
    )
    dm_cfg = VinDataModuleConfig(
        source=source_cfg,
        num_workers=0,
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )

    datamodule = dm_cfg.setup_target()
    first_batch = next(iter(datamodule.train_dataloader()))
    binner = RriOrdinalBinner.fit_from_iterable([first_batch.rri.detach().cpu()], num_classes=15, target_items=1)
    binner_path = tmp_path / "rri_binner.json"
    binner.save(binner_path, overwrite=True)

    module_cfg = VinLightningModuleConfig(
        vin=VinModelV3Config(backbone=EvlBackboneConfig(device=device), num_classes=15),
        num_classes=15,
        save_binner=False,
        binner_path=binner_path,
    )

    trainer_cfg = TrainerFactoryConfig(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        fast_dev_run=True,
        max_epochs=1,
        use_wandb=False,
        callbacks=TrainerCallbacksConfig(
            use_model_checkpoint=False,
            use_rich_model_summary=False,
            use_tqdm_progress_bar=False,
            use_rich_progress_bar=False,
        ),
    )

    trainer = trainer_cfg.setup_target()
    module = module_cfg.setup_target()

    summary = module.summarize_vin(first_batch, include_torchsummary=True, torchsummary_depth=2)
    assert "VIN v3 summary" in summary

    trainer.fit(module, datamodule=datamodule)
    assert getattr(module, "_binner", None) is not None
