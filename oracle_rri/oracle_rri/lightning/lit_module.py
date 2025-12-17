"""LightningModule for training VIN (View Introspection Network).

This module implements the same core logic as `oracle_rri/scripts/train_vin.py`,
but with PyTorch Lightning training loops and optional W&B logging via the
trainer factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import pytorch_lightning as pl
import torch
from pydantic import Field, model_validator
from torch import Tensor
from torch.optim import AdamW, Optimizer

from ..configs import PathConfig
from ..utils import BaseConfig, Console, Stage
from ..vin import RriOrdinalBinner, VinModelConfig, coral_loss
from .lit_datamodule import VinOracleBatch


def _to_jsonable(value: Any) -> Any:
    """Convert nested config dumps to logger-friendly primitives."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_jsonable(v) for v in value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.device):
        return str(value)
    return value


class AdamWConfig(BaseConfig[Optimizer]):
    """AdamW optimizer configuration for VIN."""

    target: type[Optimizer] = Field(default_factory=lambda: AdamW, exclude=True)
    """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""

    learning_rate: float = 1e-4
    """Learning rate for AdamW."""

    weight_decay: float = 1e-2
    """Weight decay for AdamW."""

    def setup_target(self, params: list[Tensor]) -> Optimizer:  # type: ignore[override]
        return AdamW(params=params, lr=float(self.learning_rate), weight_decay=float(self.weight_decay))


class VinLightningModuleConfig(BaseConfig["VinLightningModule"]):
    """Configuration for :class:`VinLightningModule`."""

    target: type["VinLightningModule"] = Field(default_factory=lambda: VinLightningModule, exclude=True)

    vin: VinModelConfig = Field(default_factory=VinModelConfig)
    """Underlying VIN model configuration (frozen EVL backbone + CORAL head)."""

    optimizer: AdamWConfig = Field(default_factory=AdamWConfig)
    """Optimizer configuration."""

    num_classes: int = 15
    """Number of ordinal classes (must match `vin.head.num_classes`)."""

    binner_fit_snippets: int = 512
    """Number of oracle-labelled snippets used to fit the ordinal binner."""

    binner_max_attempts: int = 64
    """Maximum number of skipped oracle batches while fitting the binner (guards against bad oracle settings)."""

    save_binner: bool = True
    """Persist `rri_binner.json` into the run directory on fit start."""

    binner_path: Path | None = None
    """Optional explicit path to save `rri_binner.json` (defaults to trainer root dir)."""

    @model_validator(mode="after")
    def _validate_num_classes(self) -> Self:
        model_classes = int(self.vin.head.num_classes)
        if int(self.num_classes) != model_classes:
            raise ValueError(f"num_classes={self.num_classes} must match vin.head.num_classes={model_classes}.")
        return self


class VinLightningModule(pl.LightningModule):
    """PyTorch Lightning module for VIN training with CORAL ordinal regression."""

    def __init__(self, config: VinLightningModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(_to_jsonable(config.model_dump()))

        self.console = Console.with_prefix(self.__class__.__name__)

        self.vin = self.config.vin.setup_target()
        self._binner: RriOrdinalBinner | None = None

    # --------------------------------------------------------------------- lifecycle
    def setup(self, stage: str) -> None:  # noqa: A003
        super().setup(stage)
        self._integrate_console()
        if self._binner is None:
            self._binner = self._load_binner_from_config()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self._binner is not None:
            checkpoint["rri_binner"] = self._binner.to_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        data = checkpoint.get("rri_binner")
        if data is not None:
            self._binner = RriOrdinalBinner.from_dict(data)

    # ------------------------------------------------------------------ training/val/test
    def training_step(self, batch: VinOracleBatch, batch_idx: int) -> Tensor | None:
        return self._step(batch, batch_idx, stage=Stage.TRAIN)

    def validation_step(self, batch: VinOracleBatch, batch_idx: int) -> Tensor | None:
        return self._step(batch, batch_idx, stage=Stage.VAL)

    def test_step(self, batch: VinOracleBatch, batch_idx: int) -> Tensor | None:
        return self._step(batch, batch_idx, stage=Stage.TEST)

    # ------------------------------------------------------------------ optim
    def configure_optimizers(self) -> Optimizer:
        params = [p for p in self.vin.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found (did you freeze everything?).")
        return self.config.optimizer.setup_target(params=params)

    # ------------------------------------------------------------------ internals
    def _step(self, batch: VinOracleBatch, batch_idx: int, *, stage: Stage) -> Tensor | None:
        self._integrate_console()
        Console.update_global_step(int(self.global_step))

        if self._binner is None:
            raise RuntimeError(
                "RRI binner not initialized. Provide `VinLightningModuleConfig.binner_path` (a fitted .json), "
                "or resume from a checkpoint that contains `rri_binner`."
            )

        pred = self.vin.forward(
            batch.efm,
            batch.candidate_poses_world_cam,
            reference_pose_world_rig=batch.reference_pose_world_rig,
            candidate_poses_camera_rig=batch.candidate_poses_camera_rig,
        )
        logits = pred.logits.squeeze(0)  # N x (K-1)
        valid = pred.candidate_valid.squeeze(0)  # N

        rri = batch.rri.to(device=logits.device)
        labels = self._binner.transform(rri.reshape(-1))

        mask = valid & torch.isfinite(rri)
        if not mask.any():
            self.log(f"{stage}/skip_no_valid", 1.0, on_step=True, prog_bar=False, batch_size=1)
            return None

        loss = coral_loss(
            logits[mask],
            labels[mask],
            num_classes=int(self._binner.num_classes),
            reduction="mean",
        )

        prefix = f"{stage}"
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=(stage is Stage.TRAIN),
            batch_size=1,
        )
        # Filename-friendly alias (avoid '/' in checkpoint monitor keys).
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log(
            f"{prefix}/voxel_valid_fraction",
            float(mask.sum().item()) / float(mask.numel()),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/rri_mean",
            rri[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pred_mean",
            pred.expected_normalized.squeeze(0)[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        pm_dist_before = batch.pm_dist_before.to(device=logits.device)
        pm_dist_after = batch.pm_dist_after.to(device=logits.device)
        pm_acc_before = batch.pm_acc_before.to(device=logits.device)
        pm_comp_before = batch.pm_comp_before.to(device=logits.device)
        pm_acc_after = batch.pm_acc_after.to(device=logits.device)
        pm_comp_after = batch.pm_comp_after.to(device=logits.device)

        self.log(
            f"{prefix}/pm_dist_before_mean",
            pm_dist_before[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pm_dist_after_mean",
            pm_dist_after[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pm_acc_before_mean",
            pm_acc_before[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pm_comp_before_mean",
            pm_comp_before[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pm_acc_after_mean",
            pm_acc_after[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            f"{prefix}/pm_comp_after_mean",
            pm_comp_after[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        return loss

    def _load_binner_from_config(self) -> RriOrdinalBinner:
        if self.config.binner_path is None:
            raise RuntimeError(
                "Missing `VinLightningModuleConfig.binner_path`. Fit a binner first (e.g. via `nbv-fit-binner`) "
                "and point this config field to the resulting `rri_binner.json`, or resume from a checkpoint."
            )

        resolved = PathConfig().resolve_artifact_path(
            self.config.binner_path, expected_suffix=".json", create_parent=False
        )
        if not resolved.exists():
            raise FileNotFoundError(
                f"RRI binner not found at {resolved}. Run `nbv-fit-binner --out-dir <run_dir>` to create it "
                "or set `VinLightningModuleConfig.binner_path` to an existing fitted binner JSON."
            )
        return RriOrdinalBinner.load(resolved)

    def _integrate_console(self) -> None:
        logger = getattr(self, "logger", None)
        if logger is not None:
            Console.integrate_with_logger(logger, global_step=int(self.global_step))


__all__ = ["AdamWConfig", "VinLightningModule", "VinLightningModuleConfig"]
