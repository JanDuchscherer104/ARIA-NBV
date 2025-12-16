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
from .lit_datamodule import VinDataModule, VinOracleBatch


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

    learning_rate: float = 1e-3
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

    binner_fit_snippets: int = 2
    """Number of oracle-labelled snippets used to fit the ordinal binner."""

    binner_tanh_scale: float = 1.0
    """Scale applied before tanh clipping (VIN-NBV style)."""

    binner_max_attempts: int = 25
    """Maximum attempts while fitting the binner (guards against bad oracle settings)."""

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
    def on_fit_start(self) -> None:
        self._integrate_console()
        if self._binner is None:
            self._binner = self._fit_binner_from_datamodule()
            self._maybe_save_binner()

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
            raise RuntimeError("RRI binner not initialized. Run trainer.fit() first or load from checkpoint.")

        pred = self.vin(
            batch.efm,
            batch.candidate_poses_world_cam,
            reference_pose_world_rig=batch.reference_pose_world_rig,
            candidate_poses_camera_rig=batch.candidate_poses_camera_rig,
        )
        logits = pred.logits.squeeze(0)  # N x (K-1)
        valid = pred.candidate_valid.squeeze(0)  # N

        rri = batch.rri.to(device=logits.device)
        stage_ids = batch.stage.to(device=logits.device)
        labels = self._binner.transform(rri.reshape(-1), stage_ids.reshape(-1))

        mask = valid & torch.isfinite(rri)
        if not mask.any():
            self.log(f"{stage}/skip_no_valid", 1.0, on_step=True, prog_bar=False)
            return None

        loss = coral_loss(
            logits[mask],
            labels[mask],
            num_classes=int(self._binner.num_classes),
            reduction="mean",
        )

        prefix = f"{stage}"
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=(stage is Stage.TRAIN))
        # Filename-friendly alias (avoid '/' in checkpoint monitor keys).
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            f"{prefix}/num_candidates",
            float(mask.numel()),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{prefix}/num_valid",
            float(mask.sum().item()),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{prefix}/rri_mean",
            rri[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{prefix}/pred_mean",
            pred.expected_normalized.squeeze(0)[mask].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def _fit_binner_from_datamodule(self) -> RriOrdinalBinner:
        dm = getattr(self.trainer, "datamodule", None)
        if not isinstance(dm, VinDataModule):
            raise TypeError("VinLightningModule expects VinDataModule for binner fitting.")

        fit_snippets = int(self.config.binner_fit_snippets)
        if fit_snippets <= 0:
            raise ValueError("binner_fit_snippets must be > 0 when no binner is loaded from checkpoint.")

        rri_all: list[Tensor] = []
        stage_all: list[Tensor] = []

        console = Console.with_prefix(self.__class__.__name__, "binner_fit")
        console.log(f"Fitting RRI ordinal binner on {fit_snippets} snippets.")

        it = dm.iter_oracle_batches(stage=Stage.TRAIN)
        successes = 0
        attempts = 0
        while successes < fit_snippets:
            if attempts >= int(self.config.binner_max_attempts):
                raise RuntimeError(
                    f"Unable to fit binner: only {successes}/{fit_snippets} snippets after {attempts} attempts."
                )
            attempts += 1

            batch = next(it)
            rri = batch.rri.detach().reshape(-1).to(dtype=torch.float32)
            stage = batch.stage.detach().reshape(-1).to(dtype=torch.int64)
            if rri.numel() == 0 or not torch.isfinite(rri).any():
                continue

            rri_all.append(rri)
            stage_all.append(stage)
            successes += 1
            console.log(
                f"  fit[{successes:02d}/{fit_snippets:02d}] scene={batch.scene_id} snip={batch.snippet_id} "
                f"C={int(rri.numel())} rri_mean={float(rri.mean().item()):.4f} rri_std={float(rri.std().item()):.4f}"
            )

        binner = RriOrdinalBinner.fit(
            torch.cat(rri_all, dim=0),
            torch.cat(stage_all, dim=0),
            num_classes=int(self.config.num_classes),
            tanh_scale=float(self.config.binner_tanh_scale),
        )
        binner.edges = binner.edges.detach().cpu()
        console.log("Binner fitted.")
        return binner

    def _maybe_save_binner(self) -> None:
        if not self.config.save_binner or self._binner is None:
            return
        path = self._resolve_binner_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._binner.save(path)
        self.console.log(f"Saved binner to {path}")

    def _resolve_binner_path(self) -> Path:
        if self.config.binner_path is not None:
            path = Path(self.config.binner_path)
            if not path.is_absolute():
                path = (PathConfig().root / path).resolve()
            return path

        root_dir = getattr(self.trainer, "default_root_dir", None)
        if root_dir:
            return (Path(root_dir) / "rri_binner.json").resolve()
        return (PathConfig().root / ".logs" / "rri_binner.json").resolve()

    def _integrate_console(self) -> None:
        logger = getattr(self, "logger", None)
        if logger is not None:
            Console.integrate_with_logger(logger, global_step=int(self.global_step))


__all__ = ["AdamWConfig", "VinLightningModule", "VinLightningModuleConfig"]
