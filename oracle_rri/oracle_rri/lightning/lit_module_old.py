"""LightningModule for training VIN (View Introspection Network).

This module implements the same core logic as `oracle_rri/scripts/train_vin.py`,
but with PyTorch Lightning training loops and optional W&B logging via the
trainer factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import matplotlib
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pydantic import Field, field_validator, model_validator
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from ..configs import PathConfig
from ..data.efm_views import EfmSnippetView, VinSnippetView
from ..rri_metrics import (
    Metric,
    RriOrdinalBinner,
    VinMetricsConfig,
    coral_loss,
    coral_random_loss,
    topk_accuracy_from_probs,
)
from ..rri_metrics.coral import coral_logits_to_label, coral_monotonicity_violation_rate
from ..utils import BaseConfig, Console, Optimizable, Stage, optimizable_field
from ..vin.experimental.model import VinModelConfig
from ..vin.experimental.model_v2 import VinModelV2Config
from ..vin.experimental.plotting import plot_vin_encodings_from_debug
from .lit_datamodule import VinOracleBatch


class AdamWConfig(BaseConfig[Optimizer]):
    """AdamW optimizer configuration for VIN."""

    @property
    def target(self) -> type[Optimizer]:
        """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""
        return AdamW

    learning_rate: float = optimizable_field(
        default=1e-3,
        optimizable=Optimizable.continuous(
            low=1e-5,
            high=3e-4,
            log=True,
            description="AdamW learning rate.",
        ),
    )
    """Learning rate for AdamW."""

    weight_decay: float = optimizable_field(
        default=1e-3,
        optimizable=Optimizable.continuous(
            low=1e-4,
            high=1e-1,
            log=True,
            description="AdamW weight decay.",
        ),
    )
    """Weight decay for AdamW."""

    def setup_target(self, params: list[Tensor]) -> Optimizer:  # type: ignore[override]
        return AdamW(
            params=params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class ReduceLrOnPlateauConfig(BaseConfig[ReduceLROnPlateau]):
    """ReduceLROnPlateau scheduler configuration."""

    @property
    def target(self) -> type[ReduceLROnPlateau]:
        """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""
        return ReduceLROnPlateau

    patience: int = 2
    """Number of steps with no improvement before reducing the LR."""

    factor: float = 0.2
    """Multiplicative factor of LR reduction."""

    monitor: str = "train/loss"
    """Metric name to monitor for plateau reduction."""

    interval: Literal["step", "epoch"] = "epoch"
    """Scheduler interval (step or epoch)."""

    frequency: int = 1
    """Scheduler frequency."""

    def setup_target(  # type: ignore[override]
        self,
        optimizer: Optimizer,
        *,
        trainer: pl.Trainer | None = None,
    ) -> ReduceLROnPlateau:
        del trainer
        return ReduceLROnPlateau(optimizer, patience=self.patience, factor=self.factor)

    def setup_lightning(
        self,
        optimizer: Optimizer,
        *,
        trainer: pl.Trainer | None = None,
    ) -> dict[str, Any]:
        """Build the Lightning lr_scheduler config for ReduceLROnPlateau.

        Args:
            optimizer: Optimizer instance to schedule.
            trainer: Optional Lightning trainer (unused for plateau).

        Returns:
            Lightning lr_scheduler configuration dictionary.
        """
        scheduler = self.setup_target(optimizer, trainer=trainer)
        return {
            "scheduler": scheduler,
            "monitor": self.monitor,
            "interval": self.interval,
            "frequency": self.frequency,
        }


class OneCycleSchedulerConfig(BaseConfig[OneCycleLR]):
    """OneCycle learning-rate scheduler configuration."""

    @property
    def target(self) -> type[OneCycleLR]:
        """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""
        return OneCycleLR

    max_lr: float | None = None
    """Maximum learning rate in the cycle (defaults to optimizer LR)."""

    base_momentum: float = 0.85
    """Lower momentum boundary in the cycle."""

    max_momentum: float = 0.95
    """Upper momentum boundary in the cycle."""

    div_factor: float = 25.0
    """Initial learning rate = max_lr / div_factor."""

    final_div_factor: float = 1e4
    """Final learning rate = max_lr / (div_factor * final_div_factor)."""

    pct_start: float = 0.3
    """Percentage of cycle spent increasing learning rate."""

    anneal_strategy: Literal["cos", "linear"] = "cos"
    """Annealing strategy: 'cos' or 'linear'."""

    def setup_target(  # type: ignore[override]
        self,
        optimizer: Optimizer,
        *,
        total_steps: int | None = None,
        trainer: pl.Trainer | None = None,
    ) -> OneCycleLR:
        if total_steps is None:
            total_steps = self._resolve_total_steps(trainer)
        if total_steps <= 0:
            raise ValueError("OneCycleLR requires total_steps > 0.")

        max_lr = self.max_lr
        if max_lr is None:
            max_lr = optimizer.param_groups[0]["lr"]

        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            cycle_momentum=True,
            base_momentum=self.base_momentum,
            max_momentum=self.max_momentum,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
        )

    def setup_lightning(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int | None = None,
        trainer: pl.Trainer | None = None,
    ) -> dict[str, Any]:
        """Build the Lightning lr_scheduler config for OneCycleLR.

        Args:
            optimizer: Optimizer instance to schedule.
            total_steps: Optional total step count for the cycle.
            trainer: Optional Lightning trainer used to infer total_steps.

        Returns:
            Lightning lr_scheduler configuration dictionary.
        """
        scheduler = self.setup_target(
            optimizer,
            total_steps=total_steps,
            trainer=trainer,
        )
        return {"scheduler": scheduler, "interval": "step"}

    @staticmethod
    def _resolve_total_steps(trainer: pl.Trainer | None) -> int:
        if trainer is None:
            raise ValueError(
                "OneCycleLR requires either total_steps or a configured trainer.",
            )

        total_steps = int(getattr(trainer, "estimated_stepping_batches", 0) or 0)
        if total_steps > 0:
            return total_steps

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            raise ValueError(
                "Trainer is missing a datamodule; cannot infer total_steps for OneCycleLR.",
            )

        steps_per_epoch = len(datamodule.train_dataloader())
        max_epochs = int(getattr(trainer, "max_epochs", 1) or 1)
        return steps_per_epoch * max_epochs


class VinLightningModuleConfig(BaseConfig["VinLightningModule"]):
    """Configuration for :class:`VinLightningModule`."""

    @property
    def target(self) -> type["VinLightningModule"]:
        return VinLightningModule

    vin: VinModelConfig | VinModelV2Config = Field(default_factory=VinModelV2Config)

    optimizer: AdamWConfig = Field(default_factory=AdamWConfig)
    """Optimizer configuration."""

    lr_scheduler: OneCycleSchedulerConfig | ReduceLrOnPlateauConfig | None = Field(
        default_factory=ReduceLrOnPlateauConfig,
    )
    """Learning-rate scheduler configuration (set to ``None`` to disable)."""

    num_classes: int = 15
    """Number of ordinal classes (must match `vin.head.num_classes`)."""

    binner_fit_snippets: int | None = None
    """Number of oracle-labelled snippets used to fit the ordinal binner. If `` None`` uses all available (offline) or fit until interrupted (online)."""

    binner_max_attempts: int = 64
    """Maximum number of skipped oracle batches while fitting the binner (guards against bad oracle settings)."""

    save_binner: bool = True
    """Persist `rri_binner.json` into the run directory on fit start."""

    binner_path: Path | None = None
    """Optional explicit path to save `rri_binner.json` (defaults to trainer root dir)."""

    use_valid_frac_weight: bool = True
    """Whether to weight loss by per-candidate frustum coverage."""

    valid_frac_weight_floor: float = Field(default=0.2, ge=0.0, le=1.0)
    """Minimum loss weight for candidates with low voxel coverage."""

    aux_regression_loss: Literal["mse", "huber"] | None = "huber"
    """Auxiliary regression loss on expected RRI (set to ``None`` to disable)."""

    log_interval_steps: int | None = Field(default=None)
    """Step interval for logging rank/confusion/histogram metrics (train stage only). If ``None`` only log per-epoch metrics."""

    log_grad_norms: bool = True
    """Whether to log gradient norms for key VIN submodules."""

    @field_validator("log_interval_steps")
    @classmethod
    def _validate_log_interval_steps(cls, value: int | None) -> int | None:
        if value is None or not value:
            return None
        value = int(value)
        if value < 1:
            raise ValueError("log_interval_steps must be >= 1 or None.")
        return value

    @model_validator(mode="after")
    def _validate_num_classes(self) -> Self:
        if self.num_classes != (vin_num_cls := getattr(self.vin, "num_classes", self.num_classes)):
            raise ValueError(
                f"num_classes={self.num_classes} must match vin.num_classes={vin_num_cls}.",
            )

        return self


class VinLightningModule(pl.LightningModule):
    """PyTorch Lightning module for VIN training with CORAL ordinal regression."""

    def __init__(self, config: VinLightningModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump_jsonable())

        self.console = Console.with_prefix(self.__class__.__name__)

        self.vin = config.vin.setup_target()
        self._binner: RriOrdinalBinner | None = None
        metrics_cfg = VinMetricsConfig(num_classes=self.config.num_classes)

        self._metrics = nn.ModuleDict(
            {
                f"{Stage.TRAIN.value}_stage": metrics_cfg.setup_target(),
                f"{Stage.VAL.value}_stage": metrics_cfg.setup_target(),
                f"{Stage.TEST.value}_stage": metrics_cfg.setup_target(),
            },
        )
        self._interval_metrics = metrics_cfg.setup_target()

    # --------------------------------------------------------------------- lifecycle
    def setup(self, stage: str) -> None:
        super().setup(stage)
        self._integrate_console()
        if self._binner is None:
            self._binner = self._load_binner_from_config()
        self._maybe_init_bin_values()

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

    # ------------------------------------------------------------------ epoch-end metrics
    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.TRAIN)
        self._interval_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.VAL)

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.TEST)

    def on_after_backward(self) -> None:
        if not self.config.log_grad_norms:
            return
        if getattr(self.trainer, "sanity_checking", False):
            return
        if not self.training:
            return

        def _log_grad(name: str, module: nn.Module | None) -> None:
            if module is None:
                return
            total = 0.0
            for param in module.parameters():
                if param.grad is None:
                    continue
                param_norm = float(param.grad.detach().data.norm(2).item())
                total += param_norm * param_norm
            self.log(
                f"train-gradnorms/grad_norm_{name}",
                float(total**0.5),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        _log_grad("pose_encoder_lff", getattr(self.vin, "pose_encoder_lff", None))
        _log_grad("head_mlp", getattr(self.vin, "head_mlp", None))
        _log_grad("head_coral", getattr(self.vin, "head_coral", None))

    # ------------------------------------------------------------------ optim
    def configure_optimizers(self) -> dict[str, Any]:
        params = [p for p in self.vin.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                "No trainable parameters found (did you freeze everything?).",
            )
        optimizer = self.config.optimizer.setup_target(params=params)
        scheduler_cfg = self.config.lr_scheduler
        if scheduler_cfg is None:
            return {"optimizer": optimizer}

        lr_scheduler = scheduler_cfg.setup_lightning(
            optimizer,
            trainer=getattr(self, "trainer", None),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    # ------------------------------------------------------------------ internals
    def _step(
        self,
        batch: VinOracleBatch,
        batch_idx: int,
        *,
        stage: Stage,
    ) -> Tensor | None:
        if self._binner is None:
            raise RuntimeError(
                "RRI binner not initialized. Provide `VinLightningModuleConfig.binner_path` (a fitted .json), "
                "or resume from a checkpoint that contains `rri_binner`.",
            )

        efm_snippet_view = batch.efm_snippet_view
        if isinstance(efm_snippet_view, (EfmSnippetView, VinSnippetView)):
            efm = efm_snippet_view
        else:
            raise RuntimeError(
                "VIN batch missing semidense snippet view; VinModelV3 requires VinSnippetView or EfmSnippetView.",
            )
        backbone_out = batch.backbone_out
        if backbone_out is None and not isinstance(efm_snippet_view, EfmSnippetView):
            raise RuntimeError(
                "VIN batch missing both efm snippet view and cached backbone outputs.",
            )

        if backbone_out is not None:
            backbone_out = backbone_out.to(self.device)

        p3d_cameras = batch.p3d_cameras.to(self.device)
        if p3d_cameras.device != self.device:
            p3d_cameras = p3d_cameras.to(self.device)

        candidate_poses_world_cam = batch.candidate_poses_world_cam.to(device=self.device)
        reference_pose_world_rig = batch.reference_pose_world_rig.to(device=self.device)

        pred = self.vin.forward(
            efm,
            candidate_poses_world_cam=candidate_poses_world_cam,
            reference_pose_world_rig=reference_pose_world_rig,
            p3d_cameras=p3d_cameras,
            backbone_out=backbone_out,
        )
        num_candidates = int(batch.candidate_poses_world_cam.shape[-2])
        log_batch_size = max(num_candidates, 1)
        log_enabled = not getattr(self.trainer, "sanity_checking", False)
        logits = pred.logits.squeeze(0)  # N x (K-1)
        valid_frac = pred.voxel_valid_frac.squeeze(0) if pred.voxel_valid_frac is not None else None  # N

        rri = batch.rri.to(device=logits.device)
        labels = self._binner.transform(rri.reshape(-1))

        mask = torch.isfinite(rri)
        if not mask.any():
            if log_enabled:
                self.log(
                    f"{stage}/skip_no_valid",
                    1.0,
                    on_step=True,
                    prog_bar=False,
                    batch_size=log_batch_size,
                )
            return None

        loss_per = coral_loss(
            logits[mask],
            labels[mask],
            num_classes=int(self._binner.num_classes),
            reduction="none",
        )
        if self.config.use_valid_frac_weight and valid_frac is not None:
            weights = (
                self.config.valid_frac_weight_floor + (1.0 - self.config.valid_frac_weight_floor) * valid_frac[mask]
            )
            coral_loss_value = (loss_per * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            coral_loss_value = loss_per.mean()

        probs = pred.prob.squeeze(0)
        pred_rri_proxy = None
        aux_loss = None
        if self.config.aux_regression_loss is not None or log_enabled:
            head_coral = getattr(self.vin, "head_coral", None)
            if head_coral is not None and getattr(head_coral, "has_bin_values", False):
                pred_rri_proxy = head_coral.expected_from_probs(probs)
            else:
                pred_rri_proxy = self._binner.expected_from_probs(probs)

        combined_loss = coral_loss_value
        if self.config.aux_regression_loss is not None:
            if pred_rri_proxy is None:
                raise RuntimeError("Expected pred_rri_proxy to be computed.")
            if self.config.aux_regression_loss == "mse":
                diff = pred_rri_proxy[mask] - rri[mask].to(
                    dtype=pred_rri_proxy.dtype,
                )
                aux_loss = (diff * diff).mean()
            elif self.config.aux_regression_loss == "huber":
                aux_loss = functional.smooth_l1_loss(
                    pred_rri_proxy[mask],
                    rri[mask].to(dtype=pred_rri_proxy.dtype),
                )
            else:
                raise ValueError(
                    f"Unknown aux_regression_loss='{self.config.aux_regression_loss}'.",
                )
            combined_loss = coral_loss_value + aux_loss

        if not log_enabled:
            return combined_loss

        on_step = stage is Stage.TRAIN
        random_coral_loss = coral_random_loss(int(self._binner.num_classes))
        loss_metrics: dict[str, Tensor | float] = {
            "loss": combined_loss,
            "coral_loss": coral_loss_value,
            "coral_loss_rel_random": coral_loss_value / random_coral_loss,
        }
        if aux_loss is not None:
            loss_metrics["aux_regression_loss"] = aux_loss
        self._log_loss_scalars(
            loss_metrics,
            stage=stage,
            on_step=on_step,
            on_epoch=True,
            prog_bar=(stage is Stage.TRAIN),
            batch_size=log_batch_size,
        )

        self._log_aux_scalars(
            {
                Metric.RRI_MEAN: rri[mask].mean(),
                Metric.PRED_RRI_MEAN: pred_rri_proxy[mask].mean()
                if pred_rri_proxy is not None
                else torch.tensor(float("nan"), device=combined_loss.device),
                Metric.VOXEL_VALID_FRAC_MEAN: float(valid_frac.mean().item())
                if valid_frac is not None
                else float("nan"),
                Metric.CANDIDATE_VALID_FRAC: float(
                    pred.candidate_valid.float().mean().item(),
                ),
                Metric.TOP3_ACCURACY: topk_accuracy_from_probs(
                    probs[mask],
                    labels[mask],
                    top_k=3,
                ),
            },
            stage=stage,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            batch_size=log_batch_size,
        )

        pred_class = coral_logits_to_label(logits)
        monotonicity_rate = coral_monotonicity_violation_rate(logits[mask]).mean()
        self.log(
            f"{stage.value}-aux/coral_monotonicity_violation_rate",
            monotonicity_rate,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            batch_size=log_batch_size,
        )
        stage_key = f"{stage.value}_stage"
        self._metrics[stage_key].update(
            pred_scores=pred.expected_normalized.squeeze(0)[mask].to(
                dtype=torch.float32,
            ),
            rri=rri[mask].to(dtype=torch.float32),
            pred_class=pred_class[mask],
            labels=labels[mask],
        )
        if stage is Stage.TRAIN:
            self._interval_metrics.update(
                pred_scores=pred.expected_normalized.squeeze(0)[mask].to(
                    dtype=torch.float32,
                ),
                rri=rri[mask].to(dtype=torch.float32),
                pred_class=pred_class[mask],
                labels=labels[mask],
            )
        self._log_interval_metrics(
            stage,
            batch_idx=batch_idx,
            batch_size=log_batch_size,
        )

        return combined_loss

    def _log_epoch_metrics(self, stage: Stage) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            stage_key = f"{stage.value}_stage"
            self._metrics[stage_key].reset()
            return

        stage_key = f"{stage.value}_stage"
        metrics = self._metrics[stage_key].compute()
        if not metrics:
            self._metrics[stage_key].reset()
            return

        spearman = metrics["spearman"]
        if torch.isfinite(spearman):
            self._log_aux_scalars(
                {Metric.SPEARMAN: spearman},
                stage=stage,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=1,
            )

        self._log_confusion_matrix(
            metrics["confusion"],
            stage=stage,
            tag=Metric.CONFUSION_MATRIX.value,
        )
        self._log_label_histogram(
            metrics["label_hist"],
            stage=stage,
            tag=Metric.LABEL_HISTOGRAM.value,
        )
        self._metrics[stage_key].reset()

    def _log_interval_metrics(
        self,
        stage: Stage,
        *,
        batch_idx: int,
        batch_size: int,
    ) -> None:
        if stage is not Stage.TRAIN:
            return
        interval = self.config.log_interval_steps
        if interval is None:
            return
        interval = int(interval)
        if interval <= 0 or (batch_idx + 1) % interval != 0:
            return

        if getattr(self.trainer, "sanity_checking", False):
            self._interval_metrics.reset()
            return

        metrics = self._interval_metrics.compute()
        if not metrics:
            self._interval_metrics.reset()
            return

        spearman = metrics["spearman"]
        if torch.isfinite(spearman):
            self._log_aux_scalars(
                {Metric.SPEARMAN_STEP: spearman},
                stage=stage,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch_size,
            )

        self._log_confusion_matrix(
            metrics["confusion"],
            stage=stage,
            tag=Metric.CONFUSION_MATRIX_STEP.value,
        )
        self._log_label_histogram(
            metrics["label_hist"],
            stage=stage,
            tag=Metric.LABEL_HISTOGRAM_STEP.value,
        )
        self._interval_metrics.reset()

    def _log_confusion_matrix(
        self,
        confusion: Tensor,
        *,
        stage: Stage,
        tag: str,
    ) -> None:
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)
        confusion_np = confusion.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.imshow(confusion_np, cmap="magma")
        ax.set_title(f"{stage.value} confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        self._log_figure(f"{stage.value}-figures/{tag}", fig)

    def _log_label_histogram(self, counts: Tensor, *, stage: Stage, tag: str) -> None:
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        xs = torch.arange(int(counts.shape[0])).cpu()
        counts_cpu = counts.detach().cpu()
        ax.bar(xs.numpy(), counts_cpu.numpy())
        ax.set_title(f"{stage.value} label histogram")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        self._log_figure(f"{stage.value}-figures/{tag}", fig)

    def _log_figure(self, tag: str, fig: plt.Figure) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            plt.close(fig)
            return
        experiment = getattr(logger, "experiment", None)
        if experiment is None:
            plt.close(fig)
            return
        try:
            if isinstance(logger, WandbLogger):
                import wandb

                experiment.log(
                    {
                        tag: wandb.Image(fig),
                        "epoch": int(self.current_epoch),
                    },
                )
                plt.close(fig)
                return
        except Exception:  # pragma: no cover - logger/optional deps guard
            pass
        if hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=int(self.global_step))
        plt.close(fig)

    def _log_loss_scalars(
        self,
        values: dict[str, Tensor | float],
        *,
        stage: Stage,
        on_step: bool,
        on_epoch: bool,
        prog_bar: bool,
        batch_size: int,
    ) -> None:
        prefix = f"{stage.value}/"
        self.log_dict(
            {f"{prefix}{key}": val for key, val in values.items()},
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
        )

    def _log_aux_scalars(
        self,
        values: dict[Metric | str, Tensor | float],
        *,
        stage: Stage,
        on_step: bool,
        on_epoch: bool,
        prog_bar: bool,
        batch_size: int,
    ) -> None:
        prefix = f"{stage.value}-aux/"
        payload: dict[str, Tensor | float] = {}
        for key, val in values.items():
            name = key.value if isinstance(key, Metric) else str(key)
            payload[f"{prefix}{name}"] = val
        self.log_dict(
            payload,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
        )

    def _maybe_init_bin_values(self) -> None:
        """Initialize learnable CORAL bin values from the fitted binner."""
        if self._binner is None:
            return
        head_coral = getattr(self.vin, "head_coral", None)
        if head_coral is None or not hasattr(self.vin, "init_bin_values"):
            return

        if self._binner.bin_means is not None:
            target = self._binner.bin_means
        else:
            target = self._binner.class_midpoints()

        device = next(self.vin.parameters()).device
        target = target.to(device=device, dtype=torch.float32)
        self.vin.init_bin_values(target, overwrite=False)

    def _load_binner_from_config(self) -> RriOrdinalBinner:
        if self.config.binner_path is None:
            raise RuntimeError(
                "Missing `VinLightningModuleConfig.binner_path`. Fit a binner first (e.g. via `nbv-fit-binner`) "
                "and point this config field to the resulting `rri_binner.json`, or resume from a checkpoint.",
            )

        resolved = PathConfig().resolve_artifact_path(
            self.config.binner_path,
            expected_suffix=".json",
            create_parent=False,
        )
        if not resolved.exists():
            raise FileNotFoundError(
                f"RRI binner not found at {resolved}. Run `nbv-fit-binner --out-dir <run_dir>` to create it "
                "or set `VinLightningModuleConfig.binner_path` to an existing fitted binner JSON.",
            )
        return RriOrdinalBinner.load(resolved)

    def _integrate_console(self) -> None:
        logger = getattr(self, "logger", None)
        if logger is not None:
            Console.integrate_with_logger(logger, global_step=int(self.global_step))

    def summarize_vin(
        self,
        batch: VinOracleBatch,
        *,
        include_torchsummary: bool = True,
        torchsummary_depth: int = 3,
    ) -> str:
        """Summarize VIN inputs/outputs for a single oracle-labeled batch.

        Args:
            batch: Oracle-labeled VIN batch from :class:`VinDataModule`.
            include_torchsummary: Whether to append torchsummary module summaries.
            torchsummary_depth: Max depth for torchsummary module traversal.

        Returns:
            Multiline string with VIN summary information.
        """
        return self.vin.summarize_vin(
            batch,
            include_torchsummary=include_torchsummary,
            torchsummary_depth=torchsummary_depth,
        )

    def plot_vin_encodings_batch(
        self,
        batch: VinOracleBatch,
        *,
        out_dir: Path,
        lmax: int,
        sh_normalization: str,
        radius_freqs: list[float],
        file_stem_prefix: str,
    ) -> dict[str, Path]:
        """Generate VIN encoding plots for a single oracle-labeled batch.

        Args:
            batch: Oracle-labeled VIN batch from :class:`VinDataModule`.
            out_dir: Output directory for plots.
            lmax: Max SH degree for visualization.
            sh_normalization: Spherical harmonics normalization mode.
            radius_freqs: Fourier frequencies for radius plot.
            file_stem_prefix: Filename prefix for the plots.

        Returns:
            Mapping of plot names to saved paths.
        """
        if isinstance(self.config.vin, VinModelV2Config):
            self.console.warn(
                "VIN v2 does not support SH/legacy encoding plots; returning empty plot set.",
            )
            return {}

        if batch.efm_snippet_view is None:
            raise RuntimeError(
                "VIN encoding plots require efm inputs; cached batches omit raw EFM data.",
            )

        was_training = self.vin.training
        self.vin.eval()
        with torch.no_grad():
            _, debug = self.vin.forward_with_debug(
                batch.efm_snippet_view.efm,
                candidate_poses_world_cam=batch.candidate_poses_world_cam,
                reference_pose_world_rig=batch.reference_pose_world_rig,
                p3d_cameras=batch.p3d_cameras,
                backbone_out=batch.backbone_out,
            )
        if was_training:
            self.vin.train()

        return plot_vin_encodings_from_debug(
            debug,
            out_dir=out_dir,
            lmax=int(lmax),
            sh_normalization=str(sh_normalization),
            radius_freqs=radius_freqs,
            file_stem_prefix=file_stem_prefix,
            pose_encoder_lff=self.vin.pose_encoder_lff,
        )


__all__ = [
    "AdamWConfig",
    "OneCycleSchedulerConfig",
    "ReduceLrOnPlateauConfig",
    "VinLightningModule",
    "VinLightningModuleConfig",
]
