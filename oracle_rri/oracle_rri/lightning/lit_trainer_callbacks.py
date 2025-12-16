"""Configurable Lightning Trainer callbacks (checkpointing, progress bars, LR monitor)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pytorch_lightning.callbacks import (
    BackboneFinetuning,
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    Timer,
    TQDMProgressBar,
)

from ..configs import PathConfig
from ..utils import BaseConfig, Console


class CustomTQDMProgressBar(TQDMProgressBar):
    """Custom TQDM progress bar that hides the version number (v_num)."""

    def get_metrics(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


class CustomRichProgressBar(RichProgressBar):
    """Custom Rich progress bar that hides the version number (v_num)."""

    def get_metrics(self, trainer, pl_module):  # type: ignore[no-untyped-def]
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


class TrainerCallbacksConfig(BaseConfig[list]):
    """Configuration for standard trainer callbacks."""

    target: type[list] = Field(default_factory=lambda: list, exclude=True)
    """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""

    use_model_checkpoint: bool = True
    checkpoint_monitor: str = "val_loss"
    """Metric to monitor for model checkpointing."""
    checkpoint_mode: str = "min"
    """Mode for checkpoint monitor ("min" or "max")."""
    checkpoint_dir: Path | None = None
    """Directory to save checkpoints. If None, uses `PathConfig().checkpoints`."""
    checkpoint_filename: str = "epoch={epoch}-step={step}-val_loss={val_loss:.4f}"
    """Filename template for checkpoints. Prefer metric keys without '/' to keep templates simple."""
    checkpoint_save_top_k: int = 1
    """Number of best models to save."""
    checkpoint_auto_insert_metric_name: bool = False
    """Whether Lightning should auto-prefix metric names in the filename."""

    use_early_stopping: bool = False
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = 5

    use_lr_monitor: bool = True
    lr_logging_interval: str = "epoch"

    use_rich_progress_bar: bool = False
    """Enable Rich progress bar for enhanced terminal output (mutually exclusive with TQDM)."""
    use_tqdm_progress_bar: bool = True
    """Enable TQDM progress bar (mutually exclusive with Rich)."""
    tqdm_refresh_rate: int = 1

    use_rich_model_summary: bool = True
    rich_summary_max_depth: int = 1

    use_backbone_finetuning: bool = False
    backbone_unfreeze_at_epoch: int = 10
    backbone_lambda_func: str | None = None
    backbone_train_bn: bool = True

    use_timer: bool = False
    timer_duration: dict[str, int] | None = None
    timer_interval: str = "step"

    @model_validator(mode="after")
    def _validate_progress_bars_mutually_exclusive(self) -> Self:
        if self.use_rich_progress_bar and self.use_tqdm_progress_bar:
            raise ValueError("use_rich_progress_bar and use_tqdm_progress_bar are mutually exclusive. Enable only one.")
        return self

    def setup_target(self, model_name: str | None = None, *, has_logger: bool = True) -> list[Callback]:  # type: ignore[override]
        console = Console.with_prefix(self.__class__.__name__, "setup_target")
        callbacks: list[Callback] = []

        if self.use_model_checkpoint:
            dirpath = self.checkpoint_dir if self.checkpoint_dir is not None else PathConfig().checkpoints
            dirpath.mkdir(parents=True, exist_ok=True)

            ckpt_fn = f"{model_name}-{self.checkpoint_filename}" if model_name else self.checkpoint_filename
            callbacks.append(
                ModelCheckpoint(
                    monitor=str(self.checkpoint_monitor),
                    mode=str(self.checkpoint_mode),
                    save_top_k=int(self.checkpoint_save_top_k),
                    filename=str(ckpt_fn),
                    auto_insert_metric_name=bool(self.checkpoint_auto_insert_metric_name),
                    dirpath=dirpath.as_posix(),
                ),
            )
            console.log(f"ModelCheckpoint active: monitor={self.checkpoint_monitor} dir={dirpath} template={ckpt_fn}")

        if self.use_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=str(self.early_stopping_monitor),
                    mode=str(self.early_stopping_mode),
                    patience=int(self.early_stopping_patience),
                ),
            )

        if self.use_lr_monitor and has_logger:
            callbacks.append(LearningRateMonitor(logging_interval=str(self.lr_logging_interval)))

        if self.use_rich_progress_bar:
            callbacks.append(CustomRichProgressBar())

        if self.use_tqdm_progress_bar:
            callbacks.append(CustomTQDMProgressBar(refresh_rate=int(self.tqdm_refresh_rate)))

        if self.use_rich_model_summary:
            callbacks.append(RichModelSummary(max_depth=int(self.rich_summary_max_depth)))

        if self.use_backbone_finetuning:
            callbacks.append(
                BackboneFinetuning(
                    unfreeze_backbone_at_epoch=int(self.backbone_unfreeze_at_epoch),
                    lambda_func=eval(self.backbone_lambda_func) if self.backbone_lambda_func else None,
                    backbone_initial_ratio_lr=0.1,
                    should_align=True,
                    train_bn=bool(self.backbone_train_bn),
                ),
            )

        if self.use_timer:
            callbacks.append(
                Timer(
                    duration=timedelta(**self.timer_duration) if self.timer_duration else None,
                    interval=str(self.timer_interval),
                ),
            )

        return callbacks


__all__ = ["TrainerCallbacksConfig"]
