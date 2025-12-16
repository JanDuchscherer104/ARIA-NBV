"""Trainer factory with optional W&B integration.

This is adapted from `external/doc_classifier/lightning/lit_trainer_factory.py` and keeps the
configuration surface small while supporting:

- reproducible debug runs (`fast_dev_run`, CPU forcing, anomaly detection),
- TF32 matmul control, and
- W&B logging via `WandbConfig`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import pytorch_lightning as pl
import torch
from pydantic import Field, model_validator

from ..configs.wandb_config import WandbConfig
from ..utils import BaseConfig, Console
from .lit_trainer_callbacks import TrainerCallbacksConfig


class TrainerFactoryConfig(BaseConfig):
    """Configuration for constructing a PyTorch Lightning trainer."""

    target: type[pl.Trainer] = Field(default_factory=lambda: pl.Trainer, exclude=True)

    is_debug: bool = False
    """Set fast_dev_run to True, use CPU, set num_workers to 0, disable checkpointing when True."""

    fast_dev_run: bool = False
    """Run 1 batch per split to sanity-check the full loop."""

    accelerator: str = "auto"
    devices: int | str | Sequence[int] = "auto"
    strategy: str | None = "auto"

    max_epochs: int | None = 10
    precision: str | int = "32"

    tf32_matmul_precision: str | None = "medium"
    gradient_clip_val: float | None = None
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 1
    deterministic: bool | str | None = None

    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    check_val_every_n_epoch: int = 1

    enable_model_summary: bool = True
    """Enable Lightning's default model summary callback.

    When using :class:`~pytorch_lightning.callbacks.RichModelSummary`, consider
    disabling this to avoid duplicate summaries.
    """

    callbacks: TrainerCallbacksConfig = Field(default_factory=TrainerCallbacksConfig)
    """Trainer callbacks configuration."""

    use_wandb: bool = True
    """Whether to enable W&B logging."""

    wandb_config: WandbConfig = Field(default_factory=WandbConfig)
    """W&B logger configuration (used when use_wandb=True)."""

    @model_validator(mode="after")
    def _debug_defaults(self) -> Self:
        console = Console.with_prefix(self.__class__.__name__, "_debug_defaults")

        if self.is_debug:
            object.__setattr__(self, "fast_dev_run", True)
            object.__setattr__(self, "accelerator", "cpu")
            object.__setattr__(self, "devices", 1)
            object.__setattr__(self.callbacks, "use_model_checkpoint", False)
            torch.autograd.set_detect_anomaly(True)
            console.log(
                "Debug settings: fast_dev_run=True, accelerator=cpu, devices=1, checkpointing disabled, "
                "anomaly detection enabled"
            )

        if self.fast_dev_run:
            Console.with_prefix(self.__class__.__name__).log(
                "Fast dev run enabled; trainer will use a single batch per split.",
            )
        return self

    def setup_target(self, experiment: Any | None = None) -> pl.Trainer:  # type: ignore[override]
        """Instantiate the configured trainer."""
        console = Console.with_prefix(self.__class__.__name__, "setup_target")

        if self.tf32_matmul_precision is not None:
            try:
                torch.set_float32_matmul_precision(str(self.tf32_matmul_precision))
                console.log(f"Set TF32 matmul precision to '{self.tf32_matmul_precision}'")
            except Exception as exc:  # pragma: no cover - hardware dependent
                console.warn(f"Failed to set TF32 matmul precision: {exc}")

        logger = None
        if self.is_debug:
            logger = True
            console.log("Using default logger (debug mode)")
        elif self.use_wandb:
            logger = self.wandb_config.setup_target()
            console.log(f"Using W&B logger: {self.wandb_config.name}")
        else:
            console.log("No logger configured")

        callbacks = self.callbacks.setup_target(model_name=None, has_logger=logger is not None)
        console.log(f"Configured {len(callbacks)} callbacks.")

        return pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            max_epochs=self.max_epochs,
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            accumulate_grad_batches=self.accumulate_grad_batches,
            log_every_n_steps=self.log_every_n_steps,
            fast_dev_run=self.fast_dev_run,
            deterministic=self.deterministic,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            enable_model_summary=bool(self.enable_model_summary),
            callbacks=callbacks,
            logger=logger,
        )


__all__ = ["TrainerFactoryConfig"]
