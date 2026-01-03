from __future__ import annotations

from typing import Any, Literal

import pytorch_lightning as pl
from pydantic import Field
from torch import Tensor
from torch.nn import functional as functional
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from ..utils import BaseConfig, Optimizable, optimizable_field


class AdamWConfig(BaseConfig[Optimizer]):
    """AdamW optimizer configuration for VIN."""

    target: type[Optimizer] = Field(default_factory=lambda: AdamW, exclude=True)
    """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""

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

    target: type[ReduceLROnPlateau] = Field(
        default_factory=lambda: ReduceLROnPlateau,
        exclude=True,
    )
    """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""

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

    target: type[OneCycleLR] = Field(default_factory=lambda: OneCycleLR, exclude=True)
    """Factory target for :meth:`~oracle_rri.utils.base_config.BaseConfig.setup_target`."""

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
