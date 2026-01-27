"""W&B configuration for Lightning."""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pytorch_lightning.loggers import WandbLogger

from ..utils import BaseConfig
from .path_config import PathConfig


class WandbConfig(BaseConfig):
    """Wrapper around Lightning's `WandbLogger`.

    References:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html
    """

    @property
    def target(self) -> type[WandbLogger]:
        return WandbLogger

    name: str | None = Field(default=None, description="Display name for the run.")
    project: str = Field(default="aria-nbv", description="W&B project name.")
    entity: str | None = None
    offline: bool = Field(False, description="Enable offline logging.")
    log_model: bool | str = Field(
        default=False,
        description="Forward Lightning checkpoints to W&B artefacts.",
    )
    checkpoint_name: str | None = Field(default=None, description="Checkpoint artefact name.")
    tags: list[str] | None = Field(default=None, description="Optional list of tags.")
    group: str | None = Field(default=None, description="Group multiple related runs.")
    job_type: str | None = Field(default=None, description="Attach a W&B job_type label.")
    prefix: str | None = Field(default=None, description="Namespace prefix for metric keys.")

    def setup_target(self, **kwargs: Any) -> WandbLogger:
        """Instantiate a configured `WandbLogger`."""
        wandb_dir = PathConfig().wandb.as_posix()

        return WandbLogger(
            name=self.name,
            project=self.project,
            entity=self.entity,
            save_dir=wandb_dir,
            offline=self.offline,
            log_model=self.log_model,
            prefix=self.prefix,
            checkpoint_name=self.checkpoint_name,
            tags=self.tags,
            group=self.group,
            job_type=self.job_type,
            **(kwargs or {}),
        )


__all__ = ["WandbConfig"]
