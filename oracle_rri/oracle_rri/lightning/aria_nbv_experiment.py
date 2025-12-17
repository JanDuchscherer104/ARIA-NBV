"""Experiment-style orchestration for VIN training (Lightning).

This follows the pattern used in `external/doc_classifier/configs/experiment_config.py`:

- A single top-level config (`AriaNBVExperimentConfig`) composes:
  - `VinDataModuleConfig` (which composes `AseEfmDatasetConfig` + `OracleRriLabelerConfig`)
  - `VinLightningModuleConfig` (which composes `VinModelConfig`)
  - `TrainerFactoryConfig` (which composes `TrainerCallbacksConfig` + optional W&B config)

The goal is to keep configuration nesting intact so runs are reproducible via a
single TOML file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import torch
from pydantic import Field, field_validator, model_validator

from ..configs import PathConfig
from ..utils import BaseConfig, Console, Stage
from .lit_datamodule import VinDataModule, VinDataModuleConfig
from .lit_module import VinLightningModule, VinLightningModuleConfig
from .lit_trainer_factory import TrainerFactoryConfig

if TYPE_CHECKING:
    import pytorch_lightning as pl


def _default_run_name() -> str:
    return datetime.now().strftime("R%Y-%m-%d_%H-%M-%S")


class AriaNBVExperimentConfig(BaseConfig):
    """Top-level experiment config for VIN training/evaluation."""

    run_mode: Literal["train", "fit_binner", "dump_config"] = Field(default="train")

    run_name: str = Field(default_factory=_default_run_name)
    """Human-readable run name (used for loggers and config snapshots)."""

    stage: Stage = Field(default=Stage.TRAIN)
    """Stage to execute when calling :meth:`setup_target_and_run`."""

    ckpt_path: Path | None = None
    """Optional checkpoint path for val/test (or resume training)."""

    seed: int = 0
    """Random seed applied via torch (and CUDA if available)."""

    out_dir: Path = Field(default_factory=lambda: Path(".logs") / "vin")
    """Output directory for run artifacts (config snapshots, checkpoints, binner)."""

    fit_binner_only: bool = False
    """Fit the ordinal binner on oracle labels, save it, and exit (no training)."""

    paths: PathConfig = Field(default_factory=PathConfig)
    """Filesystem layout config (singleton)."""

    trainer_config: TrainerFactoryConfig = Field(default_factory=TrainerFactoryConfig)
    """Lightning trainer factory config (callbacks + logger)."""

    datamodule_config: VinDataModuleConfig = Field(default_factory=VinDataModuleConfig)
    """VIN datamodule config (datasets + oracle labeler)."""

    module_config: VinLightningModuleConfig = Field(default_factory=VinLightningModuleConfig)
    """VIN LightningModule config (VIN model + optimizer + binner settings)."""

    @field_validator("stage", mode="before")
    @classmethod
    def _coerce_stage(cls, value: Any) -> Stage:
        if isinstance(value, Stage):
            return value
        if isinstance(value, str):
            return Stage.from_str(value.lower())
        return Stage.from_str(str(value).lower())

    @field_validator("run_mode", mode="before")
    @classmethod
    def _coerce_run_mode(cls, value: Any) -> str:
        if isinstance(value, str):
            return value.strip().lower().replace("-", "_")
        return str(value).strip().lower().replace("-", "_")

    @field_validator("seed", mode="before")
    @classmethod
    def _coerce_seed(cls, value: Any) -> int:
        return int(value)

    @property
    def resolved_out_dir(self) -> Path:
        return self.paths.resolve_run_dir(self.out_dir)

    @property
    def default_config_path(self) -> Path:
        """Default path for saving the experiment TOML."""

        return self.paths.resolve_config_toml_path(f"{self.run_name}.toml", must_exist=False)

    def save_config(
        self,
        path: Path | str | None = None,
        *,
        include_comments: bool = True,
        include_type_hints: bool = True,
    ) -> Path:
        """Save this config (and nested configs) as TOML.

        Args:
            path: Destination TOML path. If None, uses :meth:`default_config_path`.
            include_comments: Include docstring comments in the TOML.
            include_type_hints: Include type hints in the TOML comments.

        Returns:
            Resolved path that was written.
        """
        Console.with_prefix(self.__class__.__name__, "save_config").log(
            f"Saving config to {path or self.default_config_path}",
        )
        resolved = (
            self.paths.resolve_config_toml_path(path, must_exist=False)
            if path is not None
            else self.default_config_path
        )
        return self.save_toml(
            resolved,
            include_comments=include_comments,
            include_type_hints=include_type_hints,
        )

    @model_validator(mode="after")
    def _default_paths(self) -> Self:
        out_dir = self.resolved_out_dir

        if self.fit_binner_only:
            object.__setattr__(self, "run_mode", "fit_binner")

        if self.trainer_config.callbacks.checkpoint_dir is None:
            object.__setattr__(self.trainer_config.callbacks, "checkpoint_dir", out_dir / "checkpoints")

        if self.module_config.binner_path is None and bool(self.module_config.save_binner):
            object.__setattr__(self.module_config, "binner_path", out_dir / "rri_binner.json")

        if self.trainer_config.use_wandb and self.trainer_config.wandb_config.name is None:
            object.__setattr__(self.trainer_config.wandb_config, "name", self.run_name)

        return self

    # ------------------------------------------------------------------ orchestration
    def setup_target(  # type: ignore[override]
        self,
        setup_stage: Stage | str | None = None,
    ) -> tuple["pl.Trainer", VinLightningModule, VinDataModule]:
        """Instantiate trainer + module + datamodule (no execution)."""

        resolved_stage = Stage.from_str(setup_stage) if setup_stage is not None else self.stage
        console = Console.with_prefix(self.__class__.__name__, "setup_target")
        console.log(f"Setting up components (stage={resolved_stage}, run_name={self.run_name})")

        # Seed first so dataset shuffles and candidate sampling are reproducible.
        torch.manual_seed(int(self.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.seed))

        out_dir = self.resolved_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Persist a config snapshot alongside run artifacts.
        self.save_toml(out_dir / "config.toml", include_comments=True, include_type_hints=False)

        trainer: pl.Trainer = self.trainer_config.setup_target()
        module: VinLightningModule = self.module_config.setup_target()
        datamodule: VinDataModule = self.datamodule_config.setup_target()

        datamodule.setup(stage=resolved_stage)
        return trainer, module, datamodule

    def setup_target_and_run(
        self,
        stage: Stage | str | None = None,
    ) -> "pl.Trainer":
        """Instantiate components and execute the configured stage."""

        resolved_stage = Stage.from_str(stage) if stage is not None else self.stage
        trainer, module, datamodule = self.setup_target(setup_stage=resolved_stage)

        ckpt_path = self.paths.resolve_checkpoint_path(self.ckpt_path)
        ckpt_input = str(ckpt_path) if ckpt_path is not None else None
        match resolved_stage:
            case Stage.TRAIN:
                trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_input)
            case Stage.VAL:
                trainer.validate(module, datamodule=datamodule, ckpt_path=ckpt_input)
            case Stage.TEST:
                trainer.test(module, datamodule=datamodule, ckpt_path=ckpt_input)

        return trainer

    def fit_binner_and_save(self) -> Path:
        """Fit `RriOrdinalBinner` from oracle batches and persist it.

        Returns:
            Path to the saved `rri_binner.json`.
        """

        from ..vin import RriOrdinalBinner

        out_dir = self.resolved_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        console = Console.with_prefix(self.__class__.__name__, "fit_binner")
        datamodule: VinDataModule = self.datamodule_config.setup_target()

        train_loader = datamodule.train_dataloader()

        fit_snippets = self.module_config.binner_fit_snippets
        max_skips = self.module_config.binner_max_attempts

        fit_data_path = self.paths.resolve_artifact_path(out_dir / "rri_binner_fit_data.pt", expected_suffix=".pt")

        def _iter_rri():
            for batch in train_loader:
                rri = batch.rri.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
                meta = {"scene_id": batch.scene_id, "snippet_id": batch.snippet_id}
                yield rri, meta

        def _on_progress(successes: int, skipped: int, rri: torch.Tensor | None, meta: object | None) -> None:
            if rri is None:
                return
            if not isinstance(meta, dict):
                return
            scene_id = meta.get("scene_id", "unknown")
            snippet_id = meta.get("snippet_id", "unknown")
            console.log(
                f"fit[{successes:02d}/{fit_snippets:02d}] scene={scene_id} snip={snippet_id} "
                f"C={int(rri.numel())} rri_mean={float(rri.mean().item()):.4f} rri_std={float(rri.std().item()):.4f}"
            )

        console.log(f"Fitting RRI ordinal binner on {fit_snippets} snippets (resume={fit_data_path.exists()}).")
        binner = RriOrdinalBinner.fit_from_iterable(
            _iter_rri(),
            num_classes=self.module_config.num_classes,
            target_items=fit_snippets,
            max_skips=max_skips,
            fit_data_path=fit_data_path,
            resume=True,
            save_every=10,
            on_progress=_on_progress,
        )

        binner_path = self.module_config.binner_path or (out_dir / "rri_binner.json")
        binner_path = self.paths.resolve_artifact_path(binner_path, expected_suffix=".json")

        saved_path = binner.save(binner_path)
        console.log(f"Saved binner: {saved_path}")
        return saved_path

    def run(self) -> None:
        """Execute the configured action.

        This method is intended to be called from CLI entry points.
        """

        match self.run_mode:
            case "dump_config":
                self.save_config()
            case "fit_binner":
                self.fit_binner_and_save()
            case "train":
                self.setup_target_and_run()


__all__ = ["AriaNBVExperimentConfig"]
