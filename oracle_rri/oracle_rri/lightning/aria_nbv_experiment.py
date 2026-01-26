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

import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self
from zoneinfo import ZoneInfo

import torch
from pydantic import Field, field_validator, model_validator

from ..configs import OptunaConfig, PathConfig
from ..data.offline_cache import OracleRriCacheConfig, OracleRriCacheDatasetConfig
from ..utils import BaseConfig, Console, Stage
from ..utils.console import Verbosity
from .lit_datamodule import VinDataModule, VinDataModuleConfig
from .lit_module import VinLightningModule, VinLightningModuleConfig
from .lit_trainer_factory import TrainerFactoryConfig

if TYPE_CHECKING:
    import optuna
    import pytorch_lightning as pl


def _default_run_name() -> str:
    # Zone info uses IANA time zone database
    return datetime.now(tz=ZoneInfo("Europe/Berlin")).strftime("R%Y-%m-%d_%H-%M-%S")


class AriaNBVExperimentConfig(BaseConfig):
    """Top-level experiment config for VIN training/evaluation."""

    run_mode: Literal[
        "train",
        "fit_binner",
        "dump_config",
        "optuna",
        "summarize_vin",
        "plot_vin_encodings",
    ] = Field(
        default="train",
    )

    run_name: str = Field(default_factory=_default_run_name)
    """Human-readable run name (used for loggers and config snapshots)."""

    stage: Stage = Field(default=Stage.TRAIN)
    """Stage to execute when calling :meth:`setup_target_and_run`."""

    summary_stage: Stage = Field(default=Stage.TRAIN)
    """Stage used when running the VIN summary helper."""

    summary_num_batches: int = Field(default=1, gt=0)
    """Number of oracle batches to summarize when run_mode="summarize_vin"."""

    summary_include_torchsummary: bool = True
    """Whether to include torchsummary outputs in the VIN summary."""

    summary_torchsummary_depth: int = Field(default=3, gt=0)
    """Max depth for torchsummary module traversal."""

    plot_stage: Stage = Field(default=Stage.TRAIN)
    """Stage used when running VIN encoding plots."""

    plot_num_batches: int = Field(default=1, gt=0)
    """Number of oracle batches to plot when run_mode="plot_vin_encodings"."""

    plot_out_dir: Path = Field(
        default_factory=lambda: Path("docs") / "figures" / "impl" / "vin",
    )
    """Output directory for VIN encoding plots."""

    plot_lmax: int = Field(default=3, gt=0)
    """Max spherical-harmonics degree to visualize."""

    plot_sh_normalization: Literal["component", "norm"] = "component"
    """Spherical harmonics normalization mode for plots."""

    plot_radius_freqs: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    """Frequencies used for the radius Fourier feature plot."""

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

    module_config: VinLightningModuleConfig = Field(
        default_factory=VinLightningModuleConfig,
    )
    """VIN LightningModule config (VIN model + optimizer + binner settings)."""

    optuna_config: OptunaConfig | None = Field(default=None)
    """Optional Optuna configuration enabling hyperparameter sweeps."""

    verbosity: Verbosity = Verbosity.QUIET
    verbose: Verbosity = Verbosity.QUIET

    @field_validator("stage", mode="before")
    @classmethod
    def _coerce_stage(cls, value: Any) -> Stage:
        if isinstance(value, Stage):
            return value
        if isinstance(value, str):
            return Stage.from_str(value.lower())
        return Stage.from_str(str(value).lower())

    @field_validator("summary_stage", mode="before")
    @classmethod
    def _coerce_summary_stage(cls, value: Any) -> Stage:
        if isinstance(value, Stage):
            return value
        if isinstance(value, str):
            return Stage.from_str(value.lower())
        return Stage.from_str(str(value).lower())

    @field_validator("plot_stage", mode="before")
    @classmethod
    def _coerce_plot_stage(cls, value: Any) -> Stage:
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
        return self.paths.resolve_config_toml_path(
            f"{self.run_name}.toml",
            must_exist=False,
        )

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
    def _adapt_defaults(self) -> Self:
        out_dir = self.resolved_out_dir
        if self.run_mode in ("summarize_vin", "plot_vin_encodings"):
            object.__setattr__(self.trainer_config, "use_wandb", False)

        if self.fit_binner_only:
            object.__setattr__(self, "run_mode", "fit_binner")

        if self.trainer_config.callbacks.checkpoint_dir is None:
            object.__setattr__(
                self.trainer_config.callbacks,
                "checkpoint_dir",
                self.paths.checkpoints,
            )

        if self.module_config.binner_path is None and bool(
            self.module_config.save_binner,
        ):
            object.__setattr__(
                self.module_config,
                "binner_path",
                out_dir / "rri_binner.json",
            )

        if self.trainer_config.use_wandb and self.trainer_config.wandb_config.name is None:
            object.__setattr__(self.trainer_config.wandb_config, "name", self.run_name)

        return self

    # ------------------------------------------------------------------ orchestration
    def setup_target(  # type: ignore[override]
        self,
        setup_stage: Stage | str | None = None,
        *,
        trial: "optuna.Trial | None" = None,
    ) -> tuple[pl.Trainer, VinLightningModule, VinDataModule]:
        """Instantiate trainer + module + datamodule (no execution)."""
        resolved_stage = Stage.from_str(setup_stage) if setup_stage is not None else self.stage
        console = Console.with_prefix(self.__class__.__name__, "setup_target")
        console.log(
            f"Setting up components (stage={resolved_stage}, run_name={self.run_name})",
        )

        # Seed first so dataset shuffles and candidate sampling are reproducible.
        torch.manual_seed(int(self.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.seed))

        out_dir = self.resolved_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        trainer: pl.Trainer = self.trainer_config.setup_target(experiment=self, trial=trial)
        module: VinLightningModule = self.module_config.setup_target()
        datamodule: VinDataModule = self.datamodule_config.setup_target()

        datamodule.setup(stage=resolved_stage)
        return trainer, module, datamodule

    def setup_target_and_run(
        self,
        stage: Stage | str | None = None,
    ) -> pl.Trainer:
        """Instantiate components and execute the configured stage."""
        resolved_stage = Stage.from_str(stage) if stage is not None else self.stage
        trainer, module, datamodule = self.setup_target(setup_stage=resolved_stage)

        ckpt_path = self.paths.resolve_checkpoint_path(self.ckpt_path)
        ckpt_input = str(ckpt_path) if ckpt_path is not None else None
        match resolved_stage:
            case Stage.TRAIN:
                self._ensure_binner_matches_num_classes(datamodule=datamodule)
                trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_input)
                if bool(getattr(trainer, "interrupted", False)):
                    self._save_interrupt_checkpoint(trainer)
                    raise KeyboardInterrupt from None
            case Stage.VAL:
                trainer.validate(module, datamodule=datamodule, ckpt_path=ckpt_input)
            case Stage.TEST:
                trainer.test(module, datamodule=datamodule, ckpt_path=ckpt_input)

        return trainer

    def run_optuna_study(self) -> None:
        """Run an Optuna hyperparameter sweep for this experiment config."""
        if self.optuna_config is None:
            raise ValueError("OptunaConfig is not set!")

        try:
            import optuna  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "Optuna sweeps require `optuna` and `optuna-integration`. Install with "
                "`pip install optuna optuna-integration`.",
            ) from exc

        console = Console.with_prefix(self.__class__.__name__, "optuna")
        console.set_verbose(True)

        console.log(f"Starting Optuna study: {self.optuna_config.study_name}")
        console.log(f"Number of trials: {self.optuna_config.n_trials}")
        console.log(f"Monitoring metric: {self.optuna_config.monitor}")

        def _coerce_metric(raw_metric: object | None) -> float | None:
            if raw_metric is None:
                return None
            try:
                value = float(raw_metric.item()) if hasattr(raw_metric, "item") else float(raw_metric)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            return value

        def _fallback_metric(direction: str) -> float:
            return float("inf") if direction == "minimize" else float("-inf")

        def _get_metric_from_trainer(
            trainer: "pl.Trainer | None",
            monitor: str,
        ) -> tuple[float | None, str | None]:
            if trainer is None:
                return None, None
            monitor_key = str(monitor)
            suffixes = ("_epoch", "_step")

            candidate_keys: list[str] = [monitor_key]
            if any(monitor_key.endswith(suffix) for suffix in suffixes):
                base_key = monitor_key.rsplit("_", 1)[0]
                candidate_keys.append(base_key)
            else:
                candidate_keys.extend([f"{monitor_key}_epoch", f"{monitor_key}_step"])

            for source_name in ("callback_metrics", "logged_metrics", "progress_bar_metrics"):
                metrics = getattr(trainer, source_name, None)
                if not metrics:
                    continue
                for key in candidate_keys:
                    if key in metrics:
                        metric_value = _coerce_metric(metrics.get(key))
                        if metric_value is not None:
                            return metric_value, source_name
            ckpt_callback = getattr(trainer, "checkpoint_callback", None)
            if ckpt_callback is not None:
                metric_value = _coerce_metric(getattr(ckpt_callback, "best_model_score", None))
                if metric_value is not None:
                    return metric_value, "checkpoint_callback.best_model_score"
            return None, None

        def _check_pruned(trainer: "pl.Trainer", trial_console: Console) -> None:
            """Raise TrialPruned if Optuna requested pruning."""
            for callback in trainer.callbacks:
                check_pruned = getattr(callback, "check_pruned", None)
                if callable(check_pruned):
                    trial_console.log("Optuna pruning check executed.")
                    check_pruned()
                    return
            trial_console.warn("Optuna pruning enabled but no pruning callback was found.")

        if self.trainer_config.use_wandb:
            self.trainer_config.wandb_config.group = "optuna"
            self.trainer_config.wandb_config.job_type = f"Opt:{self.run_name}"
            tags = list(self.trainer_config.wandb_config.tags or [])
            if "optuna" not in tags:
                tags.append("optuna")
            self.trainer_config.wandb_config.tags = tags
            console.log("W&B configured for Optuna study")

        study = self.optuna_config.setup_target()
        console.log(f"Running optimization with {self.optuna_config.n_trials} trials...")

        def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
            trial_console = Console.with_prefix(self.__class__.__name__, f"trial_{trial.number}")
            trial_console.set_verbose(True)

            trial_console.log(f"Starting trial {trial.number}")

            experiment_config_copy = deepcopy(self)
            assert experiment_config_copy.optuna_config is not None

            trial_run_name = f"{self.run_name}_T{trial.number}"
            experiment_config_copy.run_name = trial_run_name
            experiment_config_copy.out_dir = Path(self.out_dir) / trial_run_name

            if experiment_config_copy.trainer_config.use_wandb:
                experiment_config_copy.trainer_config.wandb_config.name = trial_run_name
                trial_console.log(f"W&B run name: {trial_run_name}")

            experiment_config_copy.optuna_config.setup_optimizables(
                experiment_config_copy,
                trial,
            )

            trainer: "pl.Trainer | None" = None
            monitor = str(experiment_config_copy.optuna_config.monitor)
            metric = _fallback_metric(experiment_config_copy.optuna_config.direction)
            metric_source = "fallback"

            try:
                trainer, module, datamodule = experiment_config_copy.setup_target(
                    setup_stage=self.stage,
                    trial=trial,
                )
                experiment_config_copy._ensure_binner_matches_num_classes(
                    datamodule=datamodule,
                )
                trainer.fit(module, datamodule=datamodule)

                if experiment_config_copy.trainer_config.callbacks.use_optuna_pruning:
                    _check_pruned(trainer, trial_console)

                extracted_metric, source = _get_metric_from_trainer(trainer, monitor)
                if extracted_metric is None:
                    trial_console.warn(f"No '{monitor}' metric found after training; using fallback {metric:.4f}.")
                else:
                    metric = extracted_metric
                    metric_source = source or "unknown"
            except optuna.TrialPruned as exc:
                trial_console.warn(f"Trial {trial.number} pruned: {exc}")
                raise
            except KeyboardInterrupt:
                trial_console.warn("Keyboard interrupt received. Stopping Optuna study.")
                raise
            except Exception as exc:  # noqa: BLE001
                trial_console.error(f"Trial {trial.number} failed: {exc}")
                extracted_metric, source = _get_metric_from_trainer(trainer, monitor)
                if extracted_metric is not None:
                    metric = extracted_metric
                    metric_source = f"failed:{source}"
                    trial_console.warn(f"Recovered '{monitor}' from {source}: {metric:.4f}.")
                else:
                    metric_source = "failed:fallback"
                    trial_console.warn(f"No '{monitor}' metric available after failure; using fallback {metric:.4f}.")
                trial.set_user_attr("failed_reason", repr(exc))
            finally:
                experiment_config_copy.optuna_config.log_to_wandb()

                if self.optuna_config is not None:
                    self.optuna_config.suggested_params = experiment_config_copy.optuna_config.suggested_params.copy()

                try:
                    import wandb  # type: ignore[import-not-found]
                except ModuleNotFoundError:
                    wandb = None
                if wandb is not None and wandb.run is not None:
                    wandb.finish()

            trial.set_user_attr("metric_source", metric_source)
            trial_console.log(
                f"Trial {trial.number} finished with {monitor}: {metric:.4f}",
            )
            console.dbg(f"Trial {trial.number} params: {trial.params}")
            return metric

        try:
            study.optimize(objective, n_trials=int(self.optuna_config.n_trials))
        except KeyboardInterrupt:
            console.warn("Keyboard interrupt received. Stopping Optuna study.")
            return

        if hasattr(study, "best_value") and hasattr(study, "best_params"):
            console.log(f"Optuna study completed. Best value: {study.best_value:.4f}")
            console.log(f"Best params: {study.best_params}")
        else:
            console.log("Optuna study completed")

    def summarize_vin(self) -> None:
        """Run VIN summary on a real oracle batch from the datamodule."""
        stage = self.summary_stage
        console = Console.with_prefix(self.__class__.__name__, "summarize_vin")
        self._maybe_use_offline_cache_for_summary(stage=stage, console=console)
        _, module, datamodule = self.setup_target(setup_stage=stage)

        plan = datamodule._build_stage_plan(stage)
        if plan.use_batching:
            if stage is Stage.TRAIN:
                iterator = iter(datamodule.train_dataloader())
            else:
                iterator = iter(datamodule.val_dataloader())
        else:
            iterator = datamodule.iter_oracle_batches(stage=stage)
        for idx in range(int(self.summary_num_batches)):
            batch = next(iterator, None)
            if batch is None:
                raise RuntimeError(
                    "VIN summary requested but no oracle batches were yielded. "
                    "Check that the oracle cache is populated or that the online labeler can produce batches.",
                )
            header = (
                f"VIN summary batch {idx + 1}/{self.summary_num_batches} "
                f"(stage={stage}, scene_id={batch.scene_id}, snippet_id={batch.snippet_id})"
            )
            console.log(header)
            summary = module.summarize_vin(
                batch,
                include_torchsummary=self.summary_include_torchsummary,
                torchsummary_depth=self.summary_torchsummary_depth,
            )
            print(summary)

    def _maybe_use_offline_cache_for_summary(
        self,
        *,
        stage: Stage,
        console: Console,
    ) -> None:
        """Enable offline cache for summary runs when available."""
        from ..data.vin_oracle_datasets import VinOracleCacheDatasetConfig

        if isinstance(self.datamodule_config.source, VinOracleCacheDatasetConfig):
            return

        cache_cfg = OracleRriCacheConfig(paths=self.paths)
        if not cache_cfg.index_path.exists() or not cache_cfg.metadata_path.exists():
            console.log("No oracle cache found; using online oracle labeler.")
            return

        try:
            payload = json.loads(cache_cfg.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            console.warn(f"Invalid cache metadata at {cache_cfg.metadata_path}: {exc}")
            return

        include_backbone = payload.get("include_backbone")
        include_depths = payload.get("include_depths")
        include_pointclouds = payload.get("include_pointclouds")
        if not include_backbone:
            console.warn(
                "Offline cache missing backbone outputs; using online oracle labeler.",
            )
            return
        if include_depths is False or include_pointclouds is False:
            console.warn(
                "Offline cache missing depths/pointclouds; using online oracle labeler.",
            )
            return

        cache_dataset_cfg = OracleRriCacheDatasetConfig(
            cache=cache_cfg,
            load_backbone=True,
        )
        cache_source_cfg = VinOracleCacheDatasetConfig(cache=cache_dataset_cfg)

        object.__setattr__(self.datamodule_config, "source", cache_source_cfg)
        console.log(f"Using offline oracle cache at {cache_cfg.cache_dir} (map_location=cpu).")

    def plot_vin_encodings(self) -> None:
        """Generate VIN encoding plots using real oracle batches."""
        stage = self.plot_stage
        console = Console.with_prefix(self.__class__.__name__, "plot_vin_encodings")
        _, module, datamodule = self.setup_target(setup_stage=stage)

        out_dir = self.paths.resolve_under_root(self.plot_out_dir)
        iterator = datamodule.iter_oracle_batches(stage=stage)
        for idx in range(int(self.plot_num_batches)):
            batch = next(iterator)
            stem = f"{batch.scene_id}_{batch.snippet_id}".replace("/", "_")
            batch_dir = out_dir / stem
            plots = module.plot_vin_encodings_batch(
                batch,
                out_dir=batch_dir,
                lmax=int(self.plot_lmax),
                sh_normalization=str(self.plot_sh_normalization),
                radius_freqs=list(self.plot_radius_freqs),
                file_stem_prefix=stem,
            )
            console.log(
                f"Saved VIN encoding plots for batch {idx + 1}/{self.plot_num_batches} "
                f"(scene_id={batch.scene_id}, snippet_id={batch.snippet_id})",
            )
            for label, path in plots.items():
                console.log(f"  {label}: {path}")

    def _interrupt_checkpoint_path(self, trainer: pl.Trainer) -> Path:
        """Build the checkpoint path used when training is interrupted.

        Args:
            trainer: Lightning trainer instance used for the run.

        Returns:
            Absolute checkpoint path under :attr:`PathConfig.checkpoints`.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        epoch = int(getattr(trainer, "current_epoch", 0))
        step = int(getattr(trainer, "global_step", 0))
        filename = f"{self.run_name}-interrupt-epoch={epoch}-step={step}-{timestamp}.ckpt"
        return self.paths.resolve_artifact_path(
            self.paths.checkpoints / filename,
            expected_suffix=".ckpt",
        )

    def _save_interrupt_checkpoint(self, trainer: pl.Trainer) -> Path | None:
        """Persist a checkpoint when training is interrupted by Ctrl+C.

        Args:
            trainer: Lightning trainer instance used for the run.

        Returns:
            Path to the saved checkpoint, or None if not saved (e.g. non-zero rank).
        """
        console = Console.with_prefix(self.__class__.__name__, "interrupt_checkpoint")
        if not bool(getattr(trainer, "is_global_zero", True)):
            console.warn("Skipping interrupt checkpoint on non-global rank.")
            return None
        checkpoint_path = self._interrupt_checkpoint_path(trainer)
        trainer.save_checkpoint(checkpoint_path.as_posix())
        console.log(f"Saved interrupt checkpoint: {checkpoint_path}")
        return checkpoint_path

    def fit_binner_and_save(
        self,
        datamodule: VinDataModule | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Fit `RriOrdinalBinner` from oracle batches and persist it.

        Args:
            datamodule: Optional pre-built datamodule (avoids re-instantiation).
            overwrite: Overwrite existing binner JSON when True.

        Returns:
            Path to the saved `rri_binner.json`.
        """
        from ..rri_metrics.rri_binning import RriOrdinalBinner

        out_dir = self.resolved_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        console = Console.with_prefix(self.__class__.__name__, "fit_binner")
        if datamodule is None:
            datamodule = self.datamodule_config.setup_target()

        train_loader = datamodule.train_dataloader()

        fit_snippets = self.module_config.binner_fit_snippets or len(train_loader)
        max_skips = self.module_config.binner_max_attempts

        fit_data_path = self.paths.resolve_artifact_path(
            out_dir / "rri_binner_fit_data.pt",
            expected_suffix=".pt",
        )

        def _iter_rri():
            for batch in train_loader:
                rri = batch.rri.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
                meta = {"scene_id": batch.scene_id, "snippet_id": batch.snippet_id}
                yield rri, meta

        def _on_progress(
            successes: int,
            skipped: int,
            rri: torch.Tensor | None,
            meta: object | None,
        ) -> None:
            if rri is None:
                return
            if not isinstance(meta, dict):
                return
            scene_id = meta.get("scene_id", "unknown")
            snippet_id = meta.get("snippet_id", "unknown")
            console.log(
                f"fit[{successes:02d}/{fit_snippets:02d}] scene={scene_id} snip={snippet_id} "
                f"C={int(rri.numel())} rri_mean={float(rri.mean().item()):.4f} rri_std={float(rri.std().item()):.4f}",
            )

        console.log(
            f"Fitting RRI ordinal binner on {fit_snippets} snippets (resume={fit_data_path.exists()}).",
        )
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
        binner_path = self.paths.resolve_artifact_path(
            binner_path,
            expected_suffix=".json",
        )

        saved_path = binner.save(binner_path, overwrite=overwrite)
        console.log(f"Saved binner: {saved_path}")
        return saved_path

    def _ensure_binner_matches_num_classes(
        self,
        *,
        datamodule: VinDataModule | None = None,
    ) -> None:
        """Refit the ordinal binner if the stored num_classes mismatches the config."""
        if self.module_config.binner_path is None:
            return

        resolved = self.paths.resolve_artifact_path(
            self.module_config.binner_path,
            expected_suffix=".json",
            create_parent=False,
        )
        if not resolved.exists():
            return

        from ..rri_metrics.rri_binning import RriOrdinalBinner

        binner = RriOrdinalBinner.load(resolved)
        expected = int(self.module_config.num_classes)
        if int(binner.num_classes) == expected:
            return

        console = Console.with_prefix(self.__class__.__name__, "binner_check")
        console.warn(
            f"RRI binner num_classes mismatch (found {int(binner.num_classes)}, expected {expected}); "
            "refitting with current num_classes.",
        )
        saved_path = self.fit_binner_and_save(datamodule=datamodule, overwrite=True)
        object.__setattr__(self.module_config, "binner_path", saved_path)

    def run(self) -> None:
        """Execute the configured action.

        This method is intended to be called from CLI entry points.
        """
        match self.run_mode:
            case "dump_config":
                self.save_config()
            case "fit_binner":
                self.fit_binner_and_save()
            case "optuna":
                self.run_optuna_study()
            case "train":
                self.setup_target_and_run()
            case "summarize_vin":
                self.summarize_vin()
            case "plot_vin_encodings":
                self.plot_vin_encodings()


__all__ = ["AriaNBVExperimentConfig"]
