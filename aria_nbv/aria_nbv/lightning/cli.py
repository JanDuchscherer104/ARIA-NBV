"""CLI entry points for `AriaNBVExperimentConfig` and offline cache tooling.

This module exists so we can expose stable console scripts via `[project.scripts]`
in `aria_nbv/pyproject.toml` (unlike `aria_nbv/scripts/*.py`, which are not
importable when the package is installed).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import AliasChoices, Field, field_validator
from pydantic._internal._utils import deep_update
from pydantic_settings import SettingsConfigDict

from aria_nbv.configs import PathConfig
from aria_nbv.data_handling._legacy_cache_api import (
    OracleRriCacheConfig,
    OracleRriCacheDataset,
    OracleRriCacheDatasetConfig,
    OracleRriCacheWriter,
    OracleRriCacheWriterConfig,
    VinSnippetCacheConfig,
    VinSnippetCacheWriterConfig,
)
from aria_nbv.data_handling._legacy_vin_source import VinOracleCacheDatasetConfig
from aria_nbv.lightning.aria_nbv_experiment import AriaNBVExperimentConfig
from aria_nbv.rri_metrics import RriOrdinalBinner
from aria_nbv.utils import BaseConfig, Console, Verbosity
from aria_nbv.utils.wandb_utils import (
    WANDB_STEP_KEYS,
    _ensure_wandb_api,
    _get_run,
    _load_runs_filtered,
    build_dynamics_dataframe,
    build_run_dataframes,
    collect_run_media_images,
    load_run_histories,
)


def _ensure_run_mode(argv: list[str], run_mode: str) -> list[str]:
    for arg in argv:
        if arg.startswith("--run-mode") or arg.startswith("--run_mode"):
            return argv
    return ["--run-mode", run_mode, *argv]


class CLIAriaNBVExperimentConfig(AriaNBVExperimentConfig):
    """CLI-enabled experiment config with optional TOML config path."""

    config_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("config-path", "config_path"),
    )
    """Path to a TOML configuration file."""

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_avoid_json=True,
        env_prefix="ARIA_NBV_",
    )


class CLICacheWriterConfig(OracleRriCacheWriterConfig):
    """CLI-enabled cache writer config with optional TOML config path."""

    config_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("config-path", "config_path"),
    )
    """Path to a TOML configuration file."""

    max_samples: int | None = Field(
        default=None,
        validation_alias=AliasChoices("max-samples", "max_samples", "num-samples", "num_samples", "n"),
    )
    """Optional cap on number of cached samples."""

    fit_binner: bool = False
    """Fit an RRI ordinal binner from the cached samples after writing."""

    binner_num_classes: int = 15
    """Number of ordinal classes for the fitted binner."""

    binner_path: Path | None = None
    """Optional output path for `rri_binner.json` (defaults to cache dir)."""

    binner_max_samples: int | None = None
    """Optional cap on number of cached samples used for binner fitting."""

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_avoid_json=True,
        env_prefix="ARIA_NBV_CACHE_",
    )


class CLIVinSnippetCacheBuildConfig(BaseConfig):
    """CLI config for building VIN snippet caches from an experiment TOML."""

    config_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("config-path", "config_path"),
    )
    """Path to a TOML configuration file (expected to be an AriaNBVExperimentConfig)."""

    split: Literal["all", "train", "val"] = Field(
        default="all",
        validation_alias=AliasChoices("split", "cache-split", "cache_split"),
    )
    """Which oracle-cache split to scan for snippet IDs."""

    max_samples: int | None = Field(
        default=None,
        validation_alias=AliasChoices("max-samples", "max_samples", "num-samples", "num_samples", "n"),
    )
    """Optional cap on number of snippets to process."""

    semidense_max_points: int | None = Field(
        default=None,
        validation_alias=AliasChoices("semidense-max-points", "semidense_max_points"),
    )
    """Optional cap on the number of collapsed semidense points."""

    map_location: torch.device = Field(
        default="cpu",
        validation_alias=AliasChoices("map-location", "map_location"),
    )
    """Device string for loading EFM snippet tensors (defaults to CPU)."""

    cache_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("cache-dir", "cache_dir", "out-cache-dir", "out_cache_dir"),
    )
    """Optional override for the output VIN snippet cache directory."""

    overwrite: bool = False
    """Allow overwriting an existing VIN snippet cache index."""

    resume: bool = True
    """Reuse existing cache entries when the index already exists."""

    num_workers: int | None = Field(
        default=None,
        validation_alias=AliasChoices("num-workers", "num_workers"),
    )
    """Optional DataLoader worker count for building VIN snippets."""

    persistent_workers: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("persistent-workers", "persistent_workers"),
    )
    """Whether to keep DataLoader workers alive between batches."""

    prefetch_factor: int | None = Field(
        default=None,
        validation_alias=AliasChoices("prefetch-factor", "prefetch_factor"),
    )
    """Optional DataLoader prefetch factor when num_workers > 0."""

    use_dataloader: bool = Field(
        default=False,
        validation_alias=AliasChoices("use-dataloader", "use_dataloader"),
    )
    """Force DataLoader usage even when num_workers=0."""

    skip_missing_snippets: bool = Field(
        default=True,
        validation_alias=AliasChoices("skip-missing-snippets", "skip_missing_snippets"),
    )
    """Skip missing snippets instead of treating them as hard failures."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Verbosity level for logging progress."""

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_avoid_json=True,
        env_prefix="ARIA_NBV_VIN_SNIPPET_CACHE_",
    )

    _validate_map_location = field_validator("map_location", mode="before")(BaseConfig._resolve_device)


class CLIWandbAnalysisConfig(BaseConfig):
    """CLI config for W&B run exports and local figure discovery."""

    entity: str | None = Field(
        default=None,
        validation_alias=AliasChoices("entity", "wandb-entity", "wandb_entity"),
    )
    """W&B entity (user/team). Defaults to WANDB_ENTITY env var."""

    project: str = Field(
        default="aria-nbv",
        validation_alias=AliasChoices("project", "wandb-project", "wandb_project"),
    )
    """W&B project name (default: aria-nbv)."""

    run_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("run-ids", "run_ids", "runs", "ids"),
    )
    """Optional run ids to select from the filtered run list."""

    run_paths: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("run-paths", "run_paths"),
    )
    """Optional full W&B run paths (entity/project/runs/<id>)."""

    max_runs: int = Field(
        default=50,
        validation_alias=AliasChoices("max-runs", "max_runs"),
    )
    """Max runs fetched before filtering by ids/tags/etc."""

    name_regex: str | None = Field(
        default=None,
        validation_alias=AliasChoices("name-regex", "name_regex"),
    )
    """Optional regex to match run names or ids."""

    states: list[str] = Field(
        default_factory=lambda: ["finished"],
        validation_alias=AliasChoices("states", "state"),
    )
    """Allowed run states (default: finished)."""

    tags: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("tags", "tag"),
    )
    """Optional run tags to match (any-match)."""

    group: str | None = Field(
        default=None,
        validation_alias=AliasChoices("group", "wandb-group", "wandb_group"),
    )
    """Optional group filter."""

    job_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("job-type", "job_type"),
    )
    """Optional job_type filter."""

    min_steps: float | None = Field(
        default=None,
        validation_alias=AliasChoices("min-steps", "min_steps"),
    )
    """Optional minimum step count for filtering."""

    max_steps: float | None = Field(
        default=None,
        validation_alias=AliasChoices("max-steps", "max_steps"),
    )
    """Optional maximum step count for filtering."""

    history_rows: int = Field(
        default=20000,
        validation_alias=AliasChoices("history-rows", "history_rows"),
    )
    """Max history rows to fetch per run."""

    history_keys: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("history-keys", "history_keys"),
    )
    """Optional history keys to fetch (empty = all)."""

    base_metrics: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("base-metrics", "base_metrics"),
    )
    """Optional base metrics for dynamics summaries."""

    segment_frac: float = Field(
        default=0.2,
        ge=0.05,
        le=0.5,
        validation_alias=AliasChoices("segment-frac", "segment_frac"),
    )
    """Fraction used to define early/late windows for dynamics statistics."""

    output_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("output-dir", "output_dir"),
    )
    """Directory for exported CSV files (defaults to .logs/wandb/analysis)."""

    export_meta: bool = True
    """Export run metadata CSV."""

    export_summary: bool = True
    """Export run summary CSV."""

    export_config: bool = True
    """Export flattened run config CSV."""

    export_histories: bool = True
    """Export per-run history CSVs."""

    export_dynamics: bool = True
    """Export run dynamics summary CSV."""

    export_figures_manifest: bool = True
    """Export manifest of local train/val figure images."""

    include_latest_figures: bool = True
    """Include .logs/wandb/wandb/latest-run in local figure search."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Verbosity level for logging progress."""

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_avoid_json=True,
        env_prefix="ARIA_NBV_WANDB_",
    )

    @field_validator("run_ids", "run_paths", "tags", "states", "history_keys", "base_metrics", mode="before")
    @classmethod
    def _split_csv_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value)]


# Resolve forward refs now that OracleRriCacheWriter is imported.
CLICacheWriterConfig.model_rebuild(_types_namespace={"OracleRriCacheWriter": OracleRriCacheWriter})


def _merge_with_toml(
    base_cfg: AriaNBVExperimentConfig | OracleRriCacheWriterConfig,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    return deep_update(base_cfg.model_dump(), overrides)


def _iter_cached_rri(
    cache_cfg: OracleRriCacheConfig,
    *,
    limit: int | None,
) -> tuple[OracleRriCacheDataset, Iterator[tuple[torch.Tensor, dict[str, str]]]]:
    # NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION: helper for the
    # legacy oracle-cache CLI flow.
    dataset_cfg = OracleRriCacheDatasetConfig(
        cache=cache_cfg,
        load_backbone=False,
        limit=limit,
    )
    dataset = OracleRriCacheDataset(dataset_cfg)

    def _iter() -> Iterator[tuple[torch.Tensor, dict[str, str]]]:
        for sample in dataset:
            rri = sample.rri.rri.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            meta = {"scene_id": sample.scene_id, "snippet_id": sample.snippet_id}
            yield rri, meta

    return dataset, _iter()


def _maybe_fit_binner(
    cache_cfg: OracleRriCacheConfig,
    *,
    num_classes: int,
    binner_path: Path | None,
    max_samples: int | None,
) -> None:
    console = Console.with_prefix("cache-cli", "fit_binner")
    dataset, rri_batches = _iter_cached_rri(
        cache_cfg,
        limit=max_samples,
    )
    first = next(rri_batches, None)
    if first is None:
        console.warn("No cached samples available for binner fitting.")
        return

    def _iter_with_first() -> Iterator[tuple[torch.Tensor, dict[str, str]]]:
        yield first
        yield from rri_batches

    rri_iter = _iter_with_first()
    console.log("Fitting RRI binner from cached samples.")
    binner = RriOrdinalBinner.fit_from_iterable(
        rri_iter,
        num_classes=int(num_classes),
    )
    out_path = binner_path or (dataset.config.cache.cache_dir / "rri_binner.json")
    saved = binner.save(out_path)
    console.log(f"Saved binner: {saved}")


def main(argv: list[str] | None = None) -> None:
    """Run training/eval/binner fitting for the configured experiment."""

    argv = list(sys.argv[1:] if argv is None else argv)
    paths = PathConfig()
    console = Console.with_prefix("cli", "main")

    cli_cfg = CLIAriaNBVExperimentConfig(_cli_parse_args=argv)
    config_path = cli_cfg.config_path
    if config_path is None:
        cfg = cli_cfg
    else:
        config_path = paths.resolve_config_toml_path(config_path, must_exist=True)
        base_cfg = AriaNBVExperimentConfig.from_toml(config_path)
        overrides = cli_cfg.model_dump(exclude_unset=True)
        overrides.pop("config_path", None)
        merged = _merge_with_toml(base_cfg, overrides)
        cfg = AriaNBVExperimentConfig.model_validate(merged)
    cfg.datamodule_config.source.inspect()
    cfg.module_config.inspect()
    try:
        cfg.run()
    except KeyboardInterrupt:
        console.warn("Interrupted by user (Ctrl+C).")
        raise SystemExit(130) from None


def fit_binner_main() -> None:
    """Convenience entry point that forces `--fit-binner-only`."""

    argv = list(sys.argv[1:])
    if "--fit-binner-only" not in argv and "--fit_binner_only" not in argv:
        argv = ["--fit-binner-only", *argv]
    main(argv)


def train_main() -> None:
    """Convenience entry point that forces `--run-mode train`."""

    argv = _ensure_run_mode(list(sys.argv[1:]), "train")
    main(argv)


def summarize_main() -> None:
    """Convenience entry point that forces `--run-mode summarize-vin`."""

    argv = _ensure_run_mode(list(sys.argv[1:]), "summarize-vin")
    main(argv)


def optuna_main() -> None:
    """Convenience entry point that forces `--run-mode optuna`."""

    argv = _ensure_run_mode(list(sys.argv[1:]), "optuna")
    main(argv)


def cache_main(argv: list[str] | None = None) -> None:
    """Build an offline oracle cache from the configured dataset."""
    # NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION: remove this CLI once
    # operators write only immutable vin_offline stores.

    argv = list(sys.argv[1:] if argv is None else argv)
    paths = PathConfig()
    console = Console.with_prefix("cache-cli", "main")

    cli_cfg = CLICacheWriterConfig(_cli_parse_args=argv)
    config_path = cli_cfg.config_path
    overwrite_set = "overwrite" in cli_cfg.model_fields_set
    fit_binner = cli_cfg.fit_binner
    binner_num_classes = cli_cfg.binner_num_classes
    binner_path = cli_cfg.binner_path
    binner_max_samples = cli_cfg.binner_max_samples

    if config_path is None:
        cfg = cli_cfg
    else:
        config_path = paths.resolve_config_toml_path(config_path, must_exist=True)
        base_cfg = OracleRriCacheWriterConfig.from_toml(config_path)
        overrides = cli_cfg.model_dump(exclude_unset=True)
        overrides.pop("config_path", None)
        overrides.pop("fit_binner", None)
        overrides.pop("binner_num_classes", None)
        overrides.pop("binner_path", None)
        overrides.pop("binner_max_samples", None)
        merged = _merge_with_toml(base_cfg, overrides)
        cfg = OracleRriCacheWriterConfig.model_validate(merged)

    if not overwrite_set and not cfg.overwrite and cfg.cache.index_path.exists():
        console.warn("Cache index exists; enabling overwrite=True for this run.")
        cfg.overwrite = True
    cfg.inspect()
    try:
        cfg.setup_target().run()
        if fit_binner:
            _maybe_fit_binner(
                cfg.cache,
                num_classes=binner_num_classes,
                binner_path=binner_path,
                max_samples=binner_max_samples,
            )
    except KeyboardInterrupt:
        console.warn("Interrupted by user (Ctrl+C).")
        raise SystemExit(130) from None


def cache_vin_snippets_main(argv: list[str] | None = None) -> None:
    """Build a VIN snippet cache from the configured offline oracle cache."""
    # NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION: remove this CLI once
    # the legacy VIN snippet cache is no longer needed.

    argv = list(sys.argv[1:] if argv is None else argv)
    paths = PathConfig()
    console = Console.with_prefix("vin-snippet-cache-cli", "main")

    cli_cfg = CLIVinSnippetCacheBuildConfig(_cli_parse_args=argv)
    config_path = cli_cfg.config_path
    if config_path is None:
        default_cfg = paths.configs_dir / "offline_only.toml"
        if default_cfg.exists():
            config_path = default_cfg
            console.log(
                "No --config-path provided; defaulting to .configs/offline_only.toml "
                "(override via --config-path <file>.toml).",
            )
        else:
            available = sorted(paths.configs_dir.glob("*.toml"))
            available_str = "\n".join(f"- {p.name}" for p in available) if available else "(none)"
            raise ValueError(
                "Provide --config-path pointing to an AriaNBVExperimentConfig TOML.\n\n"
                f"Available configs under {paths.configs_dir}:\n{available_str}",
            )
    config_path = paths.resolve_config_toml_path(config_path, must_exist=True)
    exp_cfg = AriaNBVExperimentConfig.from_toml(config_path)
    if not isinstance(exp_cfg.datamodule_config.source, VinOracleCacheDatasetConfig):
        raise ValueError("datamodule_config.source must be an offline cache to build a VIN snippet cache.")

    source_cfg = exp_cfg.datamodule_config.source
    source_cache = source_cfg.cache.cache

    out_cache = source_cfg.cache.vin_snippet_cache
    if cli_cfg.cache_dir is not None:
        out_cache = VinSnippetCacheConfig(cache_dir=cli_cfg.cache_dir, paths=exp_cfg.paths)
    elif out_cache is None:
        out_cache = VinSnippetCacheConfig(
            cache_dir=source_cache.cache_dir / "vin_snippet_cache",
            paths=exp_cfg.paths,
        )
        console.warn(
            f"datamodule_config.source.cache.vin_snippet_cache not set; using {out_cache.cache_dir}",
        )

    overwrite = bool(cli_cfg.overwrite)
    resume = bool(cli_cfg.resume)

    semidense_max_points = cli_cfg.semidense_max_points
    if semidense_max_points is None:
        semidense_max_points = source_cfg.cache.semidense_max_points

    datamodule_cfg = exp_cfg.datamodule_config
    num_workers = datamodule_cfg.num_workers
    if "num_workers" in cli_cfg.model_fields_set and cli_cfg.num_workers is not None:
        num_workers = cli_cfg.num_workers
    persistent_workers = datamodule_cfg.persistent_workers
    if "persistent_workers" in cli_cfg.model_fields_set and cli_cfg.persistent_workers is not None:
        persistent_workers = cli_cfg.persistent_workers
    prefetch_factor = None
    if "prefetch_factor" in cli_cfg.model_fields_set:
        prefetch_factor = cli_cfg.prefetch_factor
    use_dataloader = False
    if "use_dataloader" in cli_cfg.model_fields_set:
        use_dataloader = bool(cli_cfg.use_dataloader)
    skip_missing_snippets = True
    if "skip_missing_snippets" in cli_cfg.model_fields_set:
        skip_missing_snippets = bool(cli_cfg.skip_missing_snippets)

    writer_cfg = VinSnippetCacheWriterConfig(
        paths=exp_cfg.paths,
        cache=out_cache,
        source_cache=source_cache,
        split=cli_cfg.split,
        max_samples=cli_cfg.max_samples,
        semidense_max_points=semidense_max_points,
        include_obs_count=source_cfg.cache.semidense_include_obs_count,
        map_location=cli_cfg.map_location,
        overwrite=overwrite,
        resume=resume,
        use_dataloader=use_dataloader,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        skip_missing_snippets=skip_missing_snippets,
        verbosity=cli_cfg.verbosity,
    )
    writer_cfg.setup_target().run()


def wandb_main(argv: list[str] | None = None) -> None:
    """Export W&B summaries, histories, dynamics, and local figure manifests."""

    argv = list(sys.argv[1:] if argv is None else argv)
    console = Console.with_prefix("wandb-cli", "main")
    cfg = CLIWandbAnalysisConfig(_cli_parse_args=argv)
    console.set_verbosity(cfg.verbosity)

    entity = cfg.entity or os.environ.get("WANDB_ENTITY", "")
    project = cfg.project or os.environ.get("WANDB_PROJECT", "aria-nbv")
    api_key = os.environ.get("WANDB_API_KEY", "")

    api = _ensure_wandb_api(api_key or None)

    runs: list[Any] = []
    if cfg.run_paths:
        for run_path in cfg.run_paths:
            runs.append(_get_run(api, run_path))
    else:
        if not entity:
            raise ValueError("Provide --entity or set WANDB_ENTITY for W&B run retrieval.")
        runs = _load_runs_filtered(
            api=api,
            entity=entity,
            project=project,
            max_runs=cfg.max_runs,
            name_regex=cfg.name_regex,
            states=cfg.states,
            tags=set(cfg.tags),
            group=cfg.group or "",
            job_type=cfg.job_type or "",
            min_steps=cfg.min_steps,
            max_steps=cfg.max_steps,
        )
        if cfg.run_ids:
            run_ids = set(cfg.run_ids)
            runs = [
                run
                for run in runs
                if str(getattr(run, "id", "")) in run_ids or str(getattr(run, "name", "")) in run_ids
            ]

    if not runs:
        console.warn("No runs matched the provided filters.")
        return

    output_dir = cfg.output_dir
    if output_dir is None:
        output_dir = PathConfig().wandb / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"Writing exports to {output_dir}")

    meta_df, summary_df, config_df = build_run_dataframes(runs)
    if cfg.export_meta and not meta_df.empty:
        meta_path = output_dir / "wandb_runs_meta.csv"
        meta_df.to_csv(meta_path)
        console.log(f"Wrote metadata: {meta_path}")
    if cfg.export_summary and not summary_df.empty:
        summary_path = output_dir / "wandb_runs_summary.csv"
        summary_df.to_csv(summary_path)
        console.log(f"Wrote summary: {summary_path}")
    if cfg.export_config and not config_df.empty:
        config_path = output_dir / "wandb_runs_config.csv"
        config_df.to_csv(config_path)
        console.log(f"Wrote config: {config_path}")

    histories = load_run_histories(
        runs,
        keys=cfg.history_keys or None,
        max_rows=cfg.history_rows,
        replace_inf=True,
    )
    if cfg.export_histories:
        history_dir = output_dir / "histories"
        history_dir.mkdir(parents=True, exist_ok=True)
        for run in runs:
            run_id = str(getattr(run, "id", ""))
            history = histories.get(run_id)
            if history is None or history.empty:
                continue
            out_path = history_dir / f"{run_id}.csv"
            history.to_csv(out_path, index=False)
        console.log(f"Wrote histories to {history_dir}")

    if cfg.export_dynamics:
        dynamics_df = build_dynamics_dataframe(
            runs,
            histories,
            base_metrics=cfg.base_metrics or None,
            prefer_x_keys=list(WANDB_STEP_KEYS),
            segment_frac=cfg.segment_frac,
        )
        if not dynamics_df.empty:
            dynamics_path = output_dir / "wandb_runs_dynamics.csv"
            dynamics_df.to_csv(dynamics_path, index=False)
            console.log(f"Wrote dynamics: {dynamics_path}")

    if cfg.export_figures_manifest:
        rows: list[dict[str, str]] = []
        for run in runs:
            run_id = str(getattr(run, "id", ""))
            media = collect_run_media_images(
                run_id,
                include_latest=cfg.include_latest_figures,
            )
            for path in media.get("train_figures", []):
                rows.append({"run_id": run_id, "split": "train", "path": str(path)})
            for path in media.get("val_figures", []):
                rows.append({"run_id": run_id, "split": "val", "path": str(path)})
        if rows:
            import pandas as pd

            fig_path = output_dir / "wandb_figures_manifest.csv"
            pd.DataFrame(rows).to_csv(fig_path, index=False)
            console.log(f"Wrote figures manifest: {fig_path}")


if __name__ == "__main__":
    main()

__all__ = [
    "cache_main",
    "cache_vin_snippets_main",
    "fit_binner_main",
    "main",
    "optuna_main",
    "summarize_main",
    "train_main",
    "wandb_main",
]
