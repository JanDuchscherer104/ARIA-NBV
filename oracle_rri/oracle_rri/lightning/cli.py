"""CLI entry points for `AriaNBVExperimentConfig` and offline cache tooling.

This module exists so we can expose stable console scripts via `[project.scripts]`
in `oracle_rri/pyproject.toml` (unlike `oracle_rri/scripts/*.py`, which are not
importable when the package is installed).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import AliasChoices, Field, field_validator
from pydantic._internal._utils import deep_update
from pydantic_settings import SettingsConfigDict

from oracle_rri.configs import PathConfig
from oracle_rri.data.offline_cache import (
    OracleRriCacheConfig,
    OracleRriCacheDataset,
    OracleRriCacheDatasetConfig,
    OracleRriCacheWriter,
    OracleRriCacheWriterConfig,
)
from oracle_rri.data.vin_snippet_cache import VinSnippetCacheConfig, VinSnippetCacheWriterConfig
from oracle_rri.lightning.aria_nbv_experiment import AriaNBVExperimentConfig
from oracle_rri.rri_metrics import RriOrdinalBinner
from oracle_rri.utils import BaseConfig, Console, Verbosity


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

    @field_validator("map_location", mode="before")
    @classmethod
    def _validate_map_location(cls, value: str | torch.device) -> torch.device:
        return cls._resolve_device(value)


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
    from ..data.vin_oracle_datasets import VinOracleCacheDatasetConfig

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
]
