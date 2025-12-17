"""CLI entry points for `AriaNBVExperimentConfig`.

This module exists so we can expose stable console scripts via `[project.scripts]`
in `oracle_rri/pyproject.toml` (unlike `oracle_rri/scripts/*.py`, which are not
importable when the package is installed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..configs import PathConfig
from .aria_nbv_experiment import AriaNBVExperimentConfig


def _extract_config_path(argv: list[str]) -> Path | None:
    for idx, arg in enumerate(argv):
        if arg in ("--config_path", "--config-path") and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        if arg.startswith("--config_path=") or arg.startswith("--config-path="):
            return Path(arg.split("=", 1)[1])
    return None


def _normalize_cli_args(argv: list[str]) -> list[str]:
    """Accept both snake_case and kebab-case flags."""

    out: list[str] = []
    for arg in argv:
        if arg.startswith("--") and "_" in arg:
            out.append(arg.replace("_", "-"))
        else:
            out.append(arg)
    return out


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


class CLIAriaNBVExperimentConfig(BaseSettings, AriaNBVExperimentConfig):
    """CLI-enabled experiment config with optional TOML config path."""

    config_path: Path | None = Field(default=None)
    """Path to a TOML configuration file."""

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        cli_parse_args=True,
        cli_kebab_case=True,
        cli_implicit_flags=True,
        env_prefix="ARIA_NBV_",
    )


def main() -> None:
    """Run training/eval/binner fitting for the configured experiment."""

    argv = _normalize_cli_args(sys.argv[1:])
    config_path = _extract_config_path(argv)
    paths = PathConfig()

    if config_path is None:
        cfg = CLIAriaNBVExperimentConfig(_cli_parse_args=argv)
        cfg.inspect()
        cfg.run()
        return

    config_path = paths.resolve_config_toml_path(config_path, must_exist=True)

    base_cfg = AriaNBVExperimentConfig.from_toml(config_path)
    cli_cfg = CLIAriaNBVExperimentConfig(_cli_parse_args=argv)
    overrides = cli_cfg.model_dump(exclude_unset=True)
    overrides.pop("config_path", None)

    merged = _deep_update(base_cfg.model_dump(), overrides)
    cfg = AriaNBVExperimentConfig.model_validate(merged)
    cfg.run()


def fit_binner_main() -> None:
    """Convenience entry point that forces `--fit-binner-only`."""

    argv = sys.argv[1:]
    if "--fit-binner-only" not in argv and "--fit_binner_only" not in argv:
        argv = ["--fit-binner-only", *argv]
    sys.argv = [sys.argv[0], *_normalize_cli_args(argv)]
    main()


__all__ = ["fit_binner_main", "main"]
