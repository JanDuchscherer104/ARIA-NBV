"""Entry point for VIN training/evaluation via PyTorch Lightning.

This script mirrors the CLI pattern used in `external/doc_classifier/run.py`:

- CLI parsing via `pydantic-settings` (no argparse usage in this script).
- Optional TOML config loading via `--config-path`.
- When a TOML config is provided, CLI args override specific values.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from oracle_rri.lightning import AriaNBVExperimentConfig

warnings.filterwarnings(
    "ignore",
    message=r".*multi-threaded, use of fork\(\) may lead to deadlocks.*",
    category=DeprecationWarning,
)


def _extract_config_path(argv: list[str]) -> Path | None:
    for idx, arg in enumerate(argv):
        if arg in ("--config_path", "--config-path") and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        if arg.startswith("--config_path=") or arg.startswith("--config-path="):
            return Path(arg.split("=", 1)[1])
    return None


def _normalize_cli_args(argv: list[str]) -> list[str]:
    """Accept both snake_case and kebab-case flags.

    The CLI is configured with `cli_kebab_case=True`, but older commands/docs
    may still use underscore flags (e.g. `--fit_binner_only`). We translate
    `--foo_bar` → `--foo-bar`.
    """

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
    argv = _normalize_cli_args(sys.argv[1:])
    config_path = _extract_config_path(argv)

    if config_path is None:
        cfg = CLIAriaNBVExperimentConfig(_cli_parse_args=argv)
        cfg.run()
        return

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_cfg = AriaNBVExperimentConfig.from_toml(config_path)
    cli_cfg = CLIAriaNBVExperimentConfig(_cli_parse_args=argv)
    overrides = cli_cfg.model_dump(exclude_unset=True)
    overrides.pop("config_path", None)

    merged = _deep_update(base_cfg.model_dump(), overrides)
    cfg = AriaNBVExperimentConfig.model_validate(merged)
    cfg.inspect()
    cfg.run()


if __name__ == "__main__":  # pragma: no cover
    main()
