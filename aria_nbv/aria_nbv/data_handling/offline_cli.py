"""CLI for building immutable VIN offline stores.

This module exposes the ``nbv-build-offline`` console script. It loads a
``VinOfflineWriterConfig`` TOML file, validates it through the normal
config-as-factory path, and runs :class:`aria_nbv.data_handling.VinOfflineWriter`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..configs import PathConfig
from ..utils import Console
from ._offline_writer import VinOfflineWriterConfig


def _resolve_config_path(path: Path) -> Path:
    """Resolve a writer TOML path from shell-relative paths or config names."""

    expanded = path.expanduser()
    if expanded.is_absolute():
        resolved = expanded.resolve()
    elif expanded.exists():
        resolved = expanded.resolve()
    else:
        resolved = PathConfig().resolve_config_toml_path(expanded, must_exist=True)
    if resolved.suffix != ".toml":
        raise ValueError(f"Config path must be a .toml file, got {resolved}.")
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for immutable offline-store creation."""

    parser = argparse.ArgumentParser(
        prog="nbv-build-offline",
        description="Build an immutable VIN offline store from raw ASE/EFM snippets and oracle RRI labels.",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Path to a VinOfflineWriterConfig TOML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the TOML and print the resolved store path without loading data or writing shards.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run immutable VIN offline-store creation from a TOML config.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv[1:]``.
    """

    args = _build_parser().parse_args(argv)
    console = Console.with_prefix("nbv-build-offline")
    config_path = _resolve_config_path(args.config_path)
    cfg = VinOfflineWriterConfig.from_toml(config_path)
    console.log(f"Loaded writer config: {config_path}")
    console.log(f"Resolved store dir: {cfg.store.store_dir}")
    if args.dry_run:
        console.log("Dry run complete; no dataset, backbone, or writer was instantiated.")
        return
    manifest = cfg.setup_target().run()
    console.log(
        "Wrote VIN offline store: "
        f"samples={manifest.stats.get('num_samples', 0)} "
        f"shards={manifest.stats.get('num_shards', 0)} "
        f"train={manifest.stats.get('num_train', 0)} "
        f"val={manifest.stats.get('num_val', 0)}",
    )


__all__ = ["main"]
