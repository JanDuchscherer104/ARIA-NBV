"""CLI for building standalone target-RRI rollout stores."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..configs import PathConfig
from ..utils import Console
from ._rollout_dataset_writer import RolloutDatasetWriterConfig


def _resolve_config_path(path: Path) -> Path:
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
    parser = argparse.ArgumentParser(
        prog="nbv-build-rollouts",
        description="Build a standalone target-RRI rollout Zarr store from VIN offline rows.",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Path to a RolloutDatasetWriterConfig TOML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the TOML and print resolved paths without loading data or writing Zarr.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    console = Console.with_prefix("nbv-build-rollouts")
    config_path = _resolve_config_path(args.config_path)
    cfg = RolloutDatasetWriterConfig.from_toml(config_path)
    console.log(f"Loaded rollout writer config: {config_path}")
    console.log(f"Resolved source store: {cfg.source.store.store_dir}")
    console.log(f"Resolved rollout store: {cfg.store.store_dir}")
    console.log(f"Target top-k: {cfg.target_selector.k}")
    console.log(f"Candidate mixture budget: {cfg.candidate_mixture.total_count}")
    if args.dry_run:
        console.log("Dry run complete; no VIN offline dataset or rollout writer was instantiated.")
        return
    result = cfg.setup_target().run()
    console.log(
        "Wrote rollout Zarr store: "
        f"rollouts={result.num_rollouts} steps={result.num_steps} candidates={result.num_candidates} "
        f"path={result.store_dir}",
    )


__all__ = ["main"]
