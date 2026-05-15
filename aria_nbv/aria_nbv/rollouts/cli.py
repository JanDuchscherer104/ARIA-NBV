"""CLI for building standalone target-RRI rollout stores."""

from __future__ import annotations

# TODO: Use typer for nicer CLI!
import argparse
import sys
from pathlib import Path

from ..utils import Console
from ..utils.config_paths import resolve_config_toml_path
from .dataset_writer import RolloutDatasetWriterConfig
from .manifest import RolloutStoreInvocation


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
    raw_argv = sys.argv[1:] if argv is None else argv
    args = _build_parser().parse_args(raw_argv)
    console = Console.with_prefix("nbv-build-rollouts")
    config_path = resolve_config_toml_path(args.config_path)
    cfg = RolloutDatasetWriterConfig.from_toml(config_path)
    console.log(f"Loaded rollout writer config: {config_path}")
    console.log(f"Resolved source store: {cfg.source.store.store_dir}")
    console.log(f"Resolved rollout store: {cfg.store.store_dir}")
    console.log(f"Target top-k: {cfg.target_selector.k}")
    console.log(f"Candidate mixture budget: {cfg.candidate_mixture.total_count}")
    if args.dry_run:
        console.log("Dry run complete; no VIN offline dataset or rollout writer was instantiated.")
        return
    result = cfg.setup_target().run(
        invocation=RolloutStoreInvocation.from_cli(argv=["nbv-build-rollouts", *raw_argv], config_path=config_path)
    )
    console.log(
        "Wrote rollout Zarr store: "
        f"rollouts={result.num_rollouts} steps={result.num_steps} candidates={result.num_candidates} "
        f"path={result.store_dir} manifest={result.manifest_path}",
    )


__all__ = ["main"]
