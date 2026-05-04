"""Command-line entry point for inspecting VIN offline samples in Rerun."""

from __future__ import annotations

import argparse
from pathlib import Path

from aria_nbv.configs import PathConfig
from aria_nbv.utils import Console

from ._config import RerunOfflineInspectorConfig
from ._loggers import RerunModule, RerunOfflineLogger
from ._metadata import collect_visual_inventory, validate_required_inventory
from ._sample import select_rerun_sample


def _resolve_config_path(path: Path) -> Path:
    """Resolve an inspector TOML path from shell-relative paths or config names."""

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
    """Build the command-line parser for ``nbv-rerun-inspect``."""

    parser = argparse.ArgumentParser(
        prog="nbv-rerun-inspect",
        description="Inspect one immutable VIN offline sample in Rerun.",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Path to a RerunOfflineInspectorConfig TOML file.",
    )
    parser.add_argument("--split", choices=("all", "train", "val"), help="Override selection.split.")
    parser.add_argument("--index", type=int, help="Override selection.index.")
    parser.add_argument("--sample-id", help="Override selection.sample_key.")
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        metavar="PATH",
        help="Override output mode to save. Optionally provide an .rrd path.",
    )
    parser.add_argument("--spawn", action="store_true", help="Override output mode to spawn a local viewer.")
    parser.add_argument(
        "--connect",
        nargs="?",
        const="",
        metavar="ADDR",
        help="Override output mode to connect over Rerun gRPC. Optionally provide an address.",
    )
    return parser


def _apply_overrides(config: RerunOfflineInspectorConfig, args: argparse.Namespace) -> RerunOfflineInspectorConfig:
    """Apply CLI overrides while preserving TOML defaults."""

    cfg = config.model_copy(deep=True)
    if args.split is not None:
        cfg.selection.split = args.split
    if args.index is not None:
        cfg.selection.index = args.index
    if args.sample_id is not None:
        cfg.selection.sample_key = args.sample_id
    if args.save is not None:
        cfg.output.mode = "save"
        if args.save:
            cfg.output.save_path = Path(args.save).expanduser()
    if args.spawn:
        cfg.output.mode = "spawn"
    if args.connect is not None:
        cfg.output.mode = "connect"
        if args.connect:
            cfg.output.connect_addr = args.connect
    return RerunOfflineInspectorConfig.model_validate(cfg.model_dump())


class RerunOfflineInspector:
    """Runtime object created from ``RerunOfflineInspectorConfig``."""

    def __init__(self, config: RerunOfflineInspectorConfig, *, rr_module: RerunModule | None = None) -> None:
        """Initialize the inspector runtime."""

        self.config = config
        self.rr_module = rr_module
        self.console = Console.with_prefix("nbv-rerun-inspect").set_verbosity(config.performance.verbosity)

    def run(self) -> None:
        """Collect inventory, select one sample, then log it to Rerun."""

        selected = select_rerun_sample(
            dataset_config=self.config.dataset.offline,
            selection=self.config.selection,
        )
        inventory = collect_visual_inventory(selected.sample)
        validate_required_inventory(self.config, inventory)
        logger = RerunOfflineLogger(self.config, rr_module=self.rr_module)
        logger.start()
        logger.log_sample(sample=selected.sample, inventory=inventory, selection=selected.description)
        logger.log_metadata(sample=selected.sample, inventory=inventory, selection=selected.description)
        self.console.log(
            f"Logged Rerun offline sample: {selected.sample.sample_key} ({selected.description})",
        )


def run_inspector(config: RerunOfflineInspectorConfig, *, rr_module: RerunModule | None = None) -> None:
    """Run the inspector for tests and CLI callers."""

    RerunOfflineInspector(config, rr_module=rr_module).run()


def main(argv: list[str] | None = None) -> None:
    """Run ``nbv-rerun-inspect`` from a TOML config and CLI overrides."""

    args = _build_parser().parse_args(argv)
    config_path = _resolve_config_path(args.config_path)
    cfg = RerunOfflineInspectorConfig.from_toml(config_path)
    run_inspector(_apply_overrides(cfg, args))


__all__ = ["RerunOfflineInspector", "main", "run_inspector"]
