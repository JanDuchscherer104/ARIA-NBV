"""Command-line entry point for inspecting VIN offline samples in Rerun."""

from __future__ import annotations

import argparse
import socket
import subprocess
from pathlib import Path

from aria_nbv.utils import Console
from aria_nbv.utils.config_paths import resolve_config_toml_path

from ._config import RerunOfflineInspectorConfig
from ._loggers import RerunModule, RerunOfflineLogger
from ._metadata import collect_visual_inventory, validate_required_inventory
from ._sample import select_rerun_sample


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
    parser.add_argument("--scene-id", help="Override selection.scene_id.")
    parser.add_argument("--snippet-id", help="Override selection.snippet_id.")
    parser.add_argument("--candidate-index", type=int, help="Override candidate.selected_index for detail layers.")
    parser.add_argument("--offline-store", type=Path, help="Override dataset.offline.store.store_dir.")
    parser.add_argument(
        "--rollout-store",
        type=Path,
        help="Inspect a standalone rollouts.zarr replay store instead of a VIN offline sample.",
    )
    parser.add_argument(
        "--rollout-index",
        type=int,
        default=0,
        help="Zero-based rollout row position to inspect when --rollout-store is set.",
    )
    parser.add_argument(
        "--rollout-row-id",
        type=int,
        help="Explicit rollouts/rollout_row_id to inspect when --rollout-store is set.",
    )
    parser.add_argument(
        "--rollout-context",
        choices=("auto", "required", "off"),
        help="VIN context policy for --rollout-store: auto, required, or off.",
    )
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
    parser.add_argument(
        "--view",
        action="store_true",
        help="Save the .rrd and open it in the native Rerun viewer in the foreground.",
    )
    parser.add_argument(
        "--serve-web",
        action="store_true",
        help="Save the .rrd and serve it with Rerun's web viewer in the foreground.",
    )
    parser.add_argument(
        "--web-viewer-port",
        type=int,
        default=0,
        help="Rerun web viewer HTTP port for --serve-web. Use 0 to let Rerun choose.",
    )
    parser.add_argument(
        "--ws-server-port",
        type=int,
        default=0,
        help="Rerun websocket port for --serve-web. Use 0 to let Rerun choose.",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="With --serve-web, bind to 0.0.0.0 and print a LAN URL hint.",
    )
    return parser


def _validate_viewer_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject incompatible CLI viewer and SDK-sink combinations."""

    if args.view and args.serve_web:
        parser.error("--view and --serve-web are mutually exclusive.")
    if (args.view or args.serve_web) and (args.spawn or args.connect is not None):
        parser.error("--view/--serve-web are post-save viewer modes and cannot be combined with --spawn/--connect.")
    if args.lan and not args.serve_web:
        parser.error("--lan requires --serve-web.")
    if args.rollout_index is not None and int(args.rollout_index) < 0:
        parser.error("--rollout-index must be >= 0.")
    if args.rollout_row_id is not None and int(args.rollout_row_id) < 0:
        parser.error("--rollout-row-id must be >= 0.")
    for name in ("web_viewer_port", "ws_server_port"):
        port = int(getattr(args, name))
        if port < 0 or port > 65535:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 65535].")


def _apply_overrides(config: RerunOfflineInspectorConfig, args: argparse.Namespace) -> RerunOfflineInspectorConfig:
    """Apply CLI overrides while preserving TOML defaults."""

    cfg = config.model_copy(deep=True)
    selection_updates: dict[str, object] = {}
    if args.split is not None:
        selection_updates["split"] = args.split
    if args.index is not None:
        selection_updates["index"] = args.index
    if args.sample_id is not None:
        selection_updates["sample_key"] = args.sample_id
    if args.scene_id is not None:
        selection_updates["scene_id"] = args.scene_id
    if args.snippet_id is not None:
        selection_updates["snippet_id"] = args.snippet_id
    if args.rollout_context is not None:
        selection_updates["rollout_context_mode"] = args.rollout_context
    if selection_updates:
        cfg.selection = type(cfg.selection).model_validate({**cfg.selection.model_dump(), **selection_updates})
    if args.offline_store is not None:
        store = cfg.dataset.offline.store.model_copy(update={"store_dir": args.offline_store.expanduser()})
        offline = cfg.dataset.offline.model_copy(update={"store": store})
        cfg.dataset = cfg.dataset.model_copy(update={"offline": offline})
    if args.candidate_index is not None:
        cfg.candidate.selected_index = args.candidate_index
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
    if args.view or args.serve_web:
        cfg.output.mode = "save"
    return RerunOfflineInspectorConfig.model_validate(cfg.model_dump())


def _detect_lan_ip() -> str:
    """Return the likely LAN-facing IPv4 address for user-facing hints."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def _viewer_command(
    *,
    save_path: Path,
    serve_web: bool,
    lan: bool,
    web_viewer_port: int,
    ws_server_port: int,
) -> list[str]:
    """Build the foreground Rerun viewer command."""

    if not serve_web:
        return ["rerun", str(save_path)]
    return [
        "rerun",
        "--bind",
        "0.0.0.0" if lan else "127.0.0.1",
        "--serve-web",
        "--web-viewer-port",
        str(web_viewer_port),
        "--ws-server-port",
        str(ws_server_port),
        str(save_path),
    ]


def _print_lan_hint(*, web_viewer_port: int) -> None:
    """Print a LAN access hint for opt-in web serving."""

    ip = _detect_lan_ip()
    if web_viewer_port > 0:
        print(f"ARIA-NBV Rerun LAN URL: http://{ip}:{web_viewer_port}/", flush=True)
    else:
        print(
            "ARIA-NBV Rerun LAN mode enabled. Rerun will choose the web-viewer port; "
            f"use host {ip} from another device.",
            flush=True,
        )


def _run_viewer_command(command: list[str], *, lan: bool, web_viewer_port: int) -> None:
    """Run Rerun in the foreground while preserving its stdout/stderr."""

    if lan:
        _print_lan_hint(web_viewer_port=web_viewer_port)
    subprocess.run(command, check=True)


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

    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_viewer_args(parser, args)
    config_path = resolve_config_toml_path(args.config_path)
    cfg = RerunOfflineInspectorConfig.from_toml(config_path)
    cfg = _apply_overrides(cfg, args)
    if args.rollout_store is not None:
        from ._rollout_zarr import run_rollout_zarr_inspector

        run_rollout_zarr_inspector(
            cfg,
            store_dir=args.rollout_store,
            rollout_index=int(args.rollout_index),
            rollout_row_id=args.rollout_row_id,
        )
    else:
        run_inspector(cfg)
    if args.view or args.serve_web:
        _run_viewer_command(
            _viewer_command(
                save_path=cfg.output.save_path,
                serve_web=bool(args.serve_web),
                lan=bool(args.lan),
                web_viewer_port=int(args.web_viewer_port),
                ws_server_port=int(args.ws_server_port),
            ),
            lan=bool(args.lan),
            web_viewer_port=int(args.web_viewer_port),
        )


__all__ = ["RerunOfflineInspector", "main", "run_inspector"]
