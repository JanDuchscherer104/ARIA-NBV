"""Small subprocess helpers for launching Rerun inspectors from Streamlit."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def build_rerun_rollout_spawn_command(
    *,
    config_path: Path | str,
    rollout_store: Path | str,
    rollout_row_id: int,
) -> list[str]:
    """Build the inspector command for a single rollout-store row."""

    return [
        "nbv-rerun-inspect",
        "--config-path",
        str(Path(config_path).expanduser()),
        "--rollout-store",
        str(Path(rollout_store).expanduser()),
        "--rollout-row-id",
        str(int(rollout_row_id)),
        "--spawn",
    ]


def build_rerun_offline_spawn_command(
    *,
    config_path: Path | str,
    offline_store: Path | str,
    split: str,
    index: int,
) -> list[str]:
    """Build the inspector command for a VIN offline sample."""

    return [
        "nbv-rerun-inspect",
        "--config-path",
        str(Path(config_path).expanduser()),
        "--offline-store",
        str(Path(offline_store).expanduser()),
        "--split",
        str(split),
        "--index",
        str(int(index)),
        "--spawn",
    ]


def spawn_background_command(command: list[str]) -> subprocess.Popen[bytes]:
    """Spawn a detached subprocess from an argument list."""

    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def format_command(command: list[str]) -> str:
    """Return a shell-readable command preview for display only."""

    return " ".join(shlex.quote(part) for part in command)


def repo_root() -> Path:
    """Return the ARIA-NBV repository root from the package path."""

    return Path(__file__).resolve().parents[3]


__all__ = [
    "build_rerun_offline_spawn_command",
    "build_rerun_rollout_spawn_command",
    "format_command",
    "repo_root",
    "spawn_background_command",
]
