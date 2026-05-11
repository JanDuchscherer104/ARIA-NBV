"""Tests for Streamlit Rerun launch command helpers."""

# ruff: noqa: S101

from pathlib import Path

from aria_nbv.app.rerun_launch import (
    build_rerun_offline_spawn_command,
    build_rerun_rollout_spawn_command,
    format_command,
)


def test_rollout_spawn_command_uses_argument_list() -> None:
    command = build_rerun_rollout_spawn_command(
        config_path=Path("cfg.toml"),
        rollout_store=Path("rollouts.zarr"),
        rollout_row_id=42,
    )

    assert command == [
        "nbv-rerun-inspect",
        "--config-path",
        "cfg.toml",
        "--rollout-store",
        "rollouts.zarr",
        "--rollout-row-id",
        "42",
        "--spawn",
    ]


def test_offline_spawn_command_uses_store_override_and_sample_selection() -> None:
    command = build_rerun_offline_spawn_command(
        config_path=Path("cfg.toml"),
        offline_store=Path("vin_offline"),
        split="val",
        index=3,
    )

    assert command == [
        "nbv-rerun-inspect",
        "--config-path",
        "cfg.toml",
        "--offline-store",
        "vin_offline",
        "--split",
        "val",
        "--index",
        "3",
        "--spawn",
    ]


def test_command_format_is_display_only_shell_escaped() -> None:
    assert format_command(["cmd", "path with spaces"]) == "cmd 'path with spaces'"
