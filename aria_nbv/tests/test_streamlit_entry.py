"""Tests for the Streamlit console-script wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

from aria_nbv.streamlit_app import _build_streamlit_argv, streamlit_entry


def test_streamlit_argv_disables_watchdog_by_default(monkeypatch) -> None:
    monkeypatch.delenv("STREAMLIT_SERVER_FILE_WATCHER_TYPE", raising=False)

    argv = _build_streamlit_argv(Path("/tmp/app.py"), [])

    assert argv == [
        "streamlit",
        "run",
        "--server.fileWatcherType",
        "none",
        "/tmp/app.py",
    ]


def test_streamlit_argv_preserves_file_watcher_cli_override(monkeypatch) -> None:
    monkeypatch.delenv("STREAMLIT_SERVER_FILE_WATCHER_TYPE", raising=False)

    argv = _build_streamlit_argv(
        Path("/tmp/app.py"),
        ["--server.fileWatcherType=poll", "--server.port", "8502", "--", "--debug"],
    )

    assert argv == [
        "streamlit",
        "run",
        "--server.fileWatcherType=poll",
        "--server.port",
        "8502",
        "/tmp/app.py",
        "--",
        "--debug",
    ]


def test_streamlit_argv_preserves_file_watcher_env_override(monkeypatch) -> None:
    monkeypatch.setenv("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")

    argv = _build_streamlit_argv(Path("/tmp/app.py"), [])

    assert argv == ["streamlit", "run", "/tmp/app.py"]


def test_streamlit_entry_rewrites_sys_argv(monkeypatch) -> None:
    captured_argv: list[str] = []

    def fake_streamlit_main() -> None:
        captured_argv[:] = sys.argv

    monkeypatch.delenv("STREAMLIT_SERVER_FILE_WATCHER_TYPE", raising=False)
    monkeypatch.setattr("streamlit.web.cli.main", fake_streamlit_main)
    monkeypatch.setattr(sys, "argv", ["nbv-st"])

    streamlit_entry()

    assert captured_argv[:4] == [
        "streamlit",
        "run",
        "--server.fileWatcherType",
        "none",
    ]
    assert captured_argv[4].endswith("streamlit_app.py")
