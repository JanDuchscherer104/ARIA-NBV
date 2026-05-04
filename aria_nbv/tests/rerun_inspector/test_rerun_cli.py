"""CLI tests for the offline Rerun inspector."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aria_nbv.rerun_inspector import _cli
from aria_nbv.rerun_inspector._config import RerunOfflineInspectorConfig
from aria_nbv.rerun_inspector._metadata import OfflineVisualInventory


def test_cli_applies_selection_and_save_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI flags should override TOML without launching Rerun."""

    config_path = tmp_path / "rerun.toml"
    save_path = tmp_path / "sample.rrd"
    RerunOfflineInspectorConfig().save_toml(config_path)
    captured: dict[str, RerunOfflineInspectorConfig] = {}

    def _capture(config: RerunOfflineInspectorConfig, *, rr_module: object | None = None) -> None:
        del rr_module
        captured["config"] = config

    monkeypatch.setattr(_cli, "run_inspector", _capture)

    _cli.main(
        [
            "--config-path",
            str(config_path),
            "--split",
            "train",
            "--index",
            "7",
            "--sample-id",
            "scene/snippet/sample",
            "--save",
            str(save_path),
        ],
    )

    cfg = captured["config"]
    assert cfg.selection.split == "train"  # noqa: S101
    assert cfg.selection.index == 7  # noqa: S101
    assert cfg.selection.sample_key == "scene/snippet/sample"  # noqa: S101
    assert cfg.output.mode == "save"  # noqa: S101
    assert cfg.output.save_path == save_path  # noqa: S101


@pytest.mark.parametrize(
    ("args", "mode", "addr"),
    [
        (["--spawn"], "spawn", None),
        (["--connect", "rerun+http://127.0.0.1:9876/proxy"], "connect", "rerun+http://127.0.0.1:9876/proxy"),
    ],
)
def test_cli_applies_output_mode_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    mode: str,
    addr: str | None,
) -> None:
    """Spawn/connect flags should override the configured output sink."""

    config_path = tmp_path / "rerun.toml"
    RerunOfflineInspectorConfig().save_toml(config_path)
    captured: dict[str, RerunOfflineInspectorConfig] = {}
    monkeypatch.setattr(
        _cli,
        "run_inspector",
        lambda config, *, rr_module=None: captured.setdefault("config", config),
    )

    _cli.main(["--config-path", str(config_path), *args])

    cfg = captured["config"]
    assert cfg.output.mode == mode  # noqa: S101
    assert cfg.output.connect_addr == addr  # noqa: S101


def test_missing_required_inventory_fails_before_rerun_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Required inventory failures should happen before sample load or Rerun init."""

    cfg = RerunOfflineInspectorConfig()
    calls: list[str] = []

    monkeypatch.setattr(_cli, "collect_visual_inventory", lambda sample: OfflineVisualInventory(has_semidense=False))
    monkeypatch.setattr(
        _cli,
        "select_rerun_sample",
        lambda **kwargs: SimpleNamespace(
            sample=SimpleNamespace(sample_key="sample-0"), description="sample_key=sample-0"
        ),
    )
    monkeypatch.setattr(
        _cli,
        "RerunOfflineLogger",
        lambda *args, **kwargs: calls.append("logger"),
    )

    with pytest.raises(RuntimeError, match="semidense"):
        _cli.run_inspector(cfg, rr_module=object())

    assert calls == []  # noqa: S101
