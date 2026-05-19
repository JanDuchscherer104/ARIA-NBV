"""CLI tests for the offline Rerun inspector."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from aria_nbv.rerun_inspector import _cli, _rollout_zarr
from aria_nbv.rerun_inspector._config import RerunOfflineInspectorConfig
from aria_nbv.rerun_inspector._metadata import OfflineVisualInventory

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_cli_applies_selection_and_save_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI flags should override TOML without launching Rerun."""

    config_path = tmp_path / "rerun.toml"
    save_path = tmp_path / "sample.rrd"
    offline_store = tmp_path / "vin_offline"
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
            "--scene-id",
            "scene-a",
            "--snippet-id",
            "snippet-b",
            "--candidate-index",
            "2",
            "--offline-store",
            str(offline_store),
            "--save",
            str(save_path),
        ],
    )

    cfg = captured["config"]
    assert cfg.selection.split == "train"  # noqa: S101
    assert cfg.selection.index == 7  # noqa: S101
    assert cfg.selection.sample_key == "scene/snippet/sample"  # noqa: S101
    assert cfg.selection.scene_id == "scene-a"  # noqa: S101
    assert cfg.selection.snippet_id == "snippet-b"  # noqa: S101
    assert cfg.candidate.selected_index == 2  # noqa: S101
    assert cfg.dataset.offline.store.store_dir == offline_store  # noqa: S101
    assert cfg.output.mode == "save"  # noqa: S101
    assert cfg.output.save_path == save_path  # noqa: S101


@pytest.mark.parametrize(
    ("args", "mode", "addr"),
    [
        (["--spawn"], "spawn", None),
        (["--connect"], "connect", None),
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


def test_cli_save_without_path_preserves_configured_save_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare --save should switch to save mode without requiring a path value."""

    config_path = tmp_path / "rerun.toml"
    base_cfg = RerunOfflineInspectorConfig()
    base_cfg.output.save_path = tmp_path / "configured.rrd"
    base_cfg.save_toml(config_path)
    captured: dict[str, RerunOfflineInspectorConfig] = {}
    monkeypatch.setattr(
        _cli,
        "run_inspector",
        lambda config, *, rr_module=None: captured.setdefault("config", config),
    )

    _cli.main(["--config-path", str(config_path), "--save"])

    cfg = captured["config"]
    assert cfg.output.mode == "save"  # noqa: S101
    assert cfg.output.save_path == tmp_path / "configured.rrd"  # noqa: S101


def test_cli_routes_rollout_store_to_multistep_inspector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--rollout-store should bypass VIN sample selection and inspect rollouts.zarr."""

    config_path = tmp_path / "rerun.toml"
    store_path = tmp_path / "rollouts.zarr"
    RerunOfflineInspectorConfig().save_toml(config_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(_cli, "run_inspector", lambda *args, **kwargs: pytest.fail("VIN inspector should not run"))

    def _capture_rollout(config, *, store_dir, rollout_index=0, rollout_row_id=None, rr_module=None):
        del rr_module
        captured["config"] = config
        captured["store_dir"] = store_dir
        captured["rollout_index"] = rollout_index
        captured["rollout_row_id"] = rollout_row_id

    monkeypatch.setattr(_rollout_zarr, "run_rollout_zarr_inspector", _capture_rollout)

    _cli.main(
        [
            "--config-path",
            str(config_path),
            "--rollout-store",
            str(store_path),
            "--rollout-index",
            "2",
            "--rollout-row-id",
            "7",
            "--rollout-context",
            "required",
        ],
    )

    assert isinstance(captured["config"], RerunOfflineInspectorConfig)  # noqa: S101
    assert captured["store_dir"] == store_path  # noqa: S101
    assert captured["rollout_index"] == 2  # noqa: S101
    assert captured["rollout_row_id"] == 7  # noqa: S101
    assert captured["config"].selection.rollout_context_mode == "required"  # noqa: S101


def test_cli_view_opens_saved_recording_in_native_viewer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--view should save first, then foreground rerun <rrd>."""

    config_path = tmp_path / "rerun.toml"
    save_path = tmp_path / "sample.rrd"
    RerunOfflineInspectorConfig().save_toml(config_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(_cli, "run_inspector", lambda config, *, rr_module=None: captured.setdefault("config", config))
    monkeypatch.setattr(
        _cli,
        "_run_viewer_command",
        lambda command, *, lan, web_viewer_port: captured.setdefault(
            "viewer",
            (command, lan, web_viewer_port),
        ),
    )

    _cli.main(["--config-path", str(config_path), "--save", str(save_path), "--view"])

    cfg = captured["config"]
    assert isinstance(cfg, RerunOfflineInspectorConfig)  # noqa: S101
    assert cfg.output.mode == "save"  # noqa: S101
    assert captured["viewer"] == (["rerun", str(save_path)], False, 0)  # noqa: S101


def test_cli_serve_web_uses_auto_ports_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--serve-web should pass Rerun's port-0 auto-selection defaults."""

    config_path = tmp_path / "rerun.toml"
    save_path = tmp_path / "sample.rrd"
    RerunOfflineInspectorConfig().save_toml(config_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(_cli, "run_inspector", lambda config, *, rr_module=None: captured.setdefault("config", config))
    monkeypatch.setattr(
        _cli,
        "_run_viewer_command",
        lambda command, *, lan, web_viewer_port: captured.setdefault(
            "viewer",
            (command, lan, web_viewer_port),
        ),
    )

    _cli.main(["--config-path", str(config_path), "--save", str(save_path), "--serve-web"])

    assert captured["viewer"] == (  # noqa: S101
        [
            "rerun",
            "--bind",
            "127.0.0.1",
            "--serve-web",
            "--web-viewer-port",
            "0",
            "--port",
            "0",
            str(save_path),
        ],
        False,
        0,
    )


def test_cli_serve_web_lan_and_explicit_ports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--lan should opt into 0.0.0.0 bind while preserving explicit ports."""

    config_path = tmp_path / "rerun.toml"
    save_path = tmp_path / "sample.rrd"
    RerunOfflineInspectorConfig().save_toml(config_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(_cli, "run_inspector", lambda config, *, rr_module=None: captured.setdefault("config", config))
    monkeypatch.setattr(
        _cli,
        "_run_viewer_command",
        lambda command, *, lan, web_viewer_port: captured.setdefault(
            "viewer",
            (command, lan, web_viewer_port),
        ),
    )

    _cli.main(
        [
            "--config-path",
            str(config_path),
            "--save",
            str(save_path),
            "--serve-web",
            "--lan",
            "--web-viewer-port",
            "9090",
            "--ws-server-port",
            "9877",
        ],
    )

    assert captured["viewer"] == (  # noqa: S101
        [
            "rerun",
            "--bind",
            "0.0.0.0",
            "--serve-web",
            "--web-viewer-port",
            "9090",
            "--port",
            "9877",
            str(save_path),
        ],
        True,
        9090,
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--view", "--serve-web"],
        ["--view", "--spawn"],
        ["--serve-web", "--connect"],
        ["--lan"],
        ["--serve-web", "--web-viewer-port", "-1"],
        ["--serve-web", "--ws-server-port", "65536"],
    ],
)
def test_cli_rejects_invalid_viewer_flag_combinations(tmp_path: Path, args: list[str]) -> None:
    """Viewer flag conflicts should fail before generation or launch."""

    config_path = tmp_path / "rerun.toml"
    RerunOfflineInspectorConfig().save_toml(config_path)

    with pytest.raises(SystemExit) as exc_info:
        _cli.main(["--config-path", str(config_path), *args])

    assert exc_info.value.code == 2  # noqa: S101


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


def test_rerun_cli_help_exits_cleanly() -> None:
    result = runner.invoke(_cli.app, ["--help"])

    assert result.exit_code == 0
    assert "--config-path" in result.output
