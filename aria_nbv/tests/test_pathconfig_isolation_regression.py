"""Regression tests for PathConfig singleton isolation across tests."""

from __future__ import annotations

from pathlib import Path

from aria_nbv.configs import PathConfig


def test_a_data_handling_style_mutation_does_not_crash(tmp_path: Path) -> None:
    """Simulate path mutation style used by data handling tests."""
    cfg = PathConfig(
        root=tmp_path,
        data_root=tmp_path / "data_root",
        url_dir=tmp_path / "urls",
        ase_meshes=tmp_path / "meshes",
    )
    assert cfg.root == tmp_path.resolve()
    assert cfg.data_root == (tmp_path / "data_root").resolve()


def test_b_pathconfig_is_restored_for_following_tests() -> None:
    """Verify later tests observe project defaults, not prior tmp overrides."""
    project_root = Path(__file__).resolve().parents[2]
    cfg = PathConfig()
    assert cfg.root == project_root
    assert cfg.data_root == (project_root / ".data").resolve()


def test_root_update_rebases_omitted_default_relative_paths(tmp_path: Path) -> None:
    """Changing only root should rebase root-relative default paths."""
    cfg = PathConfig()
    cfg = PathConfig(root=tmp_path)

    assert cfg.root == tmp_path.resolve()
    assert cfg.data_root == (tmp_path / ".data").resolve()
    assert cfg.processed_meshes == (tmp_path / ".data" / "ase_meshes_processed").resolve()
    assert cfg.checkpoints == (tmp_path / ".logs" / "checkpoints").resolve()


def test_root_update_preserves_explicit_custom_paths(tmp_path: Path) -> None:
    """Explicitly customized fields should not be overwritten by root rebasing."""
    custom_data_root = tmp_path / "custom_data"
    other_root = tmp_path / "other_root"
    other_root.mkdir()
    cfg = PathConfig(data_root=custom_data_root)
    cfg = PathConfig(root=other_root)

    assert cfg.data_root == custom_data_root.resolve()


def test_resolve_cache_dir_prefers_offline_cache_root(tmp_path: Path) -> None:
    """Relative cache paths should resolve under the configured offline cache root."""
    offline_cache_root = tmp_path / "offline_cache_root"
    cfg = PathConfig(root=tmp_path, offline_cache_dir=offline_cache_root)

    resolved = cfg.resolve_cache_dir("vin_snippets")

    assert resolved == (offline_cache_root / "vin_snippets").resolve()


def test_resolve_cache_dir_avoids_duplicating_cache_prefix(tmp_path: Path) -> None:
    """Cache paths already rooted at the cache dir name should stay project-root-relative."""
    offline_cache_root = tmp_path / "offline_cache"
    cfg = PathConfig(root=tmp_path, offline_cache_dir=offline_cache_root)

    resolved = cfg.resolve_cache_dir("offline_cache/vin_snippets")

    assert resolved == (tmp_path / "offline_cache" / "vin_snippets").resolve()
