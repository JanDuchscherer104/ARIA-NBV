"""Contract tests for the public ``aria_nbv.data_handling`` root API."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


def test_public_api_smoke_imports_all_exports() -> None:
    """Ensure every root-exported symbol resolves from the package root."""

    module = importlib.import_module("aria_nbv.data_handling")
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing  # noqa: S101


def test_public_api_omits_internal_helper_exports() -> None:
    """Keep low-level storage and migration plumbing off the root contract."""

    module = importlib.import_module("aria_nbv.data_handling")
    unexpected = {
        "OpenedShard",
        "VinOfflineStoreReader",
        "OracleRriCacheEntry",
        "OracleRriCacheMetadata",
        "OracleRriCacheSample",
        "VinSnippetCacheEntry",
        "VinSnippetCacheMetadata",
        "collapse_vin_points",
        "pad_vin_points",
        "vin_snippet_cache_config_hash",
        "LegacyOfflinePlan",
        "LegacyOfflineRecord",
        "prepare_legacy_records",
        "finalize_migrated_store",
    }
    assert not (unexpected & set(module.__all__))  # noqa: S101


def test_runtime_modules_do_not_import_data_handling_submodules() -> None:
    """Keep the runtime contract root-only outside ``data_handling`` itself."""

    package_root = Path(__file__).resolve().parents[2] / "aria_nbv"
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        if "data_handling" in path.parts:
            continue
        rel_path = path.relative_to(package_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name.startswith("aria_nbv.data_handling.") for alias in node.names):
                    offenders.append(rel_path.as_posix())
                    break
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if (
                    module.startswith("aria_nbv.data_handling.")
                    or ".data_handling." in module
                    or (node.level > 0 and module.startswith("data_handling."))
                ):
                    offenders.append(rel_path.as_posix())
                    break
    assert not offenders  # noqa: S101


def test_data_handling_has_no_legacy_data_imports() -> None:
    """Ensure the new core stays independent from ``aria_nbv.data``."""

    data_handling_root = Path(__file__).resolve().parents[2] / "aria_nbv" / "data_handling"
    offenders: list[str] = []
    for path in data_handling_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name.startswith("aria_nbv.data") for alias in node.names):
                    offenders.append(path.relative_to(data_handling_root).as_posix())
                    break
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if (
                    module.startswith("aria_nbv.data")
                    or module == "data"
                    or (node.level > 0 and module.startswith("data."))
                ):
                    offenders.append(path.relative_to(data_handling_root).as_posix())
                    break
    assert not offenders  # noqa: S101


def test_legacy_data_modules_alias_canonical_owners() -> None:
    """Ensure compatibility imports reuse the canonical module objects."""

    pairs = {
        "aria_nbv.data.efm_dataset": "aria_nbv.data_handling.efm_dataset",
        "aria_nbv.data.efm_snippet_loader": "aria_nbv.data_handling.efm_snippet_loader",
        "aria_nbv.data.efm_views": "aria_nbv.data_handling.efm_views",
        "aria_nbv.data.mesh_cache": "aria_nbv.data_handling.mesh_cache",
        "aria_nbv.data.offline_cache": "aria_nbv.data_handling.oracle_cache",
        "aria_nbv.data.offline_cache_store": "aria_nbv.data_handling.offline_cache_store",
        "aria_nbv.data.vin_oracle_datasets": "aria_nbv.data_handling.vin_oracle_datasets",
        "aria_nbv.data.vin_oracle_types": "aria_nbv.data_handling.vin_oracle_types",
        "aria_nbv.data.vin_snippet_cache": "aria_nbv.data_handling.vin_cache",
        "aria_nbv.data.vin_snippet_provider": "aria_nbv.data_handling.vin_provider",
        "aria_nbv.data.vin_snippet_utils": "aria_nbv.data_handling.vin_adapter",
    }

    for legacy_name, canonical_name in pairs.items():
        legacy_module = importlib.import_module(legacy_name)
        canonical_module = importlib.import_module(canonical_name)
        assert legacy_module is canonical_module  # noqa: S101
