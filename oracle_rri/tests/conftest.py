"""Shared pytest fixtures for oracle_rri test suite."""

from __future__ import annotations

import pytest

from oracle_rri.configs import PathConfig


@pytest.fixture(autouse=True)
def _restore_path_config_singleton() -> None:
    """Snapshot/restore global PathConfig singleton state for each test."""
    snapshot = PathConfig().model_dump()
    try:
        yield
    finally:
        PathConfig(**snapshot)
