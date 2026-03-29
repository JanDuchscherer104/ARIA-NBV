"""Shared pytest fixtures for the aria_nbv test suite."""

from __future__ import annotations

import pytest

from aria_nbv.configs import PathConfig


@pytest.fixture(autouse=True)
def _restore_path_config_singleton() -> None:
    """Snapshot/restore global PathConfig singleton state for each test."""
    snapshot = PathConfig().model_dump()
    try:
        yield
    finally:
        PathConfig(**snapshot)
