"""Shared validation helpers for data-handling configs."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationInfo

from ..configs import PathConfig


def resolve_cache_artifact_dir(value: str | Path, info: ValidationInfo) -> Path:
    """Resolve cache/store directories against the project cache roots."""
    paths: PathConfig = info.data.get("paths") or PathConfig()
    return paths.resolve_cache_artifact_dir(value)


__all__ = ["resolve_cache_artifact_dir"]
