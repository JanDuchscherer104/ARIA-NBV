"""Configuration management for oracle_rri.

This module provides centralized configuration classes for managing
paths, datasets, and other project-wide settings.
"""

from .path_config import PathConfig

try:  # Optional dependency (training-only).
    from .wandb_config import WandbConfig
except ModuleNotFoundError:  # pragma: no cover
    WandbConfig = None  # type: ignore[assignment]

__all__ = [
    "PathConfig",
    "WandbConfig",
]
