"""Configuration management for oracle_rri.

This module provides centralized configuration classes for managing
paths, datasets, and other project-wide settings.
"""

from .path_config import PathConfig

__all__ = [
    "PathConfig",
]
