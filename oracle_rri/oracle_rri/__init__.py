"""Oracle RRI package scaffolding.

This module exposes high-level stubs that mirror the final architecture of the
oracle-RRI pipeline.  Each component currently wraps the upstream libraries
(EFM3D, ATEK, Project Aria) without committing to a concrete implementation so
that future iterations can swap in optimised code while keeping a stable API.
"""

from .config import DatasetPaths, OracleConfig
from .data.factory import DatasetFactory

__all__ = [
    "DatasetPaths",
    "OracleConfig",
    "DatasetFactory",
]
