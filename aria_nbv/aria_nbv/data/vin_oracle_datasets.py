"""Legacy compatibility wrapper for canonical VIN dataset-source configs.

The active implementation now lives in
:mod:`aria_nbv.data_handling.vin_oracle_datasets`.
"""

from __future__ import annotations

from ..data_handling.vin_oracle_datasets import *  # noqa: F401,F403
from ..data_handling.vin_oracle_datasets import __all__ as canonical_all

__all__ = list(canonical_all)
