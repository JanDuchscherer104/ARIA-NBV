"""Legacy compatibility wrapper for the canonical raw EFM dataset module.

The active implementation now lives in :mod:`aria_nbv.data_handling.efm_dataset`.
Keep this module as a thin historical import surface while remaining callers
are migrated.
"""

from __future__ import annotations

from ..data_handling.efm_dataset import *  # noqa: F401,F403
from ..data_handling.efm_dataset import __all__ as canonical_all

__all__ = list(canonical_all)
