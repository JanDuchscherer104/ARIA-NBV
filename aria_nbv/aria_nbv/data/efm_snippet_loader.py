"""Legacy compatibility wrapper for the canonical EFM snippet loader.

The active implementation now lives in
:mod:`aria_nbv.data_handling.efm_snippet_loader`.
"""

from __future__ import annotations

from ..data_handling.efm_snippet_loader import *  # noqa: F401,F403
from ..data_handling.efm_snippet_loader import __all__ as canonical_all

__all__ = list(canonical_all)
