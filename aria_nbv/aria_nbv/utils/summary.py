"""Legacy compatibility wrapper for succinct tensor summaries.

The canonical implementation now lives in :mod:`aria_nbv.utils.rich_summary`.
This module remains only as the historical import surface for callers that
still import :mod:`aria_nbv.utils.summary` directly.
"""

from __future__ import annotations

from .rich_summary import summarize as summarize

__all__ = ["summarize"]
