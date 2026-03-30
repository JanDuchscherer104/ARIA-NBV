"""Shared helpers for stable sample/cache identifiers."""

from __future__ import annotations

import re


def sanitize_token(value: str) -> str:
    """Normalize a token so it is safe to embed in sample identifiers."""
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", value).strip("_")


__all__ = ["sanitize_token"]
