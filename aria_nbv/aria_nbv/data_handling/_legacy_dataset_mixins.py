"""Shared legacy dataset mixins kept outside the public cache surface."""

from __future__ import annotations


class _ResolvedLenDatasetMixin:
    """Provide the common ``__len__`` implementation for legacy cache readers."""

    _len: int

    def __len__(self) -> int:
        return self._len


__all__ = ["_ResolvedLenDatasetMixin"]
