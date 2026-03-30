"""Shared numeric helpers for pose-generation geometry and plotting.

This module holds small tensor utilities that are reused across the
pose-generation surface so orientation logic and plotting helpers do not carry
their own duplicate low-level math helpers.
"""

from __future__ import annotations

import torch


def normalize_last_dim(vectors: torch.Tensor, *, eps: float | None = None) -> torch.Tensor:
    """Normalize vectors along the last dimension.

    Args:
        vectors: Tensor whose last dimension stores vector components.
        eps: Optional lower bound for the norm. When omitted, uses the machine
            epsilon of the tensor dtype.

    Returns:
        Tensor with the same shape as ``vectors`` and unit-norm vectors along
        the last dimension whenever numerically possible.
    """

    min_norm = torch.finfo(vectors.dtype).eps if eps is None else float(eps)
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(min_norm)


__all__ = ["normalize_last_dim"]
