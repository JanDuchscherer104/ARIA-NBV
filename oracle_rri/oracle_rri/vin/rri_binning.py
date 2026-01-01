"""Thin wrapper around :mod:`oracle_rri.rri_metrics.rri_binning` for VIN imports."""

from __future__ import annotations

from ..rri_metrics.rri_binning import (  # noqa: F401
    RriOrdinalBinner,
    ordinal_labels_to_levels,
)

__all__ = ["RriOrdinalBinner", "ordinal_labels_to_levels"]
