"""Thin wrapper around :mod:`oracle_rri.rri_metrics.coral` for VIN imports."""

from __future__ import annotations

from ..rri_metrics.coral import (  # noqa: F401
    CoralLayer,
    coral_expected_from_logits,
    coral_logits_to_prob,
    coral_loss,
    coral_random_loss,
)

__all__ = [
    "CoralLayer",
    "coral_expected_from_logits",
    "coral_logits_to_prob",
    "coral_loss",
    "coral_random_loss",
]
