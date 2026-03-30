"""Compatibility tests for the legacy ``aria_nbv.utils.summary`` wrapper."""

from __future__ import annotations

import torch

from aria_nbv.utils.rich_summary import summarize as summarize_rich
from aria_nbv.utils.summary import summarize as summarize_legacy


def test_legacy_summary_wrapper_delegates_to_rich_summary() -> None:
    """Keep the legacy module path behavior-identical to the canonical owner."""

    tensor = torch.ones((2, 3), dtype=torch.float32)
    assert summarize_legacy(tensor, include_stats=True) == summarize_rich(  # noqa: S101
        tensor,
        include_stats=True,
    )
