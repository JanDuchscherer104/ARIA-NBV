"""Attribution utility tests."""

# ruff: noqa: S101, SLF001

import torch
from oracle_rri.interpretability.attribution import (
    AttributionEngine,
    AttributionMethod,
    InterpretabilityConfig,
)


def test_to_heatmap_vector() -> None:
    """Ensure vector attributions are normalised to [0, 1]."""
    config = InterpretabilityConfig(
        method=AttributionMethod.INPUT_X_GRADIENT,
        use_abs=True,
    )
    engine = AttributionEngine(config=config, model=torch.nn.Identity())
    raw_attr = torch.tensor([[1.0, -2.0, 3.0]])
    reference = torch.zeros_like(raw_attr)

    heatmap = engine._to_heatmap(raw_attr=raw_attr, reference=reference)

    assert heatmap.shape == raw_attr.shape
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
