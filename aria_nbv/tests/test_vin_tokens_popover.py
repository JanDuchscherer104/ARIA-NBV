"""Tests for VIN diagnostics tokens popover text."""

# ruff: noqa: S101

from __future__ import annotations

from aria_nbv.app.panels.vin_diag_tabs import tokens


def test_semidense_projection_popover_text() -> None:
    text = tokens.SEMIDENSE_PROJ_FEATURES_INFO
    assert isinstance(text, str)
    assert "counts" in text.lower()
    assert "weights" in text.lower()
    assert "depth_mean" in text.lower()
    assert "depth_std" in text.lower()
