"""Regression tests for experimental VIN plotting helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from aria_nbv.vin.experimental.plotting import plot_vin_encodings_from_debug


def test_plot_vin_encodings_from_debug_writes_html_files(tmp_path) -> None:
    """Persist VIN encoding figures and return their output paths."""

    debug = SimpleNamespace(
        candidate_center_dir_rig=torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32),
        candidate_forward_dir_rig=torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32),
        candidate_radius_m=torch.tensor([[[1.0]]], dtype=torch.float32),
        pose_enc=torch.zeros((1, 1, 4), dtype=torch.float32),
    )

    outputs = plot_vin_encodings_from_debug(
        debug,
        out_dir=tmp_path,
        lmax=1,
        sh_normalization="integral",
        radius_freqs=[1.0],
        file_stem_prefix="debug",
        include_legacy_sh=False,
    )

    assert set(outputs) == {"shell_descriptor"}
    output_path = outputs["shell_descriptor"]
    assert output_path.exists()
    assert output_path.name == "debug_shell_descriptor.html"
    assert "<html" in output_path.read_text(encoding="utf-8").lower()
