"""Subprocess helpers for Streamlit stage execution."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


def run_stage_subprocess(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run one app stage in a fresh Python process."""

    with tempfile.TemporaryDirectory(prefix="nbv-stage-") as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "input.pt"
        output_path = tmp_path / "output.pt"
        torch.save(payload, input_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aria_nbv.app.stage_subprocess_worker",
                "--mode",
                mode,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            details = stderr or stdout or f"exit code {result.returncode}"
            raise RuntimeError(f"Stage subprocess `{mode}` failed: {details}")
        return torch.load(output_path, map_location="cpu", weights_only=False)


__all__ = ["run_stage_subprocess"]
