"""Helpers to run Mojo-backed stages in a separate Python process."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


def run_mojo_subprocess(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one Mojo stage in a fresh Python process.

    This is used when Python-imported Mojo extensions are not stable in the
    current thread context, such as Streamlit's ScriptRunner worker thread.
    """

    with tempfile.TemporaryDirectory(prefix="nbv-mojo-") as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / "input.pt"
        output_path = tmp_path / "output.pt"
        torch.save(payload, input_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aria_nbv.mojo_subprocess_worker",
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
            raise RuntimeError(f"Mojo subprocess `{mode}` failed: {details}")
        return torch.load(output_path, map_location="cpu", weights_only=False)


__all__ = ["run_mojo_subprocess"]
