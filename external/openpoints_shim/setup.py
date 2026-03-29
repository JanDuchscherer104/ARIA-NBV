# ruff: noqa: INP001
"""Setuptools entrypoint for the OpenPoints shim build."""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from openpoints_shim.build_helpers import ensure_openpoints_built  # noqa: E402


class BuildPyWithOpenPoints(_build_py):
    """Build hook that compiles OpenPoints ops before packaging."""

    def run(self) -> None:
        """Run the OpenPoints build before the standard build."""
        ensure_openpoints_built()
        super().run()


setup(
    cmdclass={"build_py": BuildPyWithOpenPoints},
)
