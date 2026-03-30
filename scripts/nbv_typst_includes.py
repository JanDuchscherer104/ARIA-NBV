#!/usr/bin/env python3
"""Repo-root wrapper for the aria-nbv-context Typst include helper."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET = ROOT_DIR / ".agents" / "skills" / "aria-nbv-context" / "scripts" / "nbv_typst_includes.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
