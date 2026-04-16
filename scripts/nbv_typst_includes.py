#!/usr/bin/env python3
"""Repo-level wrapper for the Aria-NBV Typst include/outline helper."""

from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "aria-nbv-context"
    / "scripts"
    / "nbv_typst_includes.py"
)

runpy.run_path(str(SCRIPT), run_name="__main__")
