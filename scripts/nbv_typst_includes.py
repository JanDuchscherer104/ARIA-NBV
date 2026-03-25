#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    skill_script = root_dir / ".agents" / "skills" / "aria-nbv-context" / "scripts" / "nbv_typst_includes.py"
    runpy.run_path(skill_script.as_posix(), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
