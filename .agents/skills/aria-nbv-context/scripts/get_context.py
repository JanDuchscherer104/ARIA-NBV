#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    target = repo_root / "aria_nbv" / "scripts" / "get_context.py"
    runpy.run_path(target.as_posix(), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
