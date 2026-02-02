"""CLI wrapper for VIN encoding plots using the Lightning data pipeline."""

from __future__ import annotations

import sys

from oracle_rri.lightning.cli import main as cli_main


def main() -> None:
    argv = sys.argv[1:]
    if "--run-mode" not in argv and "--run_mode" not in argv and "--run-mode=plot-vin-encodings" not in argv:
        argv = ["--run-mode", "plot-vin-encodings", *argv]
    sys.argv = [sys.argv[0], *argv]
    cli_main()


if __name__ == "__main__":  # pragma: no cover
    main()
