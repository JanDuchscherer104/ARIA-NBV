"""CLI wrapper for VIN summaries using the Lightning data pipeline.

This script forwards to the Lightning CLI entry point with
``--run-mode summarize-vin`` so the summary always uses real oracle batches
from :class:`VinDataModule`.
"""

from __future__ import annotations

import sys

from aria_nbv.lightning.cli import main as cli_main


def main() -> None:
    argv = sys.argv[1:]
    if "--run-mode" not in argv and "--run_mode" not in argv and "--run-mode=summarize-vin" not in argv:
        argv = ["--run-mode", "summarize-vin", *argv]
    sys.argv = [sys.argv[0], *argv]
    cli_main()


if __name__ == "__main__":  # pragma: no cover
    main()
