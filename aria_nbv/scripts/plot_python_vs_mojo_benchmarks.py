"""Render polished Python vs Mojo benchmark plots from CSV results."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _ensure_aria_nbv_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main() -> None:
    _ensure_aria_nbv_importable()

    from aria_nbv.utils.benchmark_plotting import load_benchmark_csv, write_benchmark_report

    parser = argparse.ArgumentParser(description="Plot Python vs Mojo benchmark comparisons from CSV.")
    parser.add_argument("--input", type=str, required=True, help="CSV file with raw benchmark trials.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="docs/figures/benchmarks/python_vs_mojo",
        help="Directory for HTML plots and summary CSV.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="Python vs Mojo",
        help="Prefix applied to all plot titles.",
    )
    parser.add_argument(
        "--write-png",
        action="store_true",
        help="Also export PNGs. Requires a working kaleido installation.",
    )
    args = parser.parse_args()

    if args.write_png and importlib.util.find_spec("kaleido") is None:
        raise RuntimeError("--write-png requested, but kaleido is not installed in the active environment.")

    records = load_benchmark_csv(args.input)
    written = write_benchmark_report(
        records,
        out_dir=args.out_dir,
        title_prefix=args.title_prefix,
        write_png=bool(args.write_png),
    )

    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
