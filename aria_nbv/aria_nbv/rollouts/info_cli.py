"""Inspect rollout-store manifests without loading replay payload arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..utils import Console
from .zarr_store import RolloutZarrStoreReader


def _build_parser() -> argparse.ArgumentParser:
    """Build the rollout-store info CLI parser."""

    parser = argparse.ArgumentParser(
        prog="nbv-rollouts-info",
        description="Inspect top-level rollout Zarr metadata and optional validation status.",
    )
    parser.add_argument(
        "--store",
        required=True,
        type=Path,
        help="Path to a manifest-backed rollouts.zarr store.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run full Zarr table validation after reading the top-level manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run rollout-store metadata inspection."""

    args = _build_parser().parse_args(argv)
    reader = RolloutZarrStoreReader(args.store)
    payload: dict[str, Any] = reader.manifest()
    if args.validate:
        validation = reader.validate()
        payload["validation"] = {
            "ok": validation.ok,
            "num_rollouts": validation.num_rollouts,
            "num_steps": validation.num_steps,
            "num_candidates": validation.num_candidates,
            "errors": validation.errors,
        }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    _print_text_summary(payload, validate=args.validate)


def _print_text_summary(payload: dict[str, Any], *, validate: bool) -> None:
    """Print a compact human-readable manifest summary."""

    console = Console.with_prefix("nbv-rollouts-info")
    manifest = payload["manifest"]
    root_attrs = payload["root_attrs"]
    counts = manifest.get("counts", {})
    coverage = manifest.get("source_coverage", {})
    console.log(
        "store: "
        f"schema={root_attrs.get('schema_version')} rollouts={counts.get('rollouts')} "
        f"steps={counts.get('steps')} candidates={counts.get('candidates')}"
    )
    console.log(
        "source coverage: "
        f"sources={coverage.get('num_source_rows')} scenes={coverage.get('scene_counts')} "
        f"splits={coverage.get('split_counts')}"
    )
    invocation = manifest.get("generation", {}).get("invocation", {})
    console.log(
        "invocation: "
        f"mode={invocation.get('mode')} config={invocation.get('config_path')} "
        f"toml_sha256={invocation.get('raw_toml_sha256')}"
    )
    if validate:
        validation = payload.get("validation", {})
        console.log(f"validation: ok={validation.get('ok')} errors={validation.get('errors', [])}")


__all__ = ["main"]
