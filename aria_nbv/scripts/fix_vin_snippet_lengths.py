"""Fix points_length in VIN snippet cache samples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch


def _resolve_cache_dir(raw: str) -> Path:
    base = Path(raw).expanduser()
    if base.is_dir() and (base / "samples").exists():
        return base
    candidate = base / "vin_snippet_cache"
    if candidate.is_dir() and (candidate / "samples").exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find vin_snippet_cache samples under: {base}",
    )


def _iter_samples(samples_dir: Path) -> Iterable[Path]:
    return sorted(samples_dir.glob("*.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix points_length fields in VIN snippet cache samples.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to vin_snippet_cache (or its parent).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on processed files.",
    )
    args = parser.parse_args()

    cache_dir = _resolve_cache_dir(args.cache_dir)
    samples_dir = cache_dir / "samples"
    paths = list(_iter_samples(samples_dir))
    if args.max_files is not None:
        paths = paths[: int(args.max_files)]

    total = len(paths)
    updated = 0
    skipped = 0
    missing = 0

    for idx, path in enumerate(paths, start=1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        points_world = payload.get("points_world")
        if not torch.is_tensor(points_world):
            missing += 1
            continue

        finite = torch.isfinite(points_world[:, :3]).all(dim=-1)
        finite_count = int(finite.sum().item())

        stored = payload.get("points_length")
        stored_count = None
        if stored is not None:
            stored_tensor = torch.as_tensor(stored)
            if stored_tensor.numel() > 0:
                stored_count = int(stored_tensor.reshape(-1)[0].item())
            else:
                stored_count = 0

        if stored_count == finite_count:
            skipped += 1
        else:
            updated += 1
            if not args.dry_run:
                payload["points_length"] = torch.tensor([finite_count], dtype=torch.int64)
                torch.save(payload, path)

        if idx % 500 == 0 or idx == total:
            print(f"Processed {idx}/{total} (updated={updated}, skipped={skipped}, missing={missing})")

    print(
        f"Done. total={total} updated={updated} skipped={skipped} missing_points_world={missing}",
    )


if __name__ == "__main__":
    main()
