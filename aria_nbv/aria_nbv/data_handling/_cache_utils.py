"""Shared helper utilities for `data_handling` cache and offline-store code.

This module owns small internal helpers that are reused across the active
cache, migration, and immutable offline-store surfaces. The goal is to keep
those modules focused on their domain logic instead of duplicating filename,
hashing, timestamp, and tensor-conversion helpers.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from efm3d.aria.pose import PoseTW


def utc_now_iso() -> str:
    """Return the current UTC timestamp in stable ISO-8601 form."""

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def json_signature(payload: dict[str, Any]) -> str:
    """Return a stable SHA-1 hash for a JSON-serializable payload."""

    serial = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serial.encode("utf-8")).hexdigest()


def sanitize_token(value: str) -> str:
    """Normalize a token so it is safe to embed in filenames or sample keys."""

    return re.sub(r"[^0-9a-zA-Z._-]+", "_", value).strip("_")


def extract_snippet_token(snippet_id: str) -> str:
    """Extract the stable numeric suffix from a snippet identifier when present."""

    match = re.search(r"(?:AtekDataSample|DataSample)_([0-9]+)$", snippet_id)
    return match.group(1) if match else snippet_id


def format_sample_key(scene_id: str, snippet_id: str, config_hash: str) -> str:
    """Build a stable oracle/VIN cache sample key."""

    scene_safe = sanitize_token(scene_id)
    snippet_token = extract_snippet_token(snippet_id)
    snippet_safe = sanitize_token(snippet_token)
    key = f"ASE_NBV_SNIPPET_{scene_safe}_{snippet_safe}_{config_hash}"
    if len(key) <= 200:
        return key
    snippet_digest = hashlib.sha1(snippet_id.encode("utf-8")).hexdigest()[:12]
    return f"ASE_NBV_SNIPPET_{scene_safe}_{snippet_digest}_{config_hash}"


def default_sample_key(scene_id: str, snippet_id: str) -> str:
    """Build the default immutable-store sample key for one snippet."""

    return f"{sanitize_token(scene_id)}::{sanitize_token(snippet_id)}"


def unique_sample_path(samples_dir: Path, base_key: str) -> tuple[str, Path]:
    """Return a collision-free sample path for a requested base key."""

    path = samples_dir / f"{base_key}.pt"
    if not path.exists():
        return base_key, path
    while True:
        suffix = hashlib.sha1(f"{base_key}-{time.time_ns()}".encode()).hexdigest()[:6]
        key = f"{base_key}__{suffix}"
        path = samples_dir / f"{key}.pt"
        if not path.exists():
            return key, path


def to_numpy(
    value: torch.Tensor | np.ndarray | bool | int | float,
    *,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    """Convert a scalar or tensor-like value into a NumPy array."""

    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def pose_to_numpy(pose: PoseTW) -> np.ndarray:
    """Convert a ``PoseTW`` into a CPU float32 NumPy array."""

    array = to_numpy(pose.tensor(), dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    return array


def pad_first_axis(
    array: np.ndarray,
    *,
    target_len: int,
    fill_value: float | int | bool,
) -> np.ndarray:
    """Pad or truncate an array along its first axis."""

    if array.ndim == 0:
        return array
    current = int(array.shape[0])
    if current == target_len:
        return array
    if current > target_len:
        return array[:target_len]
    pad_shape = (target_len - current, *array.shape[1:])
    pad = np.full(pad_shape, fill_value, dtype=array.dtype)
    return np.concatenate([array, pad], axis=0)


__all__ = [
    "default_sample_key",
    "extract_snippet_token",
    "format_sample_key",
    "json_signature",
    "pad_first_axis",
    "pose_to_numpy",
    "sanitize_token",
    "to_numpy",
    "unique_sample_path",
    "utc_now_iso",
]
