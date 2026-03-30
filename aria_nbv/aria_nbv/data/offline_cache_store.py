"""Legacy compatibility wrapper for canonical oracle-cache metadata helpers.

The active implementation now lives in
:mod:`aria_nbv.data_handling.offline_cache_store`.
"""

from __future__ import annotations

from pathlib import Path

from ..data_handling._cache_utils import (
    extract_snippet_token as _extract_snippet_token,
)
from ..data_handling._cache_utils import (
    format_sample_key as _format_sample_key,
)
from ..data_handling._cache_utils import sanitize_token as _sanitize_token
from ..data_handling._cache_utils import (
    unique_sample_path as _canonical_unique_sample_path,
)
from ..data_handling.offline_cache_store import (
    build_cache_metadata,
    compute_config_hash,
    config_signature,
    ensure_config_hash,
    read_metadata,
    snapshot_config,
    snapshot_dataset_config,
    write_metadata,
)


def _unique_sample_path_compat(samples_dir: Path, base_key: str) -> tuple[str, Path]:
    """Compatibility alias for the canonical collision-free sample-path helper."""

    return _canonical_unique_sample_path(samples_dir, base_key)


extract_snippet_token = _extract_snippet_token
format_sample_key = _format_sample_key
sanitize_token = _sanitize_token
unique_sample_path = _unique_sample_path_compat

_compute_config_hash = compute_config_hash
_config_signature = config_signature
_ensure_config_hash = ensure_config_hash
_read_metadata = read_metadata
_write_metadata = write_metadata

__all__ = [
    "build_cache_metadata",
    "compute_config_hash",
    "config_signature",
    "ensure_config_hash",
    "extract_snippet_token",
    "format_sample_key",
    "read_metadata",
    "sanitize_token",
    "snapshot_config",
    "snapshot_dataset_config",
    "unique_sample_path",
    "write_metadata",
]

_unique_sample_path = _unique_sample_path_compat
