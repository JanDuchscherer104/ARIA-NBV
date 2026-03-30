"""Shared semidense projection feature contracts for VIN models."""

from __future__ import annotations

SEMIDENSE_PROJ_FEATURES: tuple[str, ...] = (
    "coverage",
    "empty_frac",
    "semidense_candidate_vis_frac",
    "depth_mean",
    "depth_std",
)
"""Ordered per-candidate projection statistics used by VIN."""

SEMIDENSE_PROJ_DIM = len(SEMIDENSE_PROJ_FEATURES)
"""Feature dimension for semidense projection summaries."""

SEMIDENSE_PROJ_FEATURE_ALIASES: dict[str, str] = {
    "valid_frac": "semidense_candidate_vis_frac",
    "semidense_valid_frac": "semidense_candidate_vis_frac",
}
"""Backwards-compatible aliases for projection feature lookups."""

SEMIDENSE_GRID_FEATURES: tuple[str, ...] = (
    "occupancy",
    "depth_mean",
    "depth_std",
)
"""Channels used by the VIN v3 semidense projection grid CNN."""

SEMIDENSE_GRID_CHANNELS = len(SEMIDENSE_GRID_FEATURES)
"""Channel count for semidense projection grids."""


def semidense_proj_feature_index(name: str) -> int:
    """Resolve one semidense projection feature name or alias to its index."""
    if name in SEMIDENSE_PROJ_FEATURES:
        return SEMIDENSE_PROJ_FEATURES.index(name)
    alias = SEMIDENSE_PROJ_FEATURE_ALIASES.get(name)
    if alias is not None and alias in SEMIDENSE_PROJ_FEATURES:
        return SEMIDENSE_PROJ_FEATURES.index(alias)
    raise ValueError(f"Unknown semidense projection feature '{name}'.")


__all__ = [
    "SEMIDENSE_GRID_CHANNELS",
    "SEMIDENSE_GRID_FEATURES",
    "SEMIDENSE_PROJ_DIM",
    "SEMIDENSE_PROJ_FEATURES",
    "semidense_proj_feature_index",
]
