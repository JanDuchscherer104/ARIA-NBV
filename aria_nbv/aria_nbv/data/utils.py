"""Legacy import shims for data utilities.

The implementations live in :mod:`aria_nbv.utils.viz_utils` to avoid
duplication between data and visualisation modules. Keep these names as thin
re-exports for backwards compatibility with existing callers and tests.
"""

from aria_nbv.utils.viz_utils import extract_scene_id_from_sequence_name, validate_scene_data

__all__ = ["extract_scene_id_from_sequence_name", "validate_scene_data"]
