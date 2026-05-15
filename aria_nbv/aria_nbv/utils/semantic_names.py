"""Semantic-id class-name helpers shared by data and visualization code."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

SemanticNameMap: TypeAlias = dict[int, str]
"""Sparse mapping from EFM semantic id to human-readable class name."""


def normalize_semantic_name_map(value: Mapping[object, object] | Sequence[object] | None) -> SemanticNameMap | None:
    """Normalize raw EFM semantic-name metadata into a sparse id map.

    Args:
        value: Raw semantic metadata from EFM, msgpack, or legacy dense lists.

    Returns:
        Sparse ``{semantic_id: class_name}`` mapping, or ``None`` when missing.
    """

    if value is None:
        return None
    if isinstance(value, Mapping):
        return {int(key): str(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        raise TypeError("Semantic-name metadata must be a mapping or sequence of names, not a string.")
    return {index: str(item) for index, item in enumerate(value)}


def semantic_class_name(sem_id: int | float, sem_id_to_name: Mapping[int, str] | Sequence[str] | None) -> str:
    """Return a display class name for one EFM semantic id."""

    index = int(sem_id)
    mapping = normalize_semantic_name_map(sem_id_to_name)
    if mapping is None:
        return "<unknown>"
    name = str(mapping.get(index, ""))
    if not name or name == str(index):
        return "<unknown>"
    return name
