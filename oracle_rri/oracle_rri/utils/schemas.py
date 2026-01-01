"""Small shared enums used by Lightning-based training utilities."""

from __future__ import annotations

from enum import StrEnum
from typing import Self


class Stage(StrEnum):
    """Stages of the training lifecycle.

    Members:
        TRAIN: "train"
        VAL: "val"
        TEST: "test"
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str | Self) -> Self:
        """Map strings (e.g. "fit", "validate") back to Stage members."""
        if isinstance(value, cls):
            return value
        alias_map: dict[str, Self] = {
            "fit": cls.TRAIN,
            "validate": cls.VAL,
        }
        if value in alias_map:
            return alias_map[value]
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Unknown stage value '{value}' of type {type(value)}.") from exc


__all__ = ["Stage"]
