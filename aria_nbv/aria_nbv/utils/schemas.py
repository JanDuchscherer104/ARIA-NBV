"""Small shared enums used by training and reporting utilities."""

from __future__ import annotations

from enum import StrEnum
from typing import Self


class ValueStrEnum(StrEnum):
    """StrEnum variant whose string form is always its underlying value."""

    def __str__(self) -> str:
        return self.value


class Stage(ValueStrEnum):
    """Stages of the training lifecycle.

    Members:
        TRAIN: "train"
        VAL: "val"
        TEST: "test"
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @classmethod
    def from_str(cls, value: str | Self) -> Self:
        """Map strings (e.g. "fit", "validate") back to Stage members."""
        value = value.lower().strip() if isinstance(value, str) else value
        if isinstance(value, cls):
            return value
        alias_map: dict[str, Self] = {
            "fit": cls.TRAIN,
            "validate": cls.VAL,
            "valid": cls.VAL,
        }
        if value in alias_map:
            return alias_map[value]
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Unknown stage value '{value}' of type {type(value)}.") from exc


__all__ = ["Stage", "ValueStrEnum"]
