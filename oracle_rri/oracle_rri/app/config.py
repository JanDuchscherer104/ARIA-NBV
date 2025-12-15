"""Config-as-factory wrapper for the refactored Streamlit app."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from ..data import AseEfmDatasetConfig
from ..pipelines import OracleRriLabelerConfig
from ..utils import BaseConfig

if TYPE_CHECKING:
    from .app import NbvStreamlitApp


def _target_cls():
    from .app import NbvStreamlitApp

    return NbvStreamlitApp


class NbvStreamlitAppConfig(BaseConfig["NbvStreamlitApp"]):
    """Top-level config for the refactored Streamlit app."""

    target: type["NbvStreamlitApp"] = Field(default_factory=_target_cls, exclude=True)

    dataset: AseEfmDatasetConfig = Field(default_factory=AseEfmDatasetConfig)
    """Dataset configuration used by the app."""

    labeler: OracleRriLabelerConfig = Field(default_factory=OracleRriLabelerConfig)
    """Oracle label pipeline configuration (candidates → depth → RRI)."""


__all__ = ["NbvStreamlitAppConfig"]
