"""Config-as-Factory wrapper for the Streamlit dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from ...data import AseEfmDatasetConfig
from ...pose_generation import CandidateViewGeneratorConfig
from ...rendering import CandidateDepthRendererConfig, Pytorch3DDepthRendererConfig
from ...utils import BaseConfig

if TYPE_CHECKING:
    from .app import DashboardApp


def _target_cls():
    from .app import DashboardApp

    return DashboardApp


class DashboardConfig(BaseConfig["DashboardApp"]):
    """Top-level config for Streamlit dashboard.

    Uses Config-as-Factory; :meth:`setup_target` builds :class:`DashboardApp`.
    """

    target: type["DashboardApp"] = Field(default_factory=_target_cls, exclude=True)

    dataset: AseEfmDatasetConfig = Field(default_factory=AseEfmDatasetConfig)
    generator: CandidateViewGeneratorConfig = Field(default_factory=CandidateViewGeneratorConfig)
    renderer: CandidateDepthRendererConfig = Field(
        default_factory=lambda: CandidateDepthRendererConfig(renderer=Pytorch3DDepthRendererConfig(device="cuda"))
    )

    super_fast: bool = False
    debug: bool = True


__all__ = ["DashboardConfig"]
