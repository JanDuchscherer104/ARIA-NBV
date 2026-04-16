"""Config-as-factory wrapper for the refactored Streamlit app."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, field_validator

from ..data_handling import AseEfmDatasetConfig
from ..pipelines import OracleRriLabelerConfig
from ..rl import CounterfactualPPOConfig, CounterfactualRLEnvConfig
from ..utils import BaseConfig

if TYPE_CHECKING:
    from .app import NbvStreamlitApp


def _target_cls():
    from .app import NbvStreamlitApp

    return NbvStreamlitApp


class NbvStreamlitAppConfig(BaseConfig):
    """Top-level config for the refactored Streamlit app."""

    @property
    def target(self) -> type["NbvStreamlitApp"]:
        return _target_cls()

    dataset: AseEfmDatasetConfig = Field(default_factory=AseEfmDatasetConfig)
    """Dataset configuration used by the app."""

    labeler: OracleRriLabelerConfig = Field(default_factory=OracleRriLabelerConfig)
    """Oracle label pipeline configuration (candidates → depth → RRI)."""

    rl: "RlPageConfig" = Field(default_factory=lambda: RlPageConfig())
    """Configuration for the optional RL-specific Streamlit page."""


class RlPageConfig(BaseConfig):
    """Config-gated Streamlit controls for the RL inspection page."""

    enabled: bool = True
    """Whether to expose the RL page in Streamlit navigation."""

    enable_policy_comparison: bool = True
    """Whether to show the multi-policy comparison tab."""

    enable_checkpoint_policy: bool = False
    """Whether to expose checkpoint loading and PPO policy playback controls."""

    enable_action_shell_plot: bool = True
    """Whether to show the initial action-shell plot for a chosen seed."""

    enable_step_shell_plot: bool = True
    """Whether to show per-step candidate-shell plots for completed episodes."""

    enable_episode_table: bool = True
    """Whether to show per-step episode tables."""

    default_eval_episodes: int = 4
    """Default number of episodes for policy comparison."""

    max_eval_episodes: int = 16
    """Maximum number of episodes exposed in the UI."""

    env: CounterfactualRLEnvConfig = Field(default_factory=CounterfactualRLEnvConfig)
    """Base RL environment configuration merged with the active labeler settings."""

    ppo: CounterfactualPPOConfig = Field(default_factory=CounterfactualPPOConfig)
    """Default PPO configuration used when loading or instantiating SB3 policies."""

    @field_validator("default_eval_episodes", "max_eval_episodes")
    @classmethod
    def _positive_ints(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("Episode-count controls must be >= 1.")
        return int(value)


__all__ = ["NbvStreamlitAppConfig", "RlPageConfig"]
