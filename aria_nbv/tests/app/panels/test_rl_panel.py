"""Tests for the RL Streamlit panel helpers."""

# ruff: noqa: S101

from __future__ import annotations

from types import SimpleNamespace

from aria_nbv.app.config import RlPageConfig
from aria_nbv.app.panels import rl as rl_panel
from aria_nbv.pipelines import OracleRriLabelerConfig
from aria_nbv.rl import CounterfactualRLEnvConfig


def test_effective_rl_env_config_uses_active_labeler_subconfigs() -> None:
    labeler_cfg = OracleRriLabelerConfig()
    panel_cfg = RlPageConfig(env=CounterfactualRLEnvConfig(horizon=5, invalid_action_penalty=-0.2))

    env_cfg = rl_panel._effective_rl_env_config(labeler_cfg, panel_cfg)
    shell_capacity = int(labeler_cfg.generator.num_samples * labeler_cfg.generator.oversample_factor)

    assert env_cfg.horizon == 5
    assert env_cfg.invalid_action_penalty == -0.2
    assert env_cfg.candidate_config == labeler_cfg.generator
    assert env_cfg.reward.oracle == labeler_cfg.oracle
    assert env_cfg.reward.backprojection_stride == labeler_cfg.backprojection_stride
    assert env_cfg.reward.depth.renderer == labeler_cfg.depth.renderer
    assert env_cfg.reward.depth.oversample_factor == labeler_cfg.depth.oversample_factor
    assert env_cfg.reward.depth.resolution_scale == labeler_cfg.depth.resolution_scale
    assert env_cfg.reward.depth.max_candidates_final >= shell_capacity


def test_seeded_env_config_overrides_candidate_seed_only() -> None:
    env_cfg = CounterfactualRLEnvConfig()

    seeded = rl_panel._seeded_env_config(env_cfg, seed=7)

    assert seeded.candidate_config.seed == 7
    assert seeded.horizon == env_cfg.horizon
    assert seeded.invalid_action_penalty == env_cfg.invalid_action_penalty


def test_episode_cache_key_tracks_policy_seed_and_checkpoint() -> None:
    sample = SimpleNamespace(scene_id="scene_a", snippet_id="snippet_1")
    env_cfg = CounterfactualRLEnvConfig()

    key_a = rl_panel._episode_cache_key(sample, env_cfg, policy_name="random", seed=0, checkpoint_path=None)
    key_b = rl_panel._episode_cache_key(sample, env_cfg, policy_name="greedy_reward", seed=0, checkpoint_path=None)
    key_c = rl_panel._episode_cache_key(
        sample,
        env_cfg,
        policy_name="ppo_checkpoint",
        seed=0,
        checkpoint_path="/tmp/model.zip",
    )

    assert key_a != key_b
    assert key_b != key_c
