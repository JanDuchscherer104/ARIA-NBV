"""Streamlit panel for counterfactual RL inspection and policy comparison."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import]
import streamlit as st

from ...data_handling import EfmSnippetView
from ...pipelines import OracleRriLabelerConfig
from ...pose_generation.plotting import CandidatePlotBuilder, CounterfactualPlotBuilder
from ...rl import CounterfactualRLEnv, CounterfactualRLEnvConfig
from ..config import RlPageConfig
from ..state_types import config_signature, sample_key
from .common import _info_popover, _pretty_label, _report_exception

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass(slots=True)
class RlEpisodeSummary:
    """Compact result for one evaluated RL episode."""

    policy_name: str
    seed: int
    rollout: Any
    total_return: float
    cumulative_rri: float | None
    steps: int
    invalid_actions: int
    step_rows: list[dict[str, int | float | bool | None]]


def _effective_rl_env_config(
    labeler_cfg: OracleRriLabelerConfig,
    panel_cfg: RlPageConfig,
) -> CounterfactualRLEnvConfig:
    """Merge the app labeler settings into the RL env defaults."""

    shell_capacity = int(ceil(labeler_cfg.generator.num_samples * labeler_cfg.generator.oversample_factor))
    depth_cfg = labeler_cfg.depth.model_copy(
        update={
            "max_candidates_final": max(int(labeler_cfg.depth.max_candidates_final), shell_capacity),
        }
    )
    reward_cfg = panel_cfg.env.reward.model_copy(
        update={
            "depth": depth_cfg,
            "oracle": labeler_cfg.oracle,
            "backprojection_stride": int(labeler_cfg.backprojection_stride),
        }
    )
    return panel_cfg.env.model_copy(
        update={
            "candidate_config": labeler_cfg.generator,
            "reward": reward_cfg,
        }
    )


def _seeded_env_config(env_cfg: CounterfactualRLEnvConfig, *, seed: int) -> CounterfactualRLEnvConfig:
    """Return an env config with a deterministic candidate seed for one run."""

    candidate_cfg = env_cfg.candidate_config.model_copy(update={"seed": int(seed)})
    return env_cfg.model_copy(update={"candidate_config": candidate_cfg})


def _episode_cache_key(
    sample: EfmSnippetView,
    env_cfg: CounterfactualRLEnvConfig,
    *,
    policy_name: str,
    seed: int,
    checkpoint_path: str | None,
) -> str:
    """Build a stable cache key for one policy run."""

    checkpoint_sig = "" if checkpoint_path is None else str(Path(checkpoint_path).expanduser())
    return "|".join(
        [
            sample_key(sample),
            config_signature(env_cfg),
            policy_name,
            str(int(seed)),
            checkpoint_sig,
        ]
    )


def _comparison_cache_key(
    sample: EfmSnippetView,
    env_cfg: CounterfactualRLEnvConfig,
    *,
    policy_names: list[str],
    base_seed: int,
    episode_count: int,
    checkpoint_path: str | None,
) -> str:
    """Build a stable cache key for a policy-comparison batch."""

    return "|".join(
        [
            sample_key(sample),
            config_signature(env_cfg),
            ",".join(policy_names),
            str(int(base_seed)),
            str(int(episode_count)),
            "" if checkpoint_path is None else str(Path(checkpoint_path).expanduser()),
        ]
    )


def _load_checkpoint_policy(checkpoint_path: str, env: CounterfactualRLEnv) -> "BaseAlgorithm":
    """Load an SB3 PPO checkpoint for playback."""

    from stable_baselines3 import PPO

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {path}")
    return PPO.load(path.as_posix(), env=env)


def _select_policy_action(
    env: CounterfactualRLEnv,
    obs: dict[str, Any],
    *,
    policy_name: str,
    rng,
    policy: "BaseAlgorithm | None",
) -> int:
    """Choose one shell action for the current env state."""

    if policy_name == "greedy_reward":
        action = env.greedy_action()
        return 0 if action is None else int(action)

    if policy_name == "random":
        valid = env.current_valid_shell_indices()
        return 0 if valid.size == 0 else int(rng.choice(valid))

    if policy_name == "ppo_checkpoint":
        if policy is None:
            raise ValueError("Checkpoint policy requested without a loaded SB3 model.")
        action, _ = policy.predict(obs, deterministic=True)
        return int(action)

    raise ValueError(f"Unsupported policy '{policy_name}'.")


def _run_policy_episode(
    sample: EfmSnippetView,
    env_cfg: CounterfactualRLEnvConfig,
    *,
    policy_name: str,
    seed: int,
    checkpoint_path: str | None = None,
) -> RlEpisodeSummary:
    """Run one env episode under a simple policy."""

    env = _seeded_env_config(env_cfg, seed=seed).setup_target(sample=sample)
    obs, _info = env.reset(seed=seed)

    rng = np.random.default_rng(seed)
    policy = None if policy_name != "ppo_checkpoint" else _load_checkpoint_policy(str(checkpoint_path), env)

    total_return = 0.0
    invalid_actions = 0
    step_rows: list[dict[str, int | float | bool | None]] = []
    terminated = False
    truncated = False
    env_step = 0

    while not (terminated or truncated):
        action = _select_policy_action(
            env,
            obs,
            policy_name=policy_name,
            rng=rng,
            policy=policy,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        env_step += 1

        is_valid = info.get("selected_valid_index") is not None
        if not is_valid:
            invalid_actions += 1

        final_pos = [None, None, None]
        if is_valid:
            pose = env.as_rollout_result().trajectories[0].final_pose_world().t.detach().cpu().reshape(-1).tolist()
            final_pos = [float(pose[0]), float(pose[1]), float(pose[2])]

        step_rows.append(
            {
                "env_step": env_step,
                "action": int(action),
                "valid_action": bool(is_valid),
                "reward": float(reward),
                "cumulative_return": float(total_return),
                "cumulative_rri": (None if info.get("cumulative_rri") is None else float(info["cumulative_rri"])),
                "selected_shell_index": (
                    None if info.get("selected_shell_index") is None else int(info["selected_shell_index"])
                ),
                "selected_valid_index": (
                    None if info.get("selected_valid_index") is None else int(info["selected_valid_index"])
                ),
                "final_x": final_pos[0],
                "final_y": final_pos[1],
                "final_z": final_pos[2],
            }
        )

    rollout = env.as_rollout_result()
    trajectory = rollout.trajectories[0]
    return RlEpisodeSummary(
        policy_name=policy_name,
        seed=int(seed),
        rollout=rollout,
        total_return=float(total_return),
        cumulative_rri=None if trajectory.cumulative_rri is None else float(trajectory.cumulative_rri),
        steps=len(step_rows),
        invalid_actions=invalid_actions,
        step_rows=step_rows,
    )


def _comparison_boxplot(df: pd.DataFrame) -> go.Figure:
    """Build a compact box plot over total return by policy."""

    fig = go.Figure()
    for policy_name in df["policy"].drop_duplicates().tolist():
        subset = df[df["policy"] == policy_name]
        fig.add_trace(
            go.Box(
                y=subset["total_return"],
                name=policy_name,
                boxmean=True,
            )
        )
    fig.update_layout(
        title=_pretty_label("Episode return by policy"),
        yaxis_title="Total return",
        xaxis_title="Policy",
    )
    return fig


def _comparison_rri_bar(df: pd.DataFrame) -> go.Figure:
    """Build a mean cumulative-RRI bar chart for policy comparison."""

    summary = df.groupby("policy", dropna=False)[["cumulative_rri"]].mean(numeric_only=True).reset_index().fillna(0.0)
    fig = go.Figure(
        data=go.Bar(
            x=summary["policy"],
            y=summary["cumulative_rri"],
            marker_color="teal",
        )
    )
    fig.update_layout(
        title=_pretty_label("Mean cumulative RRI by policy"),
        xaxis_title="Policy",
        yaxis_title="Mean cumulative RRI",
    )
    return fig


def render_rl_page(
    sample: EfmSnippetView | None,
    *,
    labeler_cfg: OracleRriLabelerConfig,
    panel_cfg: RlPageConfig,
) -> None:
    """Render the config-gated RL inspection page."""

    st.header("RL Inspector")
    if not panel_cfg.enabled:
        st.info("The RL page is disabled by app config.")
        return
    if sample is None:
        st.info("RL inspection requires an attached EFM snippet.")
        return
    if sample.mesh is None or sample.mesh_verts is None or sample.mesh_faces is None:
        st.info("RL inspection requires the current sample to include GT mesh, verts, and faces.")
        return

    env_cfg = _effective_rl_env_config(labeler_cfg, panel_cfg)
    shell_size = int(ceil(env_cfg.candidate_config.num_samples * env_cfg.candidate_config.oversample_factor))

    _info_popover(
        "rl inspector",
        "This page evaluates shell-based counterfactual RL episodes on the current sample. "
        "It is evaluation-first: the app can preview shells, replay one episode, and compare "
        "simple policies, while PPO training itself remains outside Streamlit.",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Horizon", int(env_cfg.horizon))
    col2.metric("Invalid penalty", f"{env_cfg.invalid_action_penalty:.3f}")
    col3.metric("Shell actions", shell_size)
    col4.metric("PPO gamma", f"{panel_cfg.ppo.gamma:.2f}")

    with st.expander("RL config", expanded=False):
        st.json(env_cfg.model_dump_cache())

    tab_names = ["Episode Inspector"]
    if panel_cfg.enable_policy_comparison:
        tab_names.append("Policy Comparison")
    tabs = st.tabs(tab_names)

    policy_options = ["greedy_reward", "random"]
    if panel_cfg.enable_checkpoint_policy:
        policy_options.append("ppo_checkpoint")

    with tabs[0]:
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            policy_name = st.selectbox(
                "Policy",
                options=policy_options,
                index=0,
                key="rl_policy_name",
            )
        with ctrl2:
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=0,
                step=1,
                key="rl_seed",
            )
        with ctrl3:
            checkpoint_path = None
            if panel_cfg.enable_checkpoint_policy and policy_name == "ppo_checkpoint":
                checkpoint_path = (
                    st.text_input(
                        "PPO checkpoint",
                        value="",
                        key="rl_checkpoint_path",
                    ).strip()
                    or None
                )
            else:
                checkpoint_path = None

        preview_cache = st.session_state.setdefault("rl_preview_cache", {})
        episode_cache = st.session_state.setdefault("rl_episode_cache", {})
        seeded_env_cfg = _seeded_env_config(env_cfg, seed=int(seed))

        if panel_cfg.enable_action_shell_plot:
            preview_key = _episode_cache_key(
                sample,
                seeded_env_cfg,
                policy_name="preview",
                seed=int(seed),
                checkpoint_path=None,
            )
            preview_requested = st.button("Preview action shell", key="rl_preview_shell")
            if preview_requested:
                try:
                    env = seeded_env_cfg.setup_target(sample=sample)
                    env.reset(seed=int(seed))
                    preview_cache[preview_key] = {
                        "candidates": env.current_candidates_result(),
                        "scores": env.current_reward_scores(),
                        "score_label": env.current_score_label() or "reward",
                        "best_action": env.greedy_action(),
                    }
                except Exception as exc:  # pragma: no cover - UI guard
                    _report_exception(exc, context="RL shell preview failed")
                    preview_cache.pop(preview_key, None)

            preview = preview_cache.get(preview_key)
            if preview is not None and preview["candidates"] is not None:
                st.caption(
                    f"Best immediate action for seed {int(seed)}: "
                    f"{preview['best_action'] if preview['best_action'] is not None else 'n/a'}"
                )
                builder = CandidatePlotBuilder.from_candidates(
                    sample,
                    preview["candidates"],
                    title=_pretty_label("Initial action shell"),
                ).add_mesh()
                if preview["scores"] is not None:
                    builder = builder.add_candidate_points(
                        use_valid=True,
                        color=preview["scores"],
                        colorbar_title=str(preview["score_label"]).replace("_", " "),
                        name="Immediate reward",
                        size=4,
                        opacity=0.85,
                        mark_reference=True,
                    )
                else:
                    builder = builder.add_candidate_cloud(use_valid=True, name="Valid actions")
                st.plotly_chart(builder.add_reference_axes(display_rotate=True).finalize(), width="stretch")

        run_key = _episode_cache_key(
            sample,
            seeded_env_cfg,
            policy_name=policy_name,
            seed=int(seed),
            checkpoint_path=checkpoint_path,
        )
        run_requested = st.button("Run episode", key="rl_run_episode")
        if run_requested:
            try:
                if policy_name == "ppo_checkpoint" and not checkpoint_path:
                    raise ValueError("Provide a PPO checkpoint path for checkpoint playback.")
                episode_cache[run_key] = _run_policy_episode(
                    sample,
                    seeded_env_cfg,
                    policy_name=policy_name,
                    seed=int(seed),
                    checkpoint_path=checkpoint_path,
                )
            except Exception as exc:  # pragma: no cover - UI guard
                _report_exception(exc, context="RL episode run failed")
                episode_cache.pop(run_key, None)

        episode = episode_cache.get(run_key)
        if episode is not None:
            stat1, stat2, stat3, stat4 = st.columns(4)
            stat1.metric("Total return", f"{episode.total_return:.4f}")
            stat2.metric("Cumulative RRI", "n/a" if episode.cumulative_rri is None else f"{episode.cumulative_rri:.4f}")
            stat3.metric("Env steps", int(episode.steps))
            stat4.metric("Invalid actions", int(episode.invalid_actions))

            trajectory = episode.rollout.trajectories[0]
            path_builder = CounterfactualPlotBuilder.from_rollouts(
                sample,
                episode.rollout,
                title=_pretty_label("RL episode trajectory"),
            ).add_mesh()
            path_builder = path_builder.add_counterfactual_paths(show_step_markers=True)
            if trajectory.steps:
                path_builder = path_builder.add_counterfactual_selected_frusta(scale=0.45)
            st.plotly_chart(path_builder.finalize(), width="stretch")

            if panel_cfg.enable_episode_table:
                st.dataframe(pd.DataFrame(episode.step_rows), width="stretch")

            if panel_cfg.enable_step_shell_plot and trajectory.steps:
                step_display = st.selectbox(
                    "Step shell",
                    options=list(range(1, len(trajectory.steps) + 1)),
                    index=0,
                    key="rl_step_shell",
                )
                step_builder = (
                    CounterfactualPlotBuilder.from_rollouts(
                        sample,
                        episode.rollout,
                        title=_pretty_label(f"RL step {step_display} shell"),
                    )
                    .add_mesh()
                    .add_counterfactual_step_shell(
                        trajectory_index=0,
                        step_index=int(step_display) - 1,
                        show_history=True,
                        show_selected=True,
                        show_frusta=True,
                        include_rejected=False,
                    )
                )
                st.plotly_chart(step_builder.finalize(), width="stretch")

    if panel_cfg.enable_policy_comparison:
        with tabs[1]:
            cmp1, cmp2, cmp3 = st.columns(3)
            with cmp1:
                selected_policies = st.multiselect(
                    "Policies",
                    options=policy_options,
                    default=["greedy_reward", "random"],
                    key="rl_compare_policies",
                )
            with cmp2:
                episode_count = st.slider(
                    "Episodes",
                    min_value=1,
                    max_value=int(panel_cfg.max_eval_episodes),
                    value=int(min(panel_cfg.default_eval_episodes, panel_cfg.max_eval_episodes)),
                    step=1,
                    key="rl_compare_episode_count",
                )
            with cmp3:
                base_seed = st.number_input(
                    "Base seed",
                    min_value=0,
                    value=0,
                    step=1,
                    key="rl_compare_base_seed",
                )

            compare_checkpoint = None
            if panel_cfg.enable_checkpoint_policy and "ppo_checkpoint" in selected_policies:
                compare_checkpoint = (
                    st.text_input(
                        "Comparison PPO checkpoint",
                        value="",
                        key="rl_compare_checkpoint_path",
                    ).strip()
                    or None
                )

            compare_cache = st.session_state.setdefault("rl_compare_cache", {})
            compare_key = _comparison_cache_key(
                sample,
                env_cfg,
                policy_names=selected_policies,
                base_seed=int(base_seed),
                episode_count=int(episode_count),
                checkpoint_path=compare_checkpoint,
            )

            compare_requested = st.button("Run comparison", key="rl_run_comparison")
            if compare_requested:
                try:
                    if "ppo_checkpoint" in selected_policies and not compare_checkpoint:
                        raise ValueError("Provide a PPO checkpoint path for checkpoint comparison.")
                    compare_rows: list[dict[str, int | float | None | str]] = []
                    for policy_name in selected_policies:
                        for offset in range(int(episode_count)):
                            episode_seed = int(base_seed) + int(offset)
                            summary = _run_policy_episode(
                                sample,
                                env_cfg,
                                policy_name=policy_name,
                                seed=episode_seed,
                                checkpoint_path=compare_checkpoint,
                            )
                            compare_rows.append(
                                {
                                    "policy": policy_name,
                                    "seed": episode_seed,
                                    "total_return": float(summary.total_return),
                                    "cumulative_rri": (
                                        None if summary.cumulative_rri is None else float(summary.cumulative_rri)
                                    ),
                                    "steps": int(summary.steps),
                                    "invalid_actions": int(summary.invalid_actions),
                                }
                            )
                    compare_cache[compare_key] = pd.DataFrame(compare_rows)
                except Exception as exc:  # pragma: no cover - UI guard
                    _report_exception(exc, context="RL policy comparison failed")
                    compare_cache.pop(compare_key, None)

            compare_df = compare_cache.get(compare_key)
            if compare_df is not None and not compare_df.empty:
                st.dataframe(compare_df, width="stretch")
                st.plotly_chart(_comparison_boxplot(compare_df), width="stretch")
                st.plotly_chart(_comparison_rri_bar(compare_df), width="stretch")


__all__ = [
    "RlEpisodeSummary",
    "_comparison_cache_key",
    "_effective_rl_env_config",
    "_episode_cache_key",
    "_seeded_env_config",
    "render_rl_page",
]
