"""Streamlit helpers for inspecting persisted rollout Zarr stores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ...pose_generation import ViewDirectionMode, candidate_strategy_id
from ...rollouts import RolloutZarrStoreReader
from ..rerun_launch import (
    build_rerun_rollout_spawn_command,
    build_rerun_rollout_web_command,
    format_command,
    repo_root,
    rerun_web_url,
    spawn_background_command,
)
from .common import _info_popover, _report_exception

_STORED_ROLLOUTS_INFO = """
Stored rollouts inspect a standalone `rollouts.zarr` shard without recomputation.

- Validation metadata checks table/mask consistency before inspection.
- The manifest records source/config lineage and source coverage.
- Target summary separates actor target validity from GT-label validity.
- Candidate rows show `actor_action`, `q_train`, target/scene RRI labels, and strategy/mixture provenance.
- `q_train` marks rows usable for finite-candidate `Q_H` training views.
- Rerun launch opens or serves the selected rollout row in the 3D inspector.
"""

_CANDIDATE_TABLE_INFO = """
Candidate row fields:

- `candidate_row_id`: stable row id inside `candidates/`.
- `step_index`: rollout step that generated this candidate shell.
- `shell_index`: index in the full sampled shell before valid-row compaction.
- `selected`: whether the rollout policy selected this candidate.
- `actor_action`: whether the actor may choose this row after hard masks.
- `q_train`: whether this row has the labels and masks required for `Q_H` training.
- `target_rri`: target-specific oracle RRI label when available.
- `scene_rri`: scene-level oracle RRI audit label when available.
- `strategy`: candidate-generation family decoded from `strategy_id`.
- `mixture`: mixture component id; component names are shown when persisted, otherwise `component_<id>`.
"""


def render_stored_rollouts_panel() -> None:
    """Render persisted rollout-Zarr validation, summaries, and Rerun launch."""

    st.header("Stored Rollout Zarr")
    st.caption("Load a standalone rollouts.zarr store, validate row contracts, and open selected rows in Rerun.")
    _info_popover("stored rollouts", _STORED_ROLLOUTS_INFO)

    default_store = repo_root() / ".data" / "offline_cache" / "rollouts_v1_smoke.zarr"
    default_config = repo_root() / ".configs" / "rerun_offline.toml"
    default_save = repo_root() / ".artifacts" / "rerun"
    store_path = Path(
        st.text_input("rollouts.zarr path", value=str(default_store), key="rollout_store_path")
    ).expanduser()
    config_path = Path(
        st.text_input("Rerun inspector config", value=str(default_config), key="rollout_rerun_config_path")
    ).expanduser()

    web_col1, web_col2, web_col3 = st.columns(3)
    web_viewer_port = int(
        web_col1.number_input("Rerun web-viewer port", min_value=0, max_value=65535, value=9090, step=1)
    )
    ws_server_port = int(
        web_col2.number_input("Rerun gRPC/proxy port", min_value=0, max_value=65535, value=9877, step=1)
    )
    save_dir = Path(web_col3.text_input("RRD save directory", value=str(default_save))).expanduser()

    if not store_path.exists():
        st.info("Enter an existing rollouts.zarr path to inspect generated rollout rows.")
        return

    try:
        reader = RolloutZarrStoreReader(store_path)
        manifest_bundle = reader.manifest()
        validation = reader.validate()
    except Exception as exc:  # pragma: no cover - UI guard
        _report_exception(exc, context="Failed to load rollout store")
        return

    root_attrs = dict(reader.root.attrs)
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    meta_col1.metric("Rollouts", validation.num_rollouts)
    meta_col2.metric("Steps", validation.num_steps)
    meta_col3.metric("Candidates", validation.num_candidates)
    meta_col4.metric("Validation", "OK" if validation.ok else "FAILED")
    if validation.ok:
        st.success("Store validation passed.")
    else:
        st.error("Store validation failed.")
        st.dataframe([{"error": error} for error in validation.errors], width="stretch", hide_index=True)

    with st.expander("Root metadata", expanded=False):
        st.json(root_attrs)
    with st.expander("Generation manifest", expanded=False):
        st.json(manifest_bundle["manifest"])

    _render_rollout_store_summaries(reader, manifest=manifest_bundle["manifest"])
    _render_stored_metric_dashboard(reader)
    rollout_ids = reader.array("rollouts/rollout_row_id").astype(int).tolist()
    if not rollout_ids:
        st.info("No rollout rows are present.")
        return

    selected_rollout = int(
        st.selectbox(
            "Rollout row",
            options=rollout_ids,
            format_func=lambda row_id: format_rollout_option(reader, row_id),
            key="rollout_row_selector",
        )
    )
    _info_popover("rollout candidate rows", _CANDIDATE_TABLE_INFO)
    st.dataframe(candidate_rows_for_rollout(reader, selected_rollout), width="stretch", hide_index=True)

    native_command = build_rerun_rollout_spawn_command(
        config_path=config_path,
        rollout_store=store_path,
        rollout_row_id=selected_rollout,
    )
    save_path = save_dir / f"rollout_row_{selected_rollout}.rrd"
    web_command = build_rerun_rollout_web_command(
        config_path=config_path,
        rollout_store=store_path,
        rollout_row_id=selected_rollout,
        save_path=save_path,
        web_viewer_port=web_viewer_port,
        ws_server_port=ws_server_port,
        lan=True,
    )
    launch_col1, launch_col2 = st.columns(2)
    with launch_col1:
        st.markdown("**Native Rerun**")
        st.code(format_command(native_command), language="bash")
        if st.button("Open in Native Rerun", key="rollout_open_native_rerun"):
            _spawn_rerun_command(native_command, success_prefix="Spawned native Rerun inspector")
    with launch_col2:
        st.markdown("**Rerun Web Viewer**")
        st.code(format_command(web_command), language="bash")
        st.caption(f"Expected URL: {rerun_web_url(web_viewer_port=web_viewer_port, lan=True)}")
        if st.button("Open in Rerun Web Viewer", key="rollout_open_web_rerun"):
            save_path.parent.mkdir(parents=True, exist_ok=True)
            _spawn_rerun_command(web_command, success_prefix="Started Rerun web viewer")


def _spawn_rerun_command(command: list[str], *, success_prefix: str) -> None:
    try:
        process = spawn_background_command(command)
    except Exception as exc:  # pragma: no cover - UI guard
        _report_exception(exc, context="Failed to launch Rerun inspector")
    else:
        st.success(f"{success_prefix} with pid {process.pid}.")


def _render_rollout_store_summaries(reader: RolloutZarrStoreReader, *, manifest: dict[str, object]) -> None:
    target_rows = reader.array("targets/target_row_id")
    rollout_rows = reader.array("rollouts/rollout_row_id")
    step_rows = reader.array("steps/step_row_id")
    candidate_rows = reader.array("candidates/candidate_row_id")
    q_h = reader.q_h_view()
    q_train = q_h["q_train_mask"]
    valid_action = q_h["valid_action_mask"]
    actor_action = reader.array("candidates/actor_action_mask")
    oracle_label = reader.array("candidates/oracle_label_mask")
    summary = [
        {"table": "targets", "rows": int(target_rows.shape[0])},
        {"table": "rollouts", "rows": int(rollout_rows.shape[0])},
        {"table": "steps", "rows": int(step_rows.shape[0])},
        {"table": "candidates", "rows": int(candidate_rows.shape[0])},
        {"table": "actor_action candidates", "rows": int(actor_action.sum())},
        {"table": "oracle_label candidates", "rows": int(oracle_label.sum())},
        {"table": "q_h valid actions", "rows": int(valid_action.sum())},
        {"table": "q_h train cells", "rows": int(q_train.sum())},
    ]
    st.dataframe(summary, width="stretch", hide_index=True)
    coverage = manifest.get("source_coverage", {})
    if isinstance(coverage, dict) and coverage:
        st.markdown("**Source coverage**")
        st.json(coverage)
    target_summary = []
    target_valid = reader.array("targets/target_valid_mask")
    gt_valid = reader.array("targets/gt_label_valid_mask")
    for row_id, valid, gt_label in zip(target_rows, target_valid, gt_valid, strict=True):
        target_summary.append(
            {
                "target_row_id": int(row_id),
                "target_valid": bool(valid),
                "gt_label_valid": bool(gt_label),
            }
        )
    if target_summary:
        st.dataframe(target_summary, width="stretch", hide_index=True)


def _render_stored_metric_dashboard(reader: RolloutZarrStoreReader) -> None:
    rows = _stored_rollout_metric_rows(reader)
    if rows.empty:
        st.info("No rollout-level metrics are available in this store.")
        return

    finite_target = rows["final_cumulative_target_rri"].dropna()
    finite_scene = rows["final_cumulative_scene_rri"].dropna()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Mean final target RRI", _format_metric(finite_target.mean()))
    metric_cols[1].metric("Best final target RRI", _format_metric(finite_target.max()))
    metric_cols[2].metric("Mean final scene RRI", _format_metric(finite_scene.mean()))
    metric_cols[3].metric("Q-train candidates", int(reader.array("candidates/q_train_mask").sum()))

    st.caption(
        "Endpoint `J_e^(H)` and log-gain require persisted target point-mesh before/after fields; current stores only persist cumulative RRI."
    )
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            px.histogram(rows, x="final_cumulative_target_rri", color="policy", title="Final cumulative target RRI"),
            width="stretch",
        )
    with chart_col2:
        grouped = rows.groupby(["policy", "horizon"], dropna=False)["final_cumulative_target_rri"].mean().reset_index()
        st.plotly_chart(
            px.bar(
                grouped,
                x="policy",
                y="final_cumulative_target_rri",
                color="horizon",
                barmode="group",
                title="Mean final target RRI by policy and horizon",
            ),
            width="stretch",
        )


def _stored_rollout_metric_rows(reader: RolloutZarrStoreReader) -> pd.DataFrame:
    policies = _string_list(reader, "dictionaries/policy")
    scenes = _string_list(reader, "dictionaries/scene")
    rollout_ids = reader.array("rollouts/rollout_row_id")
    policy_ids = reader.array("rollouts/policy_id")
    scene_ids = reader.array("rollouts/scene_id")
    target_rows = reader.array("rollouts/target_row_id")
    horizon = reader.array("rollouts/horizon")
    branch_factor = reader.array("rollouts/branch_factor")
    target_rri = reader.array("rollouts/final_cumulative_target_rri")
    scene_rri = reader.array("rollouts/final_cumulative_scene_rri")
    return pd.DataFrame(
        [
            {
                "rollout_row_id": int(row_id),
                "scene": _dict_value(scenes, int(scene_id)),
                "target_row_id": int(target_row),
                "policy": _dict_value(policies, int(policy_id)),
                "horizon": int(h),
                "branch_factor": int(b),
                "final_cumulative_target_rri": _finite_or_none(target),
                "final_cumulative_scene_rri": _finite_or_none(scene),
            }
            for row_id, scene_id, target_row, policy_id, h, b, target, scene in zip(
                rollout_ids,
                scene_ids,
                target_rows,
                policy_ids,
                horizon,
                branch_factor,
                target_rri,
                scene_rri,
                strict=True,
            )
        ]
    )


def candidate_rows_for_rollout(reader: RolloutZarrStoreReader, rollout_row_id: int) -> list[dict[str, object]]:
    """Return display rows for one rollout's full candidate table."""

    rollout_ids = reader.array("candidates/rollout_row_id")
    mask = rollout_ids == int(rollout_row_id)
    candidate_row_ids = reader.array("candidates/candidate_row_id")
    step_indices = reader.array("candidates/step_index")
    shell_indices = reader.array("candidates/shell_index")
    selected_mask = reader.array("candidates/selected_mask")
    actor_action_mask = reader.array("candidates/actor_action_mask")
    q_train_mask = reader.array("candidates/q_train_mask")
    target_rri = reader.array("candidates/target_rri")
    scene_rri = reader.array("candidates/scene_rri")
    strategy_id = reader.array("candidates/strategy_id")
    mixture_id = reader.array("candidates/mixture_id")
    rows: list[dict[str, object]] = []
    for index in np.nonzero(mask)[0].tolist():
        rows.append(
            {
                "candidate_row_id": int(candidate_row_ids[index]),
                "step_index": int(step_indices[index]),
                "shell_index": int(shell_indices[index]),
                "selected": bool(selected_mask[index]),
                "actor_action": bool(actor_action_mask[index]),
                "q_train": bool(q_train_mask[index]),
                "target_rri": _finite_or_none(target_rri[index]),
                "scene_rri": _finite_or_none(scene_rri[index]),
                "strategy": _strategy_name(int(strategy_id[index])),
                "mixture": _mixture_name(int(mixture_id[index])),
            }
        )
    return rows


def format_rollout_option(reader: RolloutZarrStoreReader, rollout_row_id: int) -> str:
    """Format a rollout-row selector label with source and rollout context."""

    rollout_rows = reader.array("rollouts/rollout_row_id")
    matches = np.nonzero(rollout_rows == int(rollout_row_id))[0]
    if matches.size != 1:
        return f"rollout {rollout_row_id}"
    index = int(matches[0])
    policies = _string_list(reader, "dictionaries/policy")
    scenes = _string_list(reader, "dictionaries/scene")
    policy = _dict_value(policies, int(reader.array("rollouts/policy_id")[index]))
    scene = _dict_value(scenes, int(reader.array("rollouts/scene_id")[index]))
    target_row = int(reader.array("rollouts/target_row_id")[index])
    chain = int(reader.array("rollouts/chain_id")[index])
    horizon = int(reader.array("rollouts/horizon")[index])
    branch_factor = int(reader.array("rollouts/branch_factor")[index])
    beam = _format_stored_beam_width(int(reader.array("rollouts/beam_width")[index]))
    return (
        f"{rollout_row_id} · scene {scene} · target {target_row} · {policy} · "
        f"chain {chain} · H={horizon} · B={branch_factor} · beam={beam}"
    )


def _strategy_name(value: int) -> str:
    for mode in ViewDirectionMode:
        if candidate_strategy_id(mode) == int(value):
            return mode.value
    return "unknown" if int(value) < 0 else f"strategy_{int(value)}"


def _mixture_name(value: int) -> str:
    return "unknown" if int(value) < 0 else f"component_{int(value)}"


def _format_stored_beam_width(value: int) -> str:
    return "NaN" if int(value) < 0 else str(int(value))


def _string_list(reader: RolloutZarrStoreReader, path: str) -> list[str]:
    try:
        return json.loads(bytes(reader.array(path).tolist()).decode("utf-8"))
    except Exception:
        return []


def _dict_value(values: list[str], index: int) -> str:
    if index < 0 or index >= len(values):
        return ""
    return values[index]


def _finite_or_none(value: object) -> float | None:
    value_float = float(value)
    return value_float if np.isfinite(value_float) else None


def _format_metric(value: object) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{value_float:.4f}" if np.isfinite(value_float) else "n/a"


__all__ = [
    "candidate_rows_for_rollout",
    "format_rollout_option",
    "render_stored_rollouts_panel",
]
