"""Page renderers for the refactored Streamlit app.

These functions are mostly UI/plotting code and intentionally contain no heavy
compute. Any expensive operations should be performed via
:class:`oracle_rri.app.controller.PipelineController`.
"""

from __future__ import annotations

import json
import math
import os
import re
import traceback
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import torch
from efm3d.aria.pose import PoseTW
from matplotlib import colormaps

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # Optional dependency for W&B diagnostics.
    import wandb  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency guard
    wandb = None
from ..configs import PathConfig
from ..data import AseEfmDatasetConfig, EfmSnippetView
from ..data.efm_views import EfmCameraView
from ..data.offline_cache import (
    OracleRriCacheConfig,
    OracleRriCacheDataset,
    OracleRriCacheDatasetConfig,
    rebuild_cache_index,
)
from ..data.offline_cache_types import OracleRriCacheSample
from ..data.plotting import (
    FrameGridBuilder,
    SnippetPlotBuilder,
    collect_frame_modalities,
    plot_first_last_frames,
    project_pointcloud_on_frame,
)
from ..lightning.aria_nbv_experiment import AriaNBVExperimentConfig
from ..lightning.lit_datamodule import VinOracleBatch
from ..pose_generation import CandidateViewGeneratorConfig
from ..pose_generation.plotting import (
    CandidatePlotBuilder,
    _euler_histogram,
    plot_direction_marginals,
    plot_direction_polar,
    plot_direction_sphere,
    plot_euler_reference,
    plot_euler_world,
    plot_min_distance_to_mesh,
    plot_path_collision_segments,
    plot_position_polar,
    plot_position_sphere,
    plot_radius_hist,
    plot_rule_masks,
    plot_rule_rejection_bar,
)
from ..pose_generation.types import CandidateSamplingResult
from ..pose_generation.utils import (
    stats_to_markdown_table,
    summarise_dirs_ref,
    summarise_offsets_ref,
)
from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rendering.candidate_pointclouds import CandidatePointClouds
from ..rendering.plotting import RenderingPlotBuilder, depth_grid, depth_histogram
from ..rri_metrics.coral import coral_loss, coral_monotonicity_violation_rate
from ..rri_metrics.rri_binning import RriOrdinalBinner
from ..rri_metrics.types import RriResult
from ..utils import Stage
from ..utils.frames import world_up_tensor
from ..vin import VinForwardDiagnostics, VinPrediction
from ..vin.model_v2 import FIELD_CHANNELS_V2
from ..vin.plotting import (
    DEFAULT_PLOT_CFG,
    build_alignment_figures,
    build_backbone_evidence_figures,
    build_candidate_encoding_figures,
    build_field_slice_figures,
    build_field_token_histograms,
    build_frustum_samples_figure,
    build_geometry_overview_figure,
    build_lff_empirical_figures,
    build_pos_grid_linearity_figure,
    build_pose_enc_pca_figure,
    build_pose_grid_pca_figure,
    build_pose_grid_slices_figure,
    build_pose_vec_histogram,
    build_prediction_alignment_figure,
    build_scene_field_evidence_figures,
    build_se3_closure_figure,
    build_vin_encoding_figures,
    build_voxel_inbounds_figure,
    build_voxel_roundtrip_figure,
    save_vin_encoding_figures,
)
from .state import VIN_DIAG_STATE_KEY, get_vin_state
from .state_types import config_signature

if TYPE_CHECKING:
    from oracle_rri.lightning.lit_module import VinLightningModule

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _pretty_label(text: str) -> str:
    """Format labels by replacing underscores and title-casing words."""
    if not text:
        return text
    return text.replace("_", " ").title()


def _info_popover(label: str, text: str) -> None:
    with st.popover(f"Info: {label.title()}", icon="ℹ️"):
        st.markdown(text, unsafe_allow_html=True)


def _report_exception(exc: Exception, *, context: str) -> None:
    """Render a full traceback in the UI and emit it to stdout."""
    trace = traceback.format_exc()
    print(trace, flush=True)
    st.error(f"{context}: {type(exc).__name__}: {exc}")
    st.exception(exc)
    with st.expander("Full traceback", expanded=False):
        st.code(trace, language="text")


@dataclass(slots=True)
class Scene3DPlotOptions:
    """Plot options for the data page 3D scene view."""

    show_scene_bounds: bool
    show_crop_bounds: bool
    show_frustum: bool
    frustum_frame_indices: list[int]
    frustum_scale: float
    mark_first_last: bool
    show_gt_obbs: bool
    gt_timestamp: int | None
    semidense_mode: str
    max_sem_points: int


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _scalar_to_rgb(
    values: np.ndarray,
    *,
    percentile: float,
    symmetric: bool,
    cmap_name: str = "viridis",
) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape + (3,), dtype=np.uint8)
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(finite), percentile))
        vmin = -vmax
    else:
        lower = max(0.0, 100.0 - percentile)
        vmin = float(np.nanpercentile(finite, lower))
        vmax = float(np.nanpercentile(finite, percentile))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (values - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    return rgb


def _plot_slice_grid(
    slices: list[np.ndarray],
    *,
    titles: list[str],
    title: str,
    cols: int = 4,
    percentile: float = 99.0,
    symmetric: bool = False,
    cmap_name: str = "viridis",
) -> go.Figure:
    if not slices:
        return go.Figure()
    num = len(slices)
    cols = max(1, min(cols, num))
    rows = int(np.ceil(num / cols))
    builder = FrameGridBuilder(
        rows=rows,
        cols=cols,
        titles=titles,
        height=320 * rows,
        width=320 * cols,
        title=title,
    )
    for idx, arr in enumerate(slices):
        r = idx // cols + 1
        c = idx % cols + 1
        rgb = _scalar_to_rgb(
            arr,
            percentile=percentile,
            symmetric=symmetric,
            cmap_name=cmap_name,
        )
        builder.add_image(rgb, row=r, col=c)
    return builder.finalize()


def _histogram_overlay(
    series: list[tuple[str, np.ndarray]],
    *,
    bins: int,
    title: str,
    xaxis_title: str,
    log1p_counts: bool,
) -> go.Figure:
    all_vals: list[np.ndarray] = []
    for _, values in series:
        vals = np.asarray(values, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            all_vals.append(vals)
    if not all_vals:
        return go.Figure()

    edges = np.histogram_bin_edges(np.concatenate(all_vals, axis=0), bins=int(bins))
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = go.Figure()
    for name, values in series:
        vals = np.asarray(values, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        counts, _ = np.histogram(vals, bins=edges)
        y = np.log1p(counts) if log1p_counts else counts
        fig.add_trace(
            go.Bar(x=centers, y=y, name=_pretty_label(name), opacity=0.6),
        )
    fig.update_layout(
        barmode="overlay",
        title=_pretty_label(title),
        xaxis_title=_pretty_label(xaxis_title),
        yaxis_title=_pretty_label("log1p(count)" if log1p_counts else "count"),
    )
    return fig


def _plot_hist_counts_mpl(
    values: list[float] | np.ndarray,
    *,
    bins: int,
    log1p_counts: bool,
    ax: plt.Axes,
    color: str | None = None,
) -> None:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    counts, edges = np.histogram(vals, bins=int(bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    y = np.log1p(counts) if log1p_counts else counts
    ax.bar(centers, y, width=widths, align="center", color=color, alpha=0.7)


def _parameter_distribution(
    model: torch.nn.Module,
    *,
    trainable_only: bool = True,
) -> pd.DataFrame:
    """Aggregate parameter counts by top-level module name."""
    rows: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        module = name.split(".", 1)[0]
        rows.append({"module": module, "num_params": int(param.numel())})
    if not rows:
        return pd.DataFrame(columns=["module", "num_params"])
    df = pd.DataFrame(rows)
    df = df.groupby("module", as_index=False)["num_params"].sum()
    return df.sort_values("num_params", ascending=False)


def _load_rri_fit_data(path: Path) -> torch.Tensor:
    """Load flattened RRI samples from a saved binner fit-data file."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    chunks = state.get("rri_chunks", [])
    if not chunks:
        return torch.empty((0,), dtype=torch.float32)
    flat = torch.cat(
        [torch.as_tensor(chunk, dtype=torch.float32).reshape(-1) for chunk in chunks],
        dim=0,
    )
    flat = flat[torch.isfinite(flat)]
    return flat


def _load_binner_data(path: Path) -> RriOrdinalBinner:
    """Load a fitted RRI binner (edges + optional bin stats) from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return RriOrdinalBinner.from_dict(data)


def _wandb_run_candidates(
    run_ref: str,
    *,
    entity: str | None,
    project: str | None,
) -> list[str]:
    ref = run_ref.strip()
    if not ref:
        return []
    candidates: list[str] = []
    if "/" in ref:
        candidates.append(ref)
        if not ref.startswith("/"):
            candidates.append(f"/{ref}")
    if entity and project:
        candidates.append(f"{entity}/{project}/{ref}")
        candidates.append(f"{entity}/{project}/runs/{ref}")
    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def _resolve_wandb_run(
    *,
    api: Any,
    run_ref: str,
    entity: str | None,
    project: str | None,
) -> Any:
    errors: list[str] = []
    for candidate in _wandb_run_candidates(run_ref, entity=entity, project=project):
        try:
            return api.run(candidate)
        except Exception as exc:  # pragma: no cover - network/API guard
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    if entity and project:
        try:
            runs = api.runs(f"{entity}/{project}", filters={"display_name": run_ref})
        except Exception:  # pragma: no cover - optional filters guard
            runs = []
        if runs:
            return runs[0]
        try:
            for run in api.runs(f"{entity}/{project}"):
                if run.name == run_ref or run.id == run_ref:
                    return run
        except Exception:  # pragma: no cover - optional fallback guard
            pass
    raise RuntimeError(
        "Unable to resolve W&B run. Tried:\n" + "\n".join(errors[:8]) + ("\n..." if len(errors) > 8 else ""),
    )


def _load_wandb_history(
    run: Any,
    *,
    keys: list[str] | None,
    max_rows: int,
) -> pd.DataFrame:
    history = run.history(keys=keys, samples=int(max_rows))
    if isinstance(history, pd.DataFrame):
        return history
    return pd.DataFrame(history)


def _wandb_media_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("path") or value.get("file") or value.get("_path")
    path = getattr(value, "path", None)
    if isinstance(path, str):
        return path
    if isinstance(value, str):
        return value
    return None


def _wandb_media_paths(history: pd.DataFrame, key: str) -> list[str]:
    if key not in history.columns:
        return []
    paths: list[str] = []
    for item in history[key].dropna().tolist():
        path = _wandb_media_path(item)
        if path:
            paths.append(path)
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _wandb_download_media(
    run: Any,
    *,
    path: str,
    cache_dir: Path,
) -> Path | None:
    if not path:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        file_ref = run.file(path)
        file_ref.download(root=str(cache_dir), replace=True)
        return cache_dir / path
    except Exception:  # pragma: no cover - optional media download guard
        return None


def _metric_pairs(columns: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    pairs: dict[str, dict[str, dict[str, str]]] = {}
    for name in columns:
        match = re.match(r"^(train|val)/(.+?)(?:_(step|epoch))?$", name)
        if not match:
            continue
        stage, base, suffix = match.groups()
        suffix = suffix or "raw"
        pairs.setdefault(base, {}).setdefault(stage, {})[suffix] = name
    return pairs


def _select_metric_key(stage_map: dict[str, str], prefer: str) -> str | None:
    if prefer in stage_map:
        return stage_map[prefer]
    if "raw" in stage_map:
        return stage_map["raw"]
    if stage_map:
        return next(iter(stage_map.values()))
    return None


def _build_experiment_config(
    *,
    toml_path: str | None,
    stage: Stage,
    use_offline_cache: bool,
    cache_dir: str | None,
    map_location: str,
    include_efm_snippet: bool,
    include_gt_mesh: bool,
) -> AriaNBVExperimentConfig:
    if toml_path:
        cfg = AriaNBVExperimentConfig.from_toml(Path(toml_path))
    else:
        cfg = AriaNBVExperimentConfig()

    cfg.run_mode = "summarize_vin"
    cfg.stage = stage
    cfg.trainer_config.use_wandb = False
    cfg.datamodule_config.train_cache = None
    cfg.datamodule_config.val_cache = None

    if use_offline_cache:
        paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
        cache_root = cache_dir or str(
            paths.offline_cache_dir or (paths.data_root / "oracle_rri_cache"),
        )
        cache_cfg = OracleRriCacheDatasetConfig(
            cache=OracleRriCacheConfig(cache_dir=Path(cache_root), paths=paths),
            load_backbone=True,
            map_location=map_location,
            include_efm_snippet=include_efm_snippet,
            include_gt_mesh=include_gt_mesh,
        )
        cfg.datamodule_config.train_cache = cache_cfg
        cfg.datamodule_config.val_cache = cache_cfg

    return cfg


def _run_vin_debug(
    module: VinLightningModule,
    batch: VinOracleBatch,
) -> tuple[VinPrediction, VinForwardDiagnostics]:
    was_training = module.vin.training
    module.vin.eval()
    if batch.efm_snippet_view is None:
        if batch.backbone_out is None:
            raise RuntimeError(
                "VIN debug requires efm inputs or cached backbone outputs.",
            )
        efm = {}
        backbone_out = batch.backbone_out
    else:
        efm = batch.efm_snippet_view.efm
        backbone_out = batch.backbone_out
    if backbone_out is not None:
        device = backbone_out.voxel_extent.device
        if next(module.vin.parameters()).device != device:
            module.vin.to(device)
        batch = batch.to(device) if hasattr(batch, "to") else batch
        backbone_out = backbone_out.to(device)
        if hasattr(batch, "p3d_cameras"):
            batch.p3d_cameras = batch.p3d_cameras.to(device)
    with torch.no_grad():
        pred, debug = module.vin.forward_with_debug(
            efm,
            candidate_poses_world_cam=batch.candidate_poses_world_cam,
            reference_pose_world_rig=batch.reference_pose_world_rig,
            p3d_cameras=batch.p3d_cameras,
            backbone_out=backbone_out,
        )
    if was_training:
        module.vin.train()
    return pred, debug


def _vin_oracle_batch_from_cache(
    cache_sample: OracleRriCacheSample,
    *,
    efm_snippet: EfmSnippetView | None,
) -> VinOracleBatch:
    rri = cache_sample.rri
    depths = cache_sample.depths
    return VinOracleBatch(
        efm_snippet_view=efm_snippet,
        candidate_poses_world_cam=depths.poses,
        reference_pose_world_rig=depths.reference_pose,
        rri=rri.rri,
        pm_dist_before=rri.pm_dist_before,
        pm_dist_after=rri.pm_dist_after,
        pm_acc_before=rri.pm_acc_before,
        pm_comp_before=rri.pm_comp_before,
        pm_acc_after=rri.pm_acc_after,
        pm_comp_after=rri.pm_comp_after,
        p3d_cameras=depths.p3d_cameras,
        scene_id=cache_sample.scene_id,
        snippet_id=cache_sample.snippet_id,
        backbone_out=cache_sample.backbone_out,
    )


def _load_efm_snippet_for_cache(
    *,
    scene_id: str,
    snippet_id: str,
    dataset_payload: dict[str, Any] | None,
    device: str,
    paths: PathConfig,
    include_gt_mesh: bool,
) -> EfmSnippetView:
    payload = dict(dataset_payload or {})
    payload["paths"] = payload.get("paths", paths)
    payload["scene_ids"] = [scene_id]
    payload["snippet_ids"] = [snippet_id]
    payload["batch_size"] = 1
    payload["device"] = device
    payload["wds_shuffle"] = False
    payload["wds_repeat"] = False
    payload["load_meshes"] = bool(include_gt_mesh)
    payload.setdefault("require_mesh", False)
    cfg = AseEfmDatasetConfig(**payload)
    dataset = cfg.setup_target()
    return next(iter(dataset))


def _prepare_offline_cache_dataset(
    *,
    cache_dir: str | None,
    map_location: str,
    paths: PathConfig,
    state: Any,
    stage: Stage | None,
    include_efm_snippet: bool,
    include_gt_mesh: bool,
) -> OracleRriCacheDataset | None:
    if cache_dir is None:
        return None
    split = "all"
    if stage is Stage.TRAIN:
        split = "train"
    elif stage in (Stage.VAL, Stage.TEST):
        split = "val"
    cache_cfg = OracleRriCacheDatasetConfig(
        cache=OracleRriCacheConfig(cache_dir=Path(cache_dir), paths=paths),
        load_backbone=True,
        map_location=map_location,
        split=split,
        include_efm_snippet=include_efm_snippet,
        include_gt_mesh=include_gt_mesh,
    )
    cfg_sig = config_signature(cache_cfg)
    if state.offline_cache_sig != cfg_sig or state.offline_cache is None:
        cache_ds = cache_cfg.setup_target()
        state.offline_cache_sig = cfg_sig
        state.offline_cache = cache_ds
        state.offline_cache_len = len(cache_ds)
        state.offline_cache_idx = 0
        state.offline_snippet_key = None
        state.offline_snippet = None
        state.offline_snippet_error = None
    return state.offline_cache


def _collect_offline_cache_stats(
    *,
    toml_path: str | None,
    stage: Stage,
    cache_dir: str | None,
    map_location: str,
    max_samples: int | None,
    num_workers: int | None,
    train_val_split: float,
) -> dict[str, Any]:
    """Collect summary stats from an offline cache without keeping full samples in memory."""

    def _normalise(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)

    def _roll_about_forward(
        *,
        forward: torch.Tensor,
        up_cam: torch.Tensor,
        up_ref: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        forward = _normalise(forward, eps=eps)
        up_cam = _normalise(up_cam, eps=eps)
        if up_ref.ndim == 1:
            up_ref = up_ref.view(1, 3).expand_as(forward)
        else:
            while up_ref.ndim < forward.ndim:
                up_ref = up_ref.unsqueeze(0)
            up_ref = up_ref.expand_as(forward)
        up_ref = _normalise(up_ref, eps=eps)

        left0 = torch.cross(up_ref, forward, dim=-1)
        left0_norm = left0.norm(dim=-1, keepdim=True)
        degenerate = left0_norm.squeeze(-1) < eps
        if degenerate.any():
            alt = torch.tensor(
                [1.0, 0.0, 0.0],
                device=forward.device,
                dtype=forward.dtype,
            )
            alt = alt.view(1, 3).expand_as(forward)
            alt = alt - (alt * forward).sum(dim=-1, keepdim=True) * forward
            alt_norm = alt.norm(dim=-1, keepdim=True)
            second = alt_norm.squeeze(-1) < eps
            if second.any():
                alt2 = torch.tensor(
                    [0.0, 1.0, 0.0],
                    device=forward.device,
                    dtype=forward.dtype,
                )
                alt2 = alt2.view(1, 3).expand_as(forward)
                alt2 = alt2 - (alt2 * forward).sum(dim=-1, keepdim=True) * forward
                alt[second] = alt2[second]
                alt_norm = alt.norm(dim=-1, keepdim=True)
            left0[degenerate] = alt[degenerate]
            left0_norm = left0.norm(dim=-1, keepdim=True)
        left0 = left0 / left0_norm.clamp_min(eps)
        up0 = _normalise(torch.cross(forward, left0, dim=-1), eps=eps)
        sin_term = (forward * torch.cross(up0, up_cam, dim=-1)).sum(dim=-1)
        cos_term = (up0 * up_cam).sum(dim=-1)
        return torch.atan2(sin_term, cos_term)

    def _as_tensor(value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            return value
        tensor_attr = getattr(value, "tensor", None)
        if callable(tensor_attr):
            try:
                return tensor_attr()
            except Exception:
                return None
        if torch.is_tensor(tensor_attr):
            return tensor_attr
        return None

    def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
        if tensor.numel() == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "abs_mean": float("nan"),
                "nz_frac": 0.0,
            }
        vals = tensor.detach()
        vals = vals[torch.isfinite(vals)]
        if vals.numel() == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "abs_mean": float("nan"),
                "nz_frac": 0.0,
            }
        vals = vals.to(dtype=torch.float32)
        return {
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()),
            "abs_mean": float(vals.abs().mean().item()),
            "nz_frac": float((vals.abs() > 1e-6).float().mean().item()),
        }

    cfg = AriaNBVExperimentConfig.from_toml(Path(toml_path)) if toml_path else AriaNBVExperimentConfig()
    cfg.run_mode = "summarize_vin"
    cfg.stage = stage
    cfg.trainer_config.use_wandb = False

    dm_cfg = cfg.datamodule_config
    dm_cfg.train_cache_new_samples_per_epoch = 0
    dm_cfg.val_cache_new_samples_per_epoch = 0

    paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
    cache_root = cache_dir or str(
        paths.offline_cache_dir or (paths.data_root / "oracle_rri_cache"),
    )
    cache_cfg = OracleRriCacheDatasetConfig(
        cache=OracleRriCacheConfig(cache_dir=Path(cache_root), paths=paths),
        load_backbone=True,
        map_location=map_location,
        train_val_split=train_val_split,
    )
    dm_cfg.train_cache = cache_cfg.model_copy(deep=True, update={"split": "train"})
    dm_cfg.val_cache = cache_cfg.model_copy(deep=True, update={"split": "val"})
    dm_cfg.use_train_as_val = False

    if num_workers is not None and num_workers > 0:
        dm_cfg.num_workers = int(num_workers)

    datamodule = dm_cfg.setup_target()
    dataloader = datamodule.train_dataloader() if stage is Stage.TRAIN else datamodule.val_dataloader()

    total_batches = None
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None

    sample_rows: list[dict[str, Any]] = []
    backbone_rows: list[dict[str, Any]] = []
    rri_values: list[float] = []
    pm_comp_after_values: list[float] = []
    pm_acc_after_values: list[float] = []
    num_valid_values: list[int] = []
    candidate_offsets: list[np.ndarray] = []
    candidate_yaw: list[np.ndarray] = []
    candidate_pitch: list[np.ndarray] = []
    candidate_roll: list[np.ndarray] = []
    candidate_rot_deg: list[np.ndarray] = []

    max_batches = None if max_samples in (None, 0) else int(max_samples)
    progress = st.progress(0.0)
    for idx, batch in enumerate(dataloader):
        if max_batches is not None and idx >= max_batches:
            break

        rri = batch.rri.detach().flatten()
        rri_mask = torch.isfinite(rri)
        rri_valid = rri[rri_mask]
        num_valid = int(rri_valid.numel())
        num_valid_values.append(num_valid)
        if num_valid:
            rri_values.extend(rri_valid.cpu().tolist())

        pm_comp_after = batch.pm_comp_after.detach().flatten()
        pm_acc_after = batch.pm_acc_after.detach().flatten()
        pm_comp_valid = pm_comp_after[torch.isfinite(pm_comp_after)]
        pm_acc_valid = pm_acc_after[torch.isfinite(pm_acc_after)]
        if pm_comp_valid.numel():
            pm_comp_after_values.extend(pm_comp_valid.cpu().tolist())
        if pm_acc_valid.numel():
            pm_acc_after_values.extend(pm_acc_valid.cpu().tolist())

        def _finite_stats(vals: torch.Tensor) -> tuple[float, float, float]:
            vals = vals[torch.isfinite(vals)]
            if vals.numel() == 0:
                return float("nan"), float("nan"), float("nan")
            return (
                float(vals.mean().item()),
                float(vals.min().item()),
                float(vals.max().item()),
            )

        rri_mean, rri_min, rri_max = _finite_stats(rri)
        pm_comp_mean, _, _ = _finite_stats(pm_comp_after)
        pm_acc_mean, _, _ = _finite_stats(pm_acc_after)
        sample_rows.append(
            {
                "scene_id": batch.scene_id,
                "snippet_id": batch.snippet_id,
                "num_valid": num_valid,
                "rri_mean": rri_mean,
                "rri_min": rri_min,
                "rri_max": rri_max,
                "pm_comp_after_mean": pm_comp_mean,
                "pm_acc_after_mean": pm_acc_mean,
            },
        )

        backbone_out = batch.backbone_out
        if backbone_out is not None:
            if is_dataclass(backbone_out):
                items = [(field.name, getattr(backbone_out, field.name)) for field in fields(backbone_out)]
            else:
                items = list(vars(backbone_out).items())
            for name, value in items:
                tensor = _as_tensor(value)
                if tensor is None:
                    continue
                stats = _tensor_stats(tensor)
                backbone_rows.append(
                    {
                        "scene_id": batch.scene_id,
                        "snippet_id": batch.snippet_id,
                        "field": name,
                        "shape": str(tuple(tensor.shape)),
                        "numel": int(tensor.numel()),
                        **stats,
                    },
                )

        poses_world_cam = batch.candidate_poses_world_cam
        ref_pose = batch.reference_pose_world_rig
        if poses_world_cam is not None and ref_pose is not None:
            r_wc = poses_world_cam.R
            t_wc = poses_world_cam.t
            r_wr = ref_pose.R
            t_wr = ref_pose.t
            if r_wr.ndim == 2:
                r_wr = r_wr.unsqueeze(0)
                t_wr = t_wr.unsqueeze(0)
            if r_wr.shape[0] == 1 and r_wc.shape[0] > 1:
                r_wr = r_wr.expand(r_wc.shape[0], -1, -1)
                t_wr = t_wr.expand(r_wc.shape[0], -1)

            r_rw = r_wr.transpose(-1, -2)
            t_rw = -(r_rw @ t_wr.unsqueeze(-1)).squeeze(-1)
            r_rc = r_rw @ r_wc
            t_rc = t_rw + (r_rw @ t_wc.unsqueeze(-1)).squeeze(-1)

            candidate_offsets.append(t_rc.detach().cpu().numpy())

            fwd = r_rc[:, :, 2]
            up = r_rc[:, :, 1]
            yaw = torch.atan2(fwd[:, 0], fwd[:, 2])
            pitch = torch.asin(_normalise(fwd)[:, 1].clamp(-1.0, 1.0))
            up_ref = torch.tensor(
                [0.0, 1.0, 0.0],
                device=fwd.device,
                dtype=fwd.dtype,
            )
            roll = _roll_about_forward(
                forward=fwd,
                up_cam=up,
                up_ref=up_ref,
            )
            yaw_deg = torch.rad2deg(yaw).detach().cpu().numpy()
            pitch_deg = torch.rad2deg(pitch).detach().cpu().numpy()
            roll_deg = torch.rad2deg(roll).detach().cpu().numpy()
            candidate_yaw.append(yaw_deg)
            candidate_pitch.append(pitch_deg)
            candidate_roll.append(roll_deg)

            trace = r_rc[:, 0, 0] + r_rc[:, 1, 1] + r_rc[:, 2, 2]
            cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
            rot_angle = torch.acos(cos_angle)
            candidate_rot_deg.append(
                torch.rad2deg(rot_angle).detach().cpu().numpy(),
            )

        if total_batches:
            progress.progress(min(1.0, float(idx + 1) / float(total_batches)))
        else:
            progress.progress(0.0)

    progress.empty()

    sample_df = pd.DataFrame(sample_rows)
    backbone_df = pd.DataFrame(backbone_rows)
    summary = {
        "samples": len(sample_rows),
        "total_candidates": int(sum(num_valid_values)),
        "rri_mean": float(np.mean(rri_values)) if rri_values else float("nan"),
        "rri_median": float(np.median(rri_values)) if rri_values else float("nan"),
        "pm_comp_after_mean": float(np.mean(pm_comp_after_values)) if pm_comp_after_values else float("nan"),
        "pm_acc_after_mean": float(np.mean(pm_acc_after_values)) if pm_acc_after_values else float("nan"),
    }
    return {
        "summary": summary,
        "sample_df": sample_df,
        "backbone_df": backbone_df,
        "rri_values": rri_values,
        "pm_comp_after_values": pm_comp_after_values,
        "pm_acc_after_values": pm_acc_after_values,
        "num_valid_values": num_valid_values,
        "candidate_offsets": np.concatenate(candidate_offsets, axis=0)
        if candidate_offsets
        else np.zeros((0, 3), dtype=np.float32),
        "candidate_yaw": np.concatenate(candidate_yaw, axis=0) if candidate_yaw else np.zeros((0,), dtype=np.float32),
        "candidate_pitch": np.concatenate(candidate_pitch, axis=0)
        if candidate_pitch
        else np.zeros((0,), dtype=np.float32),
        "candidate_roll": np.concatenate(candidate_roll, axis=0)
        if candidate_roll
        else np.zeros((0,), dtype=np.float32),
        "candidate_rot_deg": np.concatenate(candidate_rot_deg, axis=0)
        if candidate_rot_deg
        else np.zeros((0,), dtype=np.float32),
    }


def pose_world_cam(
    sample: EfmSnippetView,
    cam_view: EfmCameraView,
    frame_idx: int,
) -> tuple[PoseTW, object]:
    cam_ts = cam_view.time_ns.cpu().numpy()
    traj_ts = sample.trajectory.time_ns.cpu().numpy()
    traj_idx = int(np.argmin(np.abs(traj_ts - cam_ts[frame_idx])))

    t_world_rig = sample.trajectory.t_world_rig[traj_idx]
    t_cam_rig = cam_view.calib.T_camera_rig[frame_idx]
    return t_world_rig @ t_cam_rig.inverse(), cam_view.calib[frame_idx]


def semidense_points_for_frame(
    sample: EfmSnippetView,
    frame_idx: int | None,
    *,
    all_frames: bool,
) -> torch.Tensor:
    sem = sample.semidense
    if sem is None or sem.points_world.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.float32)

    pts = sem.points_world
    lengths = sem.lengths
    if all_frames:
        if lengths is not None:
            max_len = pts.shape[1]
            mask_valid = torch.arange(max_len, device=pts.device).unsqueeze(
                0,
            ) < lengths.clamp_max(max_len).unsqueeze(
                -1,
            )
            pts = torch.where(mask_valid.unsqueeze(-1), pts, torch.nan)
        pts = pts.reshape(-1, 3)
    else:
        if frame_idx is None:
            frame_idx = int(torch.argmax(lengths).item()) if lengths.numel() else 0
        frame_idx = max(0, min(int(frame_idx), pts.shape[0] - 1))
        n_valid = int(lengths[frame_idx].item()) if lengths is not None else pts.shape[1]
        pts = pts[frame_idx, :n_valid]

    finite = torch.isfinite(pts).all(dim=-1)
    return pts[finite]


def scene_plot_options_ui(
    sample: EfmSnippetView,
    *,
    key_prefix: str = "data_scene",
) -> tuple[str, Scene3DPlotOptions]:
    """Render UI controls for the data-page 3D plot.

    Args:
        sample: Current snippet sample used to derive available frame indices and GT timestamps.
        key_prefix: Widget key prefix to avoid collisions across pages.

    Returns:
        Tuple of ``(frustum_camera, options)``.
    """
    st.subheader("3D scene view")

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        frustum_camera = st.radio(
            "Frustum camera",
            options=["rgb", "slam-l", "slam-r"],
            horizontal=True,
            index=0,
            key=f"{key_prefix}_frustum_cam",
        )
        num_frames = int(sample.camera_rgb.images.shape[0])
        frustum_frame_indices = st.multiselect(
            "Frustum frame indices",
            options=list(range(num_frames)),
            default=[0],
            key=f"{key_prefix}_frustum_idx",
        )
        frustum_scale = st.slider(
            "Frustum scale",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.05,
            key=f"{key_prefix}_frustum_scale",
        )

    with opt_col2:
        semidense_mode = st.radio(
            "Semi-dense points",
            options=["off", "all frames", "last frame only"],
            horizontal=True,
            index=1,
            key=f"{key_prefix}_sem_mode",
        )
        max_sem_points = st.slider(
            "Max semi-dense points",
            min_value=1000,
            max_value=200000,
            value=20000,
            step=1000,
            key=f"{key_prefix}_max_sem_points",
        )
        show_scene_bounds = st.checkbox(
            "Show scene bounds",
            value=True,
            key=f"{key_prefix}_scene_bounds",
        )
        show_crop_bounds = st.checkbox(
            "Show crop bbox",
            value=True,
            key=f"{key_prefix}_crop_bounds",
        )
        show_frustum = st.checkbox(
            "Show frustum",
            value=True,
            key=f"{key_prefix}_frustum_enable",
        )
        mark_first_last = st.checkbox(
            "Mark start/finish",
            value=True,
            key=f"{key_prefix}_mark_first_last",
        )
        show_gt_obbs = st.checkbox(
            "Show GT OBBs",
            value=False,
            key=f"{key_prefix}_gt_obbs",
        )

    gt_ts = None
    if show_gt_obbs and sample.gt.timestamps:
        gt_ts = st.selectbox(
            "GT OBB timestamp",
            options=sample.gt.timestamps,
            index=0,
            key=f"{key_prefix}_gt_ts",
        )

    return (
        frustum_camera,
        Scene3DPlotOptions(
            show_scene_bounds=show_scene_bounds,
            show_crop_bounds=show_crop_bounds,
            show_frustum=show_frustum,
            frustum_frame_indices=[int(i) for i in frustum_frame_indices] if frustum_frame_indices else [0],
            frustum_scale=float(frustum_scale),
            mark_first_last=mark_first_last,
            show_gt_obbs=show_gt_obbs,
            gt_timestamp=gt_ts,
            semidense_mode=str(semidense_mode),
            max_sem_points=int(max_sem_points),
        ),
    )


def rejected_pose_tensor(candidates: CandidateSamplingResult) -> torch.Tensor | None:
    mask_valid = candidates.mask_valid
    shell_poses = candidates.shell_poses
    if mask_valid is None or shell_poses is None or mask_valid.numel() == 0:
        return None
    shell_tensor = shell_poses.tensor() if hasattr(shell_poses, "tensor") else shell_poses
    if mask_valid.shape[0] != shell_tensor.shape[0]:
        return None
    rejected_mask = ~mask_valid
    if not rejected_mask.any():
        return None
    return shell_tensor[rejected_mask]


def render_data_page(
    sample: EfmSnippetView,
    *,
    crop_margin: float | None = None,
) -> None:
    st.header("Data")
    st.write(f"Scene: **{sample.scene_id}**, snippet: **{sample.snippet_id}**")

    first_pose = sample.trajectory.t_world_rig[0]
    last_pose = sample.trajectory.t_world_rig[-1]

    def _pose_summary(pose: torch.Tensor | PoseTW):
        pt = (
            pose
            if isinstance(pose, PoseTW)
            else PoseTW.from_matrix3x4(
                pose.view(3, 4) if pose.shape[-1] == 12 else pose,
            )
        )
        r, p, y = pt.to_ypr(rad=True)
        rpy = torch.rad2deg(torch.stack([r, p, y])).tolist()
        euler = torch.rad2deg(pt.to_euler(rad=True)).tolist()
        t = pt.t.tolist()
        return t, rpy, euler

    t_first, rpy_first, eul_first = _pose_summary(first_pose)
    t_last, rpy_last, eul_last = _pose_summary(last_pose)
    st.markdown(
        "- **First frame**: "
        f"t=({t_first[0]:.3f},{t_first[1]:.3f},{t_first[2]:.3f}) m, "
        f"RPY=({rpy_first[0]:.2f},{rpy_first[1]:.2f},{rpy_first[2]:.2f})°, "
        f"Euler ZYX=({eul_first[0]:.2f},{eul_first[1]:.2f},{eul_first[2]:.2f})°",
    )
    st.markdown(
        "- **Last frame**: "
        f"t=({t_last[0]:.3f},{t_last[1]:.3f},{t_last[2]:.3f}) m, "
        f"RPY=({rpy_last[0]:.2f},{rpy_last[1]:.2f},{rpy_last[2]:.2f})°, "
        f"Euler ZYX=({eul_last[0]:.2f},{eul_last[1]:.2f},{eul_last[2]:.2f})°",
    )

    modalities, missing = collect_frame_modalities(sample, include_depth=True)
    st.plotly_chart(plot_first_last_frames(sample), width="stretch")

    missing_depth = [m for m in missing if m.startswith("Depth")]
    if modalities and not missing_depth:
        st.caption("Depth maps available for rendering and overlays.")
    elif missing_depth:
        st.info(f"Depth maps missing for: {', '.join(missing_depth)}")

    st.subheader("Point cloud overlay")
    _info_popover(
        "overlay",
        "Projects world-space points into the selected camera image using the "
        "camera intrinsics and the per-frame pose. Use this to verify that "
        "the semidense SLAM points align with RGB/SLAM frames and to spot "
        "pose or calibration mismatches.",
    )
    overlay_cam = st.selectbox(
        "Overlay camera",
        ["rgb", "slam-l", "slam-r"],
        index=0,
        key="overlay_cam",
    )
    overlay_frame = st.slider(
        "Frame index for overlay",
        0,
        int(sample.camera_rgb.images.shape[0] - 1),
        0,
    )
    overlay_source = st.selectbox(
        "Point cloud source",
        [
            "Semidense (all frames)",
            "Semidense (selected frame)",
            "Semidense (last with points)",
        ],
        index=0,
    )

    if overlay_source == "Semidense (all frames)":
        points_world = semidense_points_for_frame(sample, None, all_frames=True)
    elif overlay_source == "Semidense (selected frame)":
        points_world = semidense_points_for_frame(
            sample,
            overlay_frame,
            all_frames=False,
        )
    else:
        lengths = sample.semidense.lengths if sample.semidense is not None else None
        last_idx = (
            int(torch.nonzero(lengths > 0, as_tuple=False).max().item())
            if lengths is not None and torch.any(lengths > 0)
            else 0
        )
        points_world = semidense_points_for_frame(sample, last_idx, all_frames=False)

    if points_world is None or points_world.numel() == 0:
        st.info("No points available for overlay with current selection.")
    else:
        cam_view = sample.get_camera(overlay_cam.replace("-", ""))
        pose_wc, cam_tw = pose_world_cam(sample, cam_view, overlay_frame)
        fig_overlay = project_pointcloud_on_frame(
            img=cam_view.images[overlay_frame],
            cam=cam_tw,
            pose_world_cam=pose_wc,
            points_world=points_world,
            title=_pretty_label(
                f"Overlay on {overlay_cam.upper()} frame {overlay_frame} ({points_world.shape[0]} pts)",
            ),
            max_points=20000,
        )
        st.plotly_chart(fig_overlay, width="stretch")

    cam_choice, plot_opts = scene_plot_options_ui(sample, key_prefix="data_scene")
    _info_popover(
        "scene overview",
        "3D scene view combines GT mesh, semidense points, and the trajectory. "
        "Frusta are drawn from camera intrinsics/extrinsics. Bounds boxes show "
        "scene extents and any crop applied during RRI computation.",
    )

    builder = (
        SnippetPlotBuilder.from_snippet(
            sample,
            title=_pretty_label("Mesh + semidense + trajectory + camera frustum"),
        )
        .add_mesh()
        .add_trajectory(mark_first_last=plot_opts.mark_first_last, show=True)
    )

    if plot_opts.semidense_mode != "off":
        builder.add_semidense(
            max_points=plot_opts.max_sem_points,
            last_frame_only=(plot_opts.semidense_mode == "last frame only"),
        )

    if plot_opts.show_frustum:
        builder.add_frusta(
            camera=cam_choice,
            frame_indices=plot_opts.frustum_frame_indices,
            scale=plot_opts.frustum_scale,
            include_axes=True,
            include_center=True,
        )

    if plot_opts.show_scene_bounds:
        builder.add_bounds_box(name="Scene bounds", color="gray", dash="dash", width=2)

    if plot_opts.show_gt_obbs:
        builder.add_gt_obbs(camera=cam_choice, timestamp=plot_opts.gt_timestamp)

    if plot_opts.show_crop_bounds:
        crop_aabb = tuple(b.detach().cpu().numpy() for b in sample.crop_bounds)
        builder.add_bounds_box(
            name="Crop bounds",
            color="orange",
            dash="solid",
            width=3,
            aabb=crop_aabb,
        )

    st.plotly_chart(builder.finalize(), width="stretch")


def render_candidates_page(
    sample: EfmSnippetView,
    candidates: CandidateSamplingResult,
    cand_cfg: CandidateViewGeneratorConfig,
) -> None:
    shell_poses = candidates.shell_poses
    mask_valid = candidates.mask_valid

    st.header("Candidate Poses")

    tab_pos, tab_frusta = st.tabs(["Positions (3D)", "Frusta (3D)"])

    with tab_pos:
        _info_popover(
            "candidate positions",
            "Candidate centers are sampled around the reference pose according "
            "to the sampling shell (radius, azimuth, elevation). Only valid "
            "candidates (after rule filtering) are shown in blue. The reference "
            "axes are in the rig frame for interpretability.",
        )
        cand_fig = (
            CandidatePlotBuilder.from_candidates(
                sample,
                candidates,
                title=_pretty_label(f"Candidate positions ({cand_cfg.camera_label})"),
            )
            .add_mesh()
            .add_candidate_cloud(use_valid=True, color="royalblue", size=4, opacity=0.7)
            .add_reference_axes(display_rotate=False)
        ).finalize()
        st.plotly_chart(cand_fig, width="stretch")

    offsets_ref, dirs_ref = candidates.get_offsets_and_dirs_ref(display_rotate=False)

    with tab_frusta:
        if mask_valid is None or mask_valid.sum() == 0:
            st.warning("All candidates were rejected; frustum plot omitted.")
        else:
            _info_popover(
                "candidate frusta",
                "Frusta visualize the candidate camera extrinsics with the "
                "selected camera intrinsics. Scale is a display-only factor "
                "and does not change rendering or scoring.",
            )
            opt_col1, opt_col2 = st.columns(2)
            with opt_col1:
                frustum_scale = st.slider(
                    "Frustum scale",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="cand_frustum_scale",
                )
            with opt_col2:
                max_frustums = st.slider(
                    "Max frustums",
                    min_value=1,
                    max_value=24,
                    value=6,
                    step=1,
                    key="cand_max_frustums",
                )

            frust_fig = (
                CandidatePlotBuilder.from_candidates(
                    sample,
                    candidates,
                    title=_pretty_label(f"Candidate frusta ({cand_cfg.camera_label})"),
                )
                .add_mesh()
                .add_candidate_cloud(
                    use_valid=True,
                    color="royalblue",
                    size=3,
                    opacity=0.35,
                )
                .add_candidate_frusta(
                    scale=float(frustum_scale),
                    color="crimson",
                    name="Frustum",
                    max_frustums=int(max_frustums),
                    include_axes=False,
                    include_center=False,
                    display_rotate=False,
                )
                .add_reference_axes(display_rotate=False)
            ).finalize()
            st.plotly_chart(frust_fig, width="stretch")

    with st.expander("Distributions & Diagnostics", expanded=False):
        _info_popover(
            "candidate diagnostics",
            "Offsets and directions are expressed in the reference rig frame. "
            "These plots diagnose sampling coverage, symmetry, and any bias "
            "introduced by constraints or collision rules.",
        )
        fixed_ranges = st.checkbox(
            "Clamp axes to standard ranges",
            value=True,
            key="cand_angles_fixed_ranges",
        )
        diag_offsets, diag_dirs, diag_rules, diag_rejected = st.tabs(
            ["Offsets", "Directions", "Rules", "Rejected"],
        )

        with diag_offsets:
            _info_popover(
                "offsets",
                "Offsets are candidate translations relative to the reference pose. "
                "Polar plots show azimuth/elevation of the offset direction; the "
                "radius histogram shows the sampled distance distribution.",
            )
            st.markdown(
                stats_to_markdown_table(
                    summarise_offsets_ref(offsets_ref),
                    header=None,
                ),
            )
            offsets_np = offsets_ref.cpu().numpy()
            colp1, colp2 = st.columns(2)
            with colp1:
                st.plotly_chart(
                    plot_position_polar(
                        offsets_np,
                        title=_pretty_label("Offsets from reference pose (az/elev)"),
                        fixed_ranges=fixed_ranges,
                    ),
                    width="stretch",
                )
            with colp2:
                st.plotly_chart(
                    plot_position_sphere(offsets_np, show_axes=True),
                    width="stretch",
                )
            st.plotly_chart(plot_radius_hist(offsets_np), width="stretch")

        with diag_dirs:
            _info_popover(
                "directions",
                "View directions are unit forward vectors in the reference frame. "
                "Marginals expose angular coverage; sphere plots reveal anisotropy. "
                "Euler plots are shown in both world and reference frames to "
                "highlight frame-dependent interpretations.",
            )
            st.markdown(
                stats_to_markdown_table(summarise_dirs_ref(dirs_ref), header=None),
            )
            dirs_np = dirs_ref.cpu().numpy()

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_direction_polar(
                        dirs_np,
                        title=_pretty_label("View directions (reference frame)"),
                        fixed_ranges=fixed_ranges,
                    ),
                    width="stretch",
                )
            with col2:
                st.plotly_chart(
                    plot_direction_sphere(
                        dirs_np,
                        title=_pretty_label("View dirs on unit sphere"),
                        show_axes=True,
                    ),
                    width="stretch",
                )
            st.plotly_chart(
                plot_direction_marginals(
                    torch.as_tensor(dirs_np),
                    fixed_ranges=fixed_ranges,
                ),
                width="stretch",
            )

            r_wr = candidates.reference_pose.R
            if r_wr.ndim == 3:
                r_wr = r_wr[0]
            fwd_w = r_wr[:, 2].view(1, 3)
            up_w = r_wr[:, 1].view(1, 3)
            wup = world_up_tensor(device=fwd_w.device, dtype=fwd_w.dtype).view(1, 3)
            left0 = torch.cross(wup, fwd_w, dim=-1)
            left0_norm = left0.norm(dim=-1, keepdim=True)
            if float(left0_norm.item()) < 1e-6:
                alt = torch.tensor(
                    [1.0, 0.0, 0.0],
                    device=fwd_w.device,
                    dtype=fwd_w.dtype,
                ).view(1, 3)
                left0 = torch.cross(alt, fwd_w, dim=-1)
                left0_norm = left0.norm(dim=-1, keepdim=True)
                if float(left0_norm.item()) < 1e-6:
                    alt2 = torch.tensor(
                        [0.0, 1.0, 0.0],
                        device=fwd_w.device,
                        dtype=fwd_w.dtype,
                    ).view(1, 3)
                    left0 = torch.cross(alt2, fwd_w, dim=-1)
                    left0_norm = left0.norm(dim=-1, keepdim=True)
            left0 = left0 / left0_norm.clamp_min(1e-6)
            up0 = torch.cross(fwd_w, left0, dim=-1)
            sin_roll = (fwd_w * torch.cross(up0, up_w, dim=-1)).sum(dim=-1)
            cos_roll = (up0 * up_w).sum(dim=-1)
            roll = torch.rad2deg(torch.atan2(sin_roll, cos_roll)).item()
            yaw = torch.rad2deg(torch.atan2(fwd_w[:, 0], fwd_w[:, 1])).item()
            pitch = torch.rad2deg(torch.asin(fwd_w[:, 2].clamp(-1.0, 1.0))).item()
            st.markdown(
                f"Reference pose view angles (world): yaw={yaw:.2f}°, pitch={pitch:.2f}°, roll={roll:.2f}°",
            )
            st.plotly_chart(
                plot_euler_world(candidates, fixed_ranges=fixed_ranges),
                width="stretch",
            )
            st.plotly_chart(
                plot_euler_reference(candidates, fixed_ranges=fixed_ranges),
                width="stretch",
            )
            delta = candidates.extras.get("view_dirs_delta") if hasattr(candidates, "extras") else None
            if delta is not None:
                r_delta = delta.R
                fwd = r_delta[:, :, 2]
                yaw_d = torch.rad2deg(torch.atan2(fwd[:, 0], fwd[:, 2])).cpu()
                pitch_d = torch.rad2deg(torch.asin(fwd[:, 1].clamp(-1.0, 1.0))).cpu()
                roll_d = torch.rad2deg(
                    torch.atan2(r_delta[:, 1, 0], r_delta[:, 1, 1]),
                ).cpu()
                st.markdown(
                    stats_to_markdown_table(
                        {
                            "yaw_delta_deg": {
                                "min": float(yaw_d.min()),
                                "max": float(yaw_d.max()),
                                "mean": float(yaw_d.mean()),
                                "std": float(yaw_d.std(unbiased=False)),
                            },
                            "pitch_delta_deg": {
                                "min": float(pitch_d.min()),
                                "max": float(pitch_d.max()),
                                "mean": float(pitch_d.mean()),
                                "std": float(pitch_d.std(unbiased=False)),
                            },
                            "roll_delta_deg": {
                                "min": float(roll_d.min()),
                                "max": float(roll_d.max()),
                                "mean": float(roll_d.mean()),
                                "std": float(roll_d.std(unbiased=False)),
                            },
                        },
                        header="Orientation jitter stats (delta, LUF yaw/pitch/roll)",
                    ),
                )
                st.plotly_chart(
                    _euler_histogram(
                        yaw_d,
                        pitch_d,
                        roll_d,
                        bins=90,
                        title=_pretty_label("Orientation jitter (delta, deg)"),
                        fixed_ranges=fixed_ranges,
                    ),
                    width="stretch",
                )

        with diag_rules:
            _info_popover(
                "rules",
                "Rule masks mark which candidates were rejected by each constraint "
                "(collision, distance-to-mesh, visibility, etc.). The bar plot "
                "summarizes rejection counts per rule to reveal dominant filters.",
            )
            masks = candidates.masks
            if isinstance(masks, dict) and len(masks) > 0 and shell_poses is not None:
                masks_tensor = torch.stack(list(masks.values()))
                mask_fig = plot_rule_masks(
                    snippet=sample,
                    shell_poses=shell_poses.tensor() if hasattr(shell_poses, "tensor") else shell_poses,
                    masks=masks_tensor,
                    rule_names=list(masks.keys()),
                )
                st.plotly_chart(mask_fig, width="stretch")

            extras = candidates.extras if hasattr(candidates, "extras") else {}
            dist_min = extras.get("min_distance_to_mesh")
            path_collide = extras.get("path_collision_mask")

            if dist_min is not None:
                st.plotly_chart(
                    plot_min_distance_to_mesh(
                        snippet=sample,
                        candidates=candidates,
                        distances=dist_min,
                    ),
                    width="stretch",
                )
            if path_collide is not None:
                st.plotly_chart(
                    plot_path_collision_segments(
                        snippet=sample,
                        candidates=candidates,
                        collision_mask=path_collide,
                    ),
                    width="stretch",
                )

            st.plotly_chart(plot_rule_rejection_bar(candidates), width="stretch")

        with diag_rejected:
            _info_popover(
                "rejected",
                "Rejected poses are candidates that failed at least one rule. "
                "Plotting them can reveal systematic failures (e.g., walls, "
                "occlusions, or sampling bias).",
            )
            plot_rejected_only = st.checkbox(
                "Plot rejected poses only (if any)",
                value=False,
                key="cand_plot_rejected_only",
            )
            if plot_rejected_only:
                rejected_poses = rejected_pose_tensor(candidates)
                if rejected_poses is None:
                    st.info(
                        "No rejected poses to plot; all sampled candidates survived rule filtering.",
                    )
                else:
                    rej_fig = (
                        CandidatePlotBuilder.from_candidates(
                            sample,
                            candidates,
                            title=_pretty_label(
                                f"Rejected candidate positions ({rejected_poses.shape[0]})",
                            ),
                        )
                        .add_mesh()
                        .add_rejected_cloud()
                        .add_reference_axes()
                    ).finalize()
                    st.plotly_chart(rej_fig, width="stretch")


def render_depth_page(
    sample: EfmSnippetView | None,
    depth_batch: CandidateDepths,
    *,
    pcs: CandidatePointClouds | None,
) -> None:
    st.header("Candidate Renders")

    depths = depth_batch.depths
    indices = depth_batch.candidate_indices.tolist()
    titles = [f"cand {i} (id {cid})" for i, cid in enumerate(indices)]
    st.caption(
        "Local indices (cand 0..N-1) refer to the rendered batch order; "
        "`id` is the original candidate index (pre-render filtering).",
    )

    cam = depth_batch.camera
    if hasattr(cam, "valid_radius") and cam.valid_radius.numel() > 0:
        zfar_stat = float(cam.valid_radius.max().item())
    else:
        zfar_stat = float(depths.max().item()) * 1.05

    st.subheader("Depth grid")
    _info_popover(
        "depth grid",
        "Each tile is a rendered depth map for a candidate pose. Depth is in "
        "camera coordinates with +Z forward. Invalid hits are masked in the "
        "renderer and can appear at the far plane if not filtered.",
    )
    fig = depth_grid(depths, titles=titles, zmax=float(depths.max().item()))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Diagnostics", expanded=False):
        tab_hist, tab_hits = st.tabs(["Histograms", "Depth-hit point cloud (3D)"])

        with tab_hist:
            _info_popover(
                "depth hist",
                "Depth histograms summarize per-candidate depth distributions. "
                "A spike near the far plane often indicates many miss pixels; "
                "a bimodal shape can reveal multiple surfaces along the frustum.",
            )
            bins = st.slider(
                "Histogram bins",
                10,
                200,
                50,
                step=10,
                key="depth_hist_bins",
            )
            fig_hist = depth_histogram(depths, bins=int(bins), zfar=zfar_stat)
            st.plotly_chart(fig_hist, width="stretch")

        with tab_hits:
            _info_popover(
                "depth hits",
                "Back-projects valid depth pixels into world space using the "
                "candidate pose and camera intrinsics. The resulting points "
                "approximate the candidate view point cloud used for RRI.",
            )
            if sample is None:
                st.info("Load data first to back-project depth hits.")
                return
            if pcs is None:
                st.info(
                    "Run / refresh renders to compute backprojected CandidatePointClouds.",
                )
                return

            max_points = st.number_input(
                "Max points to display",
                min_value=1,
                max_value=200000,
                value=20000,
                step=1000,
                key="depth_hit_max_points",
            )

            cand_options = depth_batch.candidate_indices.tolist()
            selected_global = st.multiselect(
                "Select candidates to back-project",
                options=cand_options,
                default=cand_options,
                key="depth_hit_cands",
            )
            cand_to_local = {int(g): idx for idx, g in enumerate(depth_batch.candidate_indices.tolist())}
            selected = [cand_to_local[g] for g in selected_global if g in cand_to_local]
            num_frustums = int(depth_batch.poses.tensor().shape[0])

            points_selected = []
            for idx in selected:
                n_valid = int(pcs.lengths[idx].item())
                if n_valid == 0:
                    continue
                pts = pcs.points[idx, :n_valid]
                points_selected.append(pts)

            if points_selected:
                pts_cat = torch.cat(points_selected, dim=0)
                if pts_cat.shape[0] > max_points:
                    rand_idx = torch.randperm(pts_cat.shape[0], device=pts_cat.device)[: int(max_points)]
                    pts_cat = pts_cat[rand_idx]

                builder = (
                    RenderingPlotBuilder.from_snippet(
                        sample,
                        title=_pretty_label("Depth hit back-projection"),
                    )
                    .add_mesh()
                    .add_points(
                        pts_cat,
                        name="Depth hits",
                        color="teal",
                        size=3,
                        opacity=0.8,
                    )
                    .add_frusta_selection(
                        poses=depth_batch.poses,
                        camera=depth_batch.camera,
                        max_frustums=min(16, num_frustums),
                        candidate_indices=selected,
                    )
                )
                st.plotly_chart(builder.finalize(), width="stretch")
            else:
                st.info("No valid depth hits to display for the selected candidates.")


def render_rri_page(
    sample: EfmSnippetView,
    depth_batch: CandidateDepths,
    pcs: CandidatePointClouds,
    rri: RriResult,
) -> None:
    st.header("RRI Preview: Point Clouds vs Mesh")

    candidate_ids = depth_batch.candidate_indices.cpu().tolist()
    if len(candidate_ids) == 0:
        st.warning("No candidate renders available for RRI scoring.")
        return

    labels = [str(int(cid)) for cid in candidate_ids]
    baseline_label = "-1"

    qualitative = px.colors.qualitative.Plotly
    bar_color_map = {label: qualitative[i % len(qualitative)] for i, label in enumerate(labels)}

    _info_popover(
        "rri",
        "RRI measures relative improvement in mesh distance after adding the "
        "candidate point cloud: (d_before - d_after) / d_before. Higher is better. "
        "Scores are computed against the GT mesh with semidense points as baseline.",
    )
    st.plotly_chart(
        go.Figure(
            data=go.Bar(
                x=labels,
                y=rri.rri,
                marker_color=[bar_color_map[label] for label in labels],
            ),
            layout_title_text=_pretty_label("Oracle RRI per candidate"),
        ),
        width="stretch",
    )

    _info_popover(
        "pm dist",
        "Bidirectional point-mesh distance (Chamfer-style). "
        "Before uses semidense points only; after includes the candidate "
        "point cloud. Lower is better.",
    )
    baseline_pm_dist = float(rri.pm_dist_before[0].item())
    fig_pm_dist = go.Figure(
        data=[
            go.Bar(
                x=[baseline_label],
                y=[baseline_pm_dist],
                name="before (semi-dense, -1)",
                marker_color="lightgray",
            ),
            go.Bar(
                x=labels,
                y=rri.pm_dist_after,
                name="after",
                marker_color=[bar_color_map[label] for label in labels],
            ),
        ],
    )
    fig_pm_dist.update_layout(
        title_text=_pretty_label("Chamfer-like (bidirectional)"),
        xaxis={"categoryorder": "array", "categoryarray": [baseline_label, *labels]},
    )
    st.plotly_chart(fig_pm_dist, width="stretch")

    _info_popover(
        "pm acc",
        "Point-to-mesh accuracy: distance from reconstruction points to GT mesh. "
        "Captures how well points lie on the surface. Lower is better.",
    )
    baseline_pm_acc = float(rri.pm_acc_before[0].item())
    fig_pm_acc = go.Figure(
        data=[
            go.Bar(
                x=[baseline_label],
                y=[baseline_pm_acc],
                name="point→mesh (before, -1)",
                marker_color="lightgray",
            ),
            go.Bar(
                x=labels,
                y=rri.pm_acc_after,
                name="point→mesh (after)",
                marker_color=[bar_color_map[label] for label in labels],
            ),
        ],
    )
    fig_pm_acc.update_layout(
        title_text=_pretty_label("Point→Mesh (accuracy)"),
        xaxis={"categoryorder": "array", "categoryarray": [baseline_label, *labels]},
    )
    st.plotly_chart(fig_pm_acc, width="stretch")

    _info_popover(
        "pm comp",
        "Mesh-to-point completeness: distance from GT mesh to reconstruction points. "
        "Captures coverage of the surface. Lower is better.",
    )
    baseline_pm_comp = float(rri.pm_comp_before[0].item())
    fig_pm_comp = go.Figure(
        data=[
            go.Bar(
                x=[baseline_label],
                y=[baseline_pm_comp],
                name="mesh→point (before, -1)",
                marker_color="lightgray",
            ),
            go.Bar(
                x=labels,
                y=rri.pm_comp_after,
                name="mesh→point (after)",
                marker_color=[bar_color_map[label] for label in labels],
            ),
        ],
    )
    fig_pm_comp.update_layout(
        title_text=_pretty_label("Mesh→Point (completeness)"),
        xaxis={"categoryorder": "array", "categoryarray": [baseline_label, *labels]},
    )
    st.plotly_chart(fig_pm_comp, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        default_selection = candidate_ids[: min(6, len(candidate_ids))]
        selected_ids = st.multiselect(
            "Candidates to display",
            options=candidate_ids,
            default=default_selection,
            key="rri_cands",
        )
        cid_to_local = {int(cid): idx for idx, cid in enumerate(candidate_ids)}
        selected_local = [cid_to_local[cid] for cid in selected_ids if cid in cid_to_local]
    with col2:
        show_frusta = st.checkbox("Show frusta", value=True, key="rri_show_frusta")

    max_sem_pts = st.number_input(
        "Max semi-dense points",
        min_value=1000,
        max_value=200000,
        value=50000,
        step=1000,
        key="rri_max_sem_pts",
    )

    builder = (
        RenderingPlotBuilder.from_snippet(
            sample,
            title=_pretty_label("Mesh + Semi-dense + Candidate PCs"),
        )
        .add_mesh()
        .add_semidense(last_frame_only=False, max_points=max_sem_pts)
    )
    _info_popover(
        "rri scene",
        "3D overlay of GT mesh, semidense points, and selected candidate "
        "point clouds. This helps validate that high-RRI candidates add "
        "new surface coverage rather than noisy points.",
    )
    if show_frusta:
        builder.add_frusta_selection(
            poses=depth_batch.poses,
            camera=depth_batch.camera,
            max_frustums=min(16, len(selected_local)),
            candidate_indices=selected_local,
        )

    for idx_i, cid_int in enumerate(candidate_ids):
        if cid_int not in selected_ids:
            continue
        pts = pcs.points[idx_i, : int(pcs.lengths[idx_i].item())]
        builder.add_points(
            pts,
            name=f"Candidate {cid_int}",
            color=bar_color_map.get(
                str(cid_int),
                px.colors.qualitative.Plotly[idx_i % len(px.colors.qualitative.Plotly)],
            ),
            size=3,
            opacity=0.7,
        )

    st.plotly_chart(builder.finalize(), width="stretch")


def render_vin_diagnostics_page() -> None:
    """Render VIN diagnostics using AriaNBVExperimentConfig (independent from app pipeline)."""
    st.header("VIN Diagnostics")
    st.caption(
        "Run VIN forward_with_debug on oracle batches and inspect internal tensors.",
    )

    state = get_vin_state()

    run = False
    use_offline_cache = False
    attach_snippet = True
    include_gt_mesh = False
    with st.sidebar.form("vin_diag_form"):
        st.subheader("VIN Diagnostics")
        toml_path = st.text_input("Experiment config TOML (optional)", value="")
        data_source = st.selectbox(
            "Data source",
            options=["online (oracle labeler)", "offline cache"],
            index=0,
        )
        use_offline_cache = data_source == "offline cache"
        cache_dir = None
        map_location = "cpu"
        if use_offline_cache:
            cache_dir = st.text_input(
                "Offline cache dir",
                value=str(PathConfig().offline_cache_dir),
            )
            map_location = st.selectbox(
                "Cache map_location",
                options=["cpu", "cuda"],
                index=0,
            )
            attach_snippet = st.checkbox(
                "Attach EFM snippet (geometry plots)",
                value=True,
            )
            if attach_snippet:
                include_gt_mesh = st.checkbox(
                    "Include GT mesh",
                    value=False,
                    key="vin_diag_include_mesh",
                )
        stage = st.selectbox(
            "Stage",
            options=[Stage.TRAIN, Stage.VAL, Stage.TEST],
            format_func=lambda s: s.value,
        )
        run = st.form_submit_button("Run / refresh VIN diagnostics")

    if st.sidebar.button("Clear VIN cache"):
        st.session_state.pop(VIN_DIAG_STATE_KEY, None)
        st.rerun()

    cache_ds = None
    if use_offline_cache:
        paths = PathConfig()
        try:
            cache_ds = _prepare_offline_cache_dataset(
                cache_dir=cache_dir,
                map_location=map_location,
                paths=paths,
                state=state,
                stage=stage,
                include_efm_snippet=attach_snippet,
                include_gt_mesh=include_gt_mesh,
            )
        except Exception as exc:  # pragma: no cover - UI guard
            trace = traceback.format_exc()
            print(trace, flush=True)
            state.offline_cache_len = 0
            state.offline_cache = None
            state.offline_cache_sig = None
            st.sidebar.error(f"{type(exc).__name__}: {exc}")
            with st.sidebar.expander("Full traceback", expanded=False):
                st.code(trace, language="text")
            cache_ds = None
        cache_len = int(state.offline_cache_len or 0)
        if cache_len > 0:
            advance = st.sidebar.button("Next cached sample")
            if advance:
                state.offline_cache_idx = (state.offline_cache_idx + 1) % cache_len
                st.session_state["vin_cache_index"] = state.offline_cache_idx
                run = True
            cache_idx = st.sidebar.number_input(
                "Cache index",
                min_value=0,
                max_value=max(0, cache_len - 1),
                value=int(state.offline_cache_idx),
                step=1,
                key="vin_cache_index",
            )
            state.offline_cache_idx = int(cache_idx)
            st.sidebar.caption(f"Cache samples: {cache_len}")
        else:
            st.sidebar.warning("Offline cache is empty or missing.")

    if run:
        try:
            cfg = _build_experiment_config(
                toml_path=toml_path.strip() or None,
                stage=stage,
                use_offline_cache=data_source == "offline cache",
                cache_dir=cache_dir,
                map_location=map_location,
                include_efm_snippet=attach_snippet,
                include_gt_mesh=include_gt_mesh,
            )
            cfg_sig = config_signature(cfg)

            if state.cfg_sig != cfg_sig or state.module is None or state.datamodule is None:
                trainer, module, datamodule = cfg.setup_target(setup_stage=stage)
                _ = trainer  # unused but kept for future diagnostics
                state.cfg_sig = cfg_sig
                state.experiment = cfg
                state.module = module
                state.datamodule = datamodule

            assert state.module is not None and state.datamodule is not None
            with st.spinner("Running oracle labeler + VIN forward..."):
                if use_offline_cache:
                    if cache_ds is None:
                        raise RuntimeError("Offline cache dataset is not available.")
                    cache_len = int(state.offline_cache_len or 0)
                    if cache_len == 0:
                        raise RuntimeError("Offline cache is empty.")
                    cache_idx = min(
                        max(int(state.offline_cache_idx), 0),
                        cache_len - 1,
                    )
                    cache_sample = cache_ds[cache_idx]
                    efm_snippet = cache_sample.efm_snippet_view
                    if attach_snippet:
                        snippet_key = f"{cache_sample.scene_id}:{cache_sample.snippet_id}"
                        if efm_snippet is None and (
                            state.offline_snippet_key != snippet_key or state.offline_snippet is None
                        ):
                            try:
                                paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
                                efm_snippet = _load_efm_snippet_for_cache(
                                    scene_id=cache_sample.scene_id,
                                    snippet_id=cache_sample.snippet_id,
                                    dataset_payload=cache_ds.metadata.dataset_config,
                                    device=map_location,
                                    paths=paths,
                                    include_gt_mesh=include_gt_mesh,
                                )
                                state.offline_snippet_key = snippet_key
                                state.offline_snippet = efm_snippet
                                state.offline_snippet_error = None
                            except Exception as exc:  # pragma: no cover - IO guard
                                state.offline_snippet_key = snippet_key
                                state.offline_snippet = None
                                state.offline_snippet_error = f"{type(exc).__name__}: {exc}"
                        if efm_snippet is None:
                            efm_snippet = state.offline_snippet
                        else:
                            state.offline_snippet_key = snippet_key
                            state.offline_snippet = efm_snippet
                            state.offline_snippet_error = None
                    else:
                        state.offline_snippet_error = None
                        state.offline_snippet = None
                        state.offline_snippet_key = None
                    batch = _vin_oracle_batch_from_cache(
                        cache_sample,
                        efm_snippet=efm_snippet,
                    )
                else:
                    batch = next(datamodule.iter_oracle_batches(stage=stage))

                pred, debug = _run_vin_debug(state.module, batch)

            state.batch = batch
            state.pred = pred
            state.debug = debug
            state.error = None
        except Exception:  # pragma: no cover - UI error guard
            trace = traceback.format_exc()
            print(trace, flush=True)
            state.error = trace

    if state.error:
        st.error("VIN diagnostics failed. See traceback below.")
        st.code(state.error, language="text")
        return

    if state.debug is None or state.pred is None or state.batch is None or state.experiment is None:
        st.info("Run the VIN diagnostics to load a batch.")
        return

    debug = state.debug
    pred = state.pred
    batch = state.batch
    cfg = state.experiment

    num_candidates = int(debug.candidate_valid.shape[-1])
    valid_mask = debug.candidate_valid.reshape(-1)
    valid_count = int(valid_mask.sum().item())
    has_tokens = hasattr(debug, "token_valid")
    if has_tokens:
        valid_frac = debug.token_valid.float().mean(dim=-1).reshape(-1)
        mean_valid_frac = f"{float(valid_frac.mean().item()):.3f}"
    else:
        valid_frac = debug.candidate_valid.float().reshape(-1)
        mean_valid_frac = "n/a"

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Candidates", num_candidates)
    col_b.metric("Valid candidates", f"{valid_count}/{num_candidates}")
    col_c.metric("Mean token valid frac", mean_valid_frac)

    snippet_label = (
        f"**Scene:** `{batch.scene_id}` &nbsp;&nbsp; "
        f"**Snippet:** `{batch.snippet_id}` &nbsp;&nbsp; "
        f"**Device:** `{debug.candidate_center_rig_m.device!s}`"
    )
    if use_offline_cache and state.offline_cache_len:
        snippet_label += f" &nbsp;&nbsp; **Cache idx:** `{state.offline_cache_idx}`"
    st.markdown(snippet_label)

    (
        tab_summary,
        tab_pose,
        tab_geometry,
        tab_field,
        tab_tokens,
        tab_evidence,
        tab_transforms,
        tab_concept,
        tab_coral,
    ) = st.tabs(
        [
            "Summary",
            "Pose Descriptor",
            "Geometry",
            "Field Slices",
            "Frustum Tokens",
            "Backbone Evidence",
            "Transforms",
            "FF Encodings",
            "CORAL / Ordinal",
        ],
    )

    with tab_summary:
        if batch.rri is not None:
            _info_popover(
                "scatter",
                "Each point is a candidate view. **X** is the oracle RRI computed from "
                "mesh distances (before vs after adding the candidate point cloud). "
                "**Y** is the VIN expected score from the CORAL ordinal head "
                "(mean of `P(y>k)` across bins, normalized to `[0,1]`). "
                "VIN v2 uses pose features from `[t, r6d]` in the reference rig "
                "frame plus global voxel context; VIN v1 may also use local "
                "frustum tokens.",
            )
            rri = _to_numpy(batch.rri.reshape(-1))
            expected = _to_numpy(pred.expected_normalized.reshape(-1))
            use_log_axes = st.checkbox(
                "Log scale axes",
                value=False,
                key="vin_summary_log_axes",
            )
            if use_log_axes:
                pos_mask = (rri > 0) & (expected > 0)
                if not np.all(pos_mask):
                    st.info(
                        "Log axes require positive values; non-positive points are omitted.",
                    )
                    rri = rri[pos_mask]
                    expected = expected[pos_mask]
            fig = px.scatter(
                x=rri,
                y=expected,
                labels={
                    "x": _pretty_label("Oracle RRI"),
                    "y": _pretty_label("VIN expected (normalized)"),
                },
                title=_pretty_label("Predicted score vs oracle RRI"),
                log_x=use_log_axes,
                log_y=use_log_axes,
            )
            st.plotly_chart(fig, width="stretch")

        feature_dims: list[tuple[str, int]] = []
        if hasattr(debug, "pose_enc"):
            feature_dims.append(("pose_enc", int(debug.pose_enc.shape[-1])))
        if getattr(debug, "global_feat", None) is not None:
            feature_dims.append(("global_feat", int(debug.global_feat.shape[-1])))
        if hasattr(debug, "local_feat"):
            feature_dims.append(("local_feat", int(debug.local_feat.shape[-1])))
        if feature_dims:
            _info_popover(
                "feature dims",
                "Feature blocks concatenated before the scorer MLP. "
                "VIN v2 uses `pose_enc` (LFF over translation + rotation-6D "
                "with learned scales) and `global_feat` (pose-conditioned "
                "attention pooling over the voxel field). "
                "VIN v1 can add `local_feat` from frustum sampling.",
            )
            dims_df = {
                "modality": [name for name, _ in feature_dims],
                "num_features": [count for _, count in feature_dims],
            }
            fig_dims = px.bar(
                dims_df,
                x="modality",
                y="num_features",
                title=_pretty_label("Feature dimensions by modality"),
            )
            st.plotly_chart(fig_dims, width="stretch")

        vin_model = state.module.vin if state.module is not None else None
        if vin_model is not None:
            params_df = _parameter_distribution(vin_model, trainable_only=True)
            if not params_df.empty:
                _info_popover(
                    "param counts",
                    "Trainable parameter counts grouped by top-level VIN submodule "
                    "(frozen backbone params are excluded). This highlights where "
                    "capacity is concentrated (pose encoder vs global pool vs head).",
                )
                fig_params = px.bar(
                    params_df,
                    x="module",
                    y="num_params",
                    title=_pretty_label("Trainable parameter counts by VIN module"),
                    labels={"num_params": _pretty_label("parameters")},
                )
                st.plotly_chart(fig_params, width="stretch")
                total_params = int(params_df["num_params"].sum())
                st.caption(f"Total trainable parameters: {total_params:,}")

        field_in = getattr(debug, "field_in", None)
        if isinstance(field_in, torch.Tensor) and field_in.ndim == 5:
            _info_popover(
                "field hists",
                "Per-channel scene-field distributions **before** projection. "
                "VIN v2 builds channels from EVL heads: `occ_pr` (occupancy prob), "
                "`cent_pr` (centerness), `occ_input` (occupied evidence), "
                "`counts_norm` (log1p-normalized coverage), `observed`=counts_norm, "
                "`unknown`=1-counts_norm, `free_input` (EVL free-space or derived), "
                "`new_surface_prior`=unknown*occ_pr. Values are mostly in `[0,1]` "
                "and plotted as |value|.",
            )
            channel_count = int(field_in.shape[1])
            if channel_count == len(FIELD_CHANNELS_V2):
                channel_names = list(FIELD_CHANNELS_V2)
            elif channel_count == len(cfg.module_config.vin.scene_field_channels):
                channel_names = list(cfg.module_config.vin.scene_field_channels)
            else:
                channel_names = [f"ch_{idx}" for idx in range(channel_count)]
            default_channels = channel_names[: min(len(channel_names), 6)]
            selected_channels = st.multiselect(
                "Scene field channels",
                options=channel_names,
                default=default_channels,
                key="vin_summary_field_hist_channels",
            )
            log1p_counts = st.checkbox(
                "Log1p histogram counts",
                value=False,
                key="vin_summary_field_hist_log1p",
            )
            hist_bins = int(
                st.slider(
                    "Histogram bins",
                    min_value=10,
                    max_value=200,
                    value=60,
                    key="vin_summary_field_hist_bins",
                ),
            )
            channel_vals = field_in.abs().detach().cpu()
            series: list[tuple[str, np.ndarray]] = []
            for idx, name in enumerate(channel_names):
                if name not in selected_channels:
                    continue
                vals = channel_vals[:, idx, ...].reshape(-1).numpy()
                series.append((name, vals))
            fig_hist = _histogram_overlay(
                series,
                bins=hist_bins,
                title=_pretty_label("Scene field channel |value| distributions"),
                xaxis_title=_pretty_label("|value|"),
                log1p_counts=log1p_counts,
            )
            st.plotly_chart(fig_hist, width="stretch")

        _info_popover(
            "feature norms",
            "Per-candidate L2 norms of feature blocks (pose/global/local). "
            "Very low norms suggest weak signal; very high norms can dominate the "
            "MLP. Compare modalities to spot imbalance or saturation.",
        )
        feat_norms: dict[str, torch.Tensor] = {
            "pose_enc": torch.linalg.vector_norm(debug.pose_enc, dim=-1).reshape(-1),
            "feats": torch.linalg.vector_norm(debug.feats, dim=-1).reshape(-1),
        }
        if hasattr(debug, "local_feat"):
            feat_norms["local_feat"] = torch.linalg.vector_norm(
                debug.local_feat,
                dim=-1,
            ).reshape(-1)
        if getattr(debug, "global_feat", None) is not None:
            feat_norms["global_feat"] = torch.linalg.vector_norm(
                debug.global_feat,
                dim=-1,
            ).reshape(-1)

        log1p_norm_counts = st.checkbox(
            "Log1p feature histogram counts",
            value=False,
            key="vin_summary_feat_norm_log1p",
        )
        norm_series = [(name, _to_numpy(vals)) for name, vals in feat_norms.items()]
        fig = _histogram_overlay(
            norm_series,
            bins=60,
            title=_pretty_label("Feature norms (per-candidate)"),
            xaxis_title=_pretty_label("norm"),
            log1p_counts=log1p_norm_counts,
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("VIN summarize_vin output"):
            include_torchsummary = st.checkbox(
                "Include torchsummary modules",
                value=False,
                key="vin_summary_include_ts",
            )
            torchsummary_depth = int(
                st.slider(
                    "Torchsummary depth",
                    min_value=1,
                    max_value=6,
                    value=3,
                    key="vin_summary_depth",
                ),
            )
            summary_key = (
                f"{state.cfg_sig}|{batch.scene_id}|{batch.snippet_id}|{include_torchsummary}|{torchsummary_depth}"
            )
            if state.summary_key != summary_key:
                state.summary_key = summary_key
                state.summary_text = None
                state.summary_error = None

            auto_run = state.summary_text is None and state.summary_error is None
            if st.button("Generate summary", key="vin_summary_generate") or auto_run:
                try:
                    with st.spinner("Generating VIN summary..."):
                        state.summary_text = state.module.summarize_vin(
                            batch,
                            include_torchsummary=include_torchsummary,
                            torchsummary_depth=torchsummary_depth,
                        )
                    state.summary_error = None
                except Exception as exc:  # pragma: no cover - UI guard
                    state.summary_error = f"{type(exc).__name__}: {exc}"
                    state.summary_text = None

            if state.summary_error:
                st.error(state.summary_error)
            elif state.summary_text:
                st.code(_strip_ansi(state.summary_text))
            else:
                st.info("Click 'Generate summary' to render summarize_vin output.")

    with tab_pose:
        _info_popover(
            "pose descriptor",
            "Candidate centers are translations of `T_rig_ref_cam` "
            "(reference rig frame). Radii are `||t||` in meters. "
            "For VIN v1, direction plots show unit vectors for center "
            "directions and forward axes; view alignment is `dot(f, -u)`, "
            "measuring how much the camera looks back toward the rig. "
            "VIN v2 does not compute frustum tokens.",
        )
        centers = _to_numpy(debug.candidate_center_rig_m.reshape(-1, 3))
        st.plotly_chart(
            plot_radius_hist(
                centers,
                title=_pretty_label("Candidate radii (reference rig)"),
            ),
            width="stretch",
        )

        if has_tokens:
            center_dirs = _to_numpy(debug.candidate_center_dir_rig.reshape(-1, 3))
            forward_dirs = _to_numpy(debug.candidate_forward_dir_rig.reshape(-1, 3))
            view_align = _to_numpy(debug.view_alignment.reshape(-1))

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_direction_polar(
                        center_dirs,
                        title=_pretty_label("Candidate center directions (rig frame)"),
                    ),
                    width="stretch",
                )
            with col2:
                st.plotly_chart(
                    plot_direction_sphere(
                        center_dirs,
                        title=_pretty_label("Center directions on unit sphere"),
                    ),
                    width="stretch",
                )

            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(
                    plot_direction_polar(
                        forward_dirs,
                        title=_pretty_label("Candidate forward directions (rig frame)"),
                    ),
                    width="stretch",
                )
            with col4:
                st.plotly_chart(
                    plot_direction_sphere(
                        forward_dirs,
                        title=_pretty_label("Forward directions on unit sphere"),
                    ),
                    width="stretch",
                )

            log1p_pose_counts = st.checkbox(
                "Log1p alignment histogram counts",
                value=False,
                key="vin_pose_align_log1p",
            )
            fig_align = _histogram_overlay(
                [("alignment", view_align)],
                bins=60,
                title=_pretty_label("View alignment dot(f, -u)"),
                xaxis_title=_pretty_label("dot(f, -u)"),
                log1p_counts=log1p_pose_counts,
            )
            st.plotly_chart(fig_align, width="stretch")
        else:
            st.info("Pose-direction plots are only available for VIN v1 diagnostics.")

    with tab_geometry:
        _info_popover(
            "geometry overview",
            "Combines candidate centers, trajectory, semidense points, and GT mesh "
            "in the same world frame. Frusta and GT OBBs help verify that the "
            "candidate poses align with scene geometry and annotations.",
        )
        snippet_view = batch.efm_snippet_view
        if snippet_view is None and use_offline_cache and attach_snippet:
            snippet_key = f"{batch.scene_id}:{batch.snippet_id}"
            if state.offline_snippet_key != snippet_key or state.offline_snippet is None:
                with st.spinner("Loading EFM snippet for geometry..."):
                    try:
                        cache_ds = state.offline_cache
                        dataset_payload = cache_ds.metadata.dataset_config if cache_ds else None
                        paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
                        snippet_view = _load_efm_snippet_for_cache(
                            scene_id=batch.scene_id,
                            snippet_id=batch.snippet_id,
                            dataset_payload=dataset_payload,
                            device="cpu",
                            paths=paths,
                            include_gt_mesh=include_gt_mesh,
                        )
                        state.offline_snippet_key = snippet_key
                        state.offline_snippet = snippet_view
                        state.offline_snippet_error = None
                    except Exception as exc:  # pragma: no cover - IO guard
                        state.offline_snippet_key = snippet_key
                        state.offline_snippet = None
                        state.offline_snippet_error = f"{type(exc).__name__}: {exc}"
            else:
                snippet_view = state.offline_snippet
            if snippet_view is not None:
                batch.efm_snippet_view = snippet_view

        if snippet_view is None:
            if state.offline_snippet_error:
                st.warning(state.offline_snippet_error)
            st.info(
                "Geometry plots require raw EFM snippets; enable 'Attach EFM snippet' or use online data.",
            )
        else:
            cam_choice, plot_opts = scene_plot_options_ui(
                snippet_view,
                key_prefix="vin_geom",
            )
            frustum_indices = plot_opts.frustum_frame_indices[-1:] if plot_opts.frustum_frame_indices else []
            axis_expander = st.expander("Axes & candidate settings", expanded=False)
            with axis_expander:
                show_reference_axes = st.checkbox(
                    "Show reference axes",
                    value=True,
                    key="vin_geom_ref_axes",
                )
                show_voxel_axes = st.checkbox(
                    "Show voxel axes",
                    value=True,
                    key="vin_geom_voxel_axes",
                )
                distinct_axes = st.checkbox(
                    "Distinct voxel axis colors",
                    value=True,
                    key="vin_geom_distinct_axes",
                )
                candidate_pose_mode = st.selectbox(
                    "Candidate coordinates",
                    options=["ref rig", "world cam", "rig (raw)"],
                    index=0,
                    key="vin_geom_candidate_pose_mode",
                )
                candidate_color_mode = st.selectbox(
                    "Candidate color mode",
                    options=["valid fraction", "solid", "loss"],
                    index=0,
                    key="vin_geom_candidate_color_mode",
                )
                candidate_color = "#ffd966"
                candidate_colorscale = "Viridis"
                if candidate_color_mode == "solid":
                    candidate_color = st.color_picker(
                        "Candidate color",
                        value="#ffd966",
                        key="vin_geom_candidate_color",
                    )
                else:
                    candidate_colorscale = st.selectbox(
                        "Candidate colorscale",
                        options=["Viridis", "Cividis", "Plasma", "Turbo", "Magma"],
                        index=0,
                        key="vin_geom_candidate_colorscale",
                    )

            candidate_frusta_indices: list[int] = []
            candidate_frusta_scale = 0.5
            candidate_frusta_color = "#ff4d4d"
            candidate_frusta_show_axes = False
            candidate_frusta_show_center = False
            candidate_frusta_camera = cam_choice
            candidate_frusta_frame_index = frustum_indices[0] if frustum_indices else 0
            with st.expander("Candidate frusta", expanded=False):
                show_candidate_frusta = st.checkbox(
                    "Show candidate frusta",
                    value=False,
                    key="vin_geom_candidate_frusta",
                )
                if show_candidate_frusta:
                    options = list(range(num_candidates))
                    default = options[: min(4, len(options))] if options else []
                    candidate_frusta_indices = st.multiselect(
                        "Candidate indices",
                        options=options,
                        default=default,
                        key="vin_geom_candidate_frusta_indices",
                    )
                    candidate_frusta_camera = st.selectbox(
                        "Candidate frusta camera",
                        options=["rgb", "slam-l", "slam-r"],
                        index=0,
                        key="vin_geom_candidate_frusta_camera",
                    )
                    candidate_frusta_scale = float(
                        st.slider(
                            "Candidate frusta scale",
                            min_value=0.1,
                            max_value=2.0,
                            value=0.5,
                            step=0.05,
                            key="vin_geom_candidate_frusta_scale",
                        ),
                    )
                    candidate_frusta_color = st.color_picker(
                        "Candidate frusta color",
                        value="#ff4d4d",
                        key="vin_geom_candidate_frusta_color",
                    )
                    candidate_frusta_show_axes = st.checkbox(
                        "Show candidate axes",
                        value=False,
                        key="vin_geom_candidate_frusta_axes",
                    )
                    candidate_frusta_show_center = st.checkbox(
                        "Show candidate center",
                        value=False,
                        key="vin_geom_candidate_frusta_center",
                    )
            backbone_fields: list[str] = []
            backbone_threshold = 0.5
            backbone_max_points = 40_000
            backbone_colorscale = "Viridis"
            available_fields: list[str] = []
            if debug.backbone_out is not None:
                available_fields = [
                    name
                    for name in ("occ_pr", "occ_input", "counts")
                    if getattr(debug.backbone_out, name, None) is not None
                ]
            with st.expander("Backbone evidence overlay", expanded=False):
                show_backbone = st.checkbox(
                    "Overlay backbone evidence",
                    value=False,
                    key="vin_geom_backbone_enable",
                )
                if not available_fields:
                    st.info("Backbone evidence not available in debug outputs.")
                elif show_backbone:
                    backbone_fields = st.multiselect(
                        "Backbone fields",
                        options=available_fields,
                        default=available_fields,
                        key="vin_geom_backbone_fields",
                    )
                    backbone_colorscale = st.selectbox(
                        "Evidence colorscale",
                        options=["Viridis", "Cividis", "Plasma", "Turbo", "Magma"],
                        index=0,
                        key="vin_geom_backbone_colorscale",
                    )
                    backbone_threshold = float(
                        st.slider(
                            "Evidence threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.01,
                            key="vin_geom_backbone_threshold",
                        ),
                    )
                    backbone_max_points = int(
                        st.slider(
                            "Max evidence points",
                            min_value=1000,
                            max_value=200000,
                            value=40000,
                            step=1000,
                            key="vin_geom_backbone_max_points",
                        ),
                    )
            candidate_loss: torch.Tensor | None = None
            if candidate_color_mode == "loss":
                loss_error = None
                binner = getattr(state.module, "_binner", None)
                if binner is None:
                    loss_error = "RRI binner unavailable; cannot compute loss hue."
                elif batch.rri is None:
                    loss_error = "Loss hue requires oracle RRI labels."
                else:
                    try:
                        with torch.no_grad():
                            logits = pred.logits
                            rri = batch.rri.to(device=logits.device)
                            rri_flat = rri.reshape(-1)
                            mask = torch.isfinite(rri_flat)
                            if mask.any():
                                labels = binner.transform(rri_flat)
                                loss_per = coral_loss(
                                    logits.reshape(-1, logits.shape[-1])[mask],
                                    labels[mask],
                                    num_classes=int(binner.num_classes),
                                    reduction="none",
                                )
                                loss_flat = torch.full(
                                    (rri_flat.numel(),),
                                    float("nan"),
                                    device=logits.device,
                                    dtype=torch.float32,
                                )
                                loss_flat[mask] = loss_per
                                candidate_loss = loss_flat.reshape(rri.shape)
                            else:
                                loss_error = "Loss hue requires finite RRI labels."
                    except Exception as exc:  # pragma: no cover - UI guard
                        loss_error = f"{type(exc).__name__}: {exc}"
                if loss_error:
                    st.info(loss_error)
            st.plotly_chart(
                build_geometry_overview_figure(
                    debug,
                    snippet=snippet_view,
                    reference_pose_world_rig=batch.reference_pose_world_rig,
                    max_candidates=512,
                    show_scene_bounds=plot_opts.show_scene_bounds,
                    show_crop_bounds=plot_opts.show_crop_bounds,
                    show_frustum=plot_opts.show_frustum,
                    frustum_camera=cam_choice,
                    frustum_frame_indices=frustum_indices,
                    frustum_scale=plot_opts.frustum_scale,
                    show_gt_obbs=plot_opts.show_gt_obbs,
                    gt_timestamp=plot_opts.gt_timestamp,
                    semidense_mode=plot_opts.semidense_mode,
                    max_sem_points=plot_opts.max_sem_points,
                    show_trajectory=plot_opts.mark_first_last,
                    mark_first_last=plot_opts.mark_first_last,
                    show_reference_axes=show_reference_axes,
                    show_voxel_axes=show_voxel_axes,
                    reference_axis_colors=["red", "green", "blue"],
                    voxel_axis_colors=["cyan", "magenta", "yellow"] if distinct_axes else ["red", "green", "blue"],
                    candidate_pose_mode="ref_rig"
                    if candidate_pose_mode == "ref rig"
                    else "world_cam"
                    if candidate_pose_mode == "world cam"
                    else "rig",
                    candidate_poses_world_cam=batch.candidate_poses_world_cam,
                    candidate_color_mode="solid"
                    if candidate_color_mode == "solid"
                    else "loss"
                    if candidate_color_mode == "loss"
                    else "valid_fraction",
                    candidate_color=candidate_color,
                    candidate_colorscale=candidate_colorscale,
                    candidate_loss=candidate_loss,
                    candidate_frusta_indices=candidate_frusta_indices,
                    candidate_frusta_camera=candidate_frusta_camera,
                    candidate_frusta_frame_index=candidate_frusta_frame_index,
                    candidate_frusta_scale=candidate_frusta_scale,
                    candidate_frusta_color=candidate_frusta_color,
                    candidate_frusta_show_axes=candidate_frusta_show_axes,
                    candidate_frusta_show_center=candidate_frusta_show_center,
                    backbone_fields=backbone_fields,
                    backbone_occ_threshold=backbone_threshold,
                    backbone_max_points=backbone_max_points,
                    backbone_colorscale=backbone_colorscale,
                ),
                width="stretch",
                key="vin_geometry_overview",
            )

        if has_tokens:
            log1p_align_counts = st.checkbox(
                "Log1p alignment histogram counts",
                value=False,
                key="vin_geom_align_log1p",
            )
            alignment_figs = build_alignment_figures(
                debug,
                log1p_counts=log1p_align_counts,
            )
            for key, fig in alignment_figs.items():
                st.plotly_chart(fig, width="stretch", key=f"vin_align_{key}")

    with tab_field:
        _info_popover(
            "scene field",
            "`field_in` is the raw concatenated EVL channels. `field` is the "
            "projected version after 1x1x1 Conv3d + GroupNorm + GELU "
            "(VIN v2), which compresses the channels for attention pooling.",
        )
        channel_labels = cfg.module_config.vin.scene_field_channels
        field_in = debug.field_in
        field = debug.field
        if field_in.cpu().ndim == 5:
            field_in = field_in[0]
        if field.cpu().ndim == 5:
            field = field[0]

        if field_in.cpu().ndim == 4:
            st.subheader("field_in slices (raw)")
            figs_in = build_field_slice_figures(
                field_in,
                channel_names=channel_labels,
                max_channels=4,
                title_prefix="field_in",
            )
            for key, fig in figs_in.items():
                st.plotly_chart(fig, width="stretch", key=f"vin_field_in_{key}")

        if field.cpu().ndim == 4:
            st.subheader("field slices (projected)")
            figs_out = build_field_slice_figures(
                field,
                channel_names=[f"proj_{i}" for i in range(field.shape[0])],
                max_channels=4,
                title_prefix="field",
            )
            for key, fig in figs_out.items():
                st.plotly_chart(fig, width="stretch", key=f"vin_field_{key}")

        if has_tokens:
            log1p_field_counts = st.checkbox(
                "Log1p token histogram counts",
                value=False,
                key="vin_field_token_log1p",
            )
            token_figs = build_field_token_histograms(
                debug,
                channel_names=channel_labels,
                max_channels=4,
                log1p_counts=log1p_field_counts,
            )
            for key, fig in token_figs.items():
                st.plotly_chart(fig, width="stretch", key=f"vin_field_token_{key}")

    with tab_tokens:
        if not has_tokens:
            st.info("Frustum token diagnostics are only available for VIN v1.")
        else:
            _info_popover(
                "frustum tokens",
                "VIN v1 samples the scene field along a frustum grid of size "
                "grid_size x grid_size x num_depths. Token norms show feature "
                "strength per sample; token_valid marks whether a sample lies "
                "inside the voxel field.",
            )
            grid_size = int(cfg.module_config.vin.frustum_grid_size)
            depth_values = list(cfg.module_config.vin.frustum_depths_m)
            num_depths = len(depth_values)

            cand_idx = st.slider(
                "Candidate index",
                0,
                max(0, num_candidates - 1),
                0,
                key="vin_token_cand_idx",
            )
            max_depths = st.slider(
                "Max depth planes",
                1,
                max(1, num_depths),
                min(4, num_depths),
                key="vin_token_depths",
            )

            tokens = debug.tokens[0, cand_idx]
            token_valid = debug.token_valid[0, cand_idx]
            token_norm = torch.linalg.vector_norm(tokens, dim=-1)

            expected_k = grid_size * grid_size * num_depths
            if int(token_norm.numel()) != expected_k:
                st.warning(
                    "Token count does not match grid_size² × num_depths; skipping grid view.",
                )
            else:
                token_norm = token_norm.view(num_depths, grid_size, grid_size)
                token_valid = token_valid.view(num_depths, grid_size, grid_size).float()

                depth_labels = [f"d={d:g}m" for d in depth_values[:max_depths]]
                norm_slices = [_to_numpy(token_norm[i]) for i in range(max_depths)]
                valid_slices = [_to_numpy(token_valid[i]) for i in range(max_depths)]

                fig_norm = _plot_slice_grid(
                    norm_slices,
                    titles=depth_labels,
                    title=_pretty_label("Token feature norm per depth plane"),
                    percentile=99.0,
                    symmetric=False,
                    cmap_name="viridis",
                )
                st.plotly_chart(fig_norm, width="stretch")

                fig_valid = _plot_slice_grid(
                    valid_slices,
                    titles=depth_labels,
                    title=_pretty_label("Token validity per depth plane"),
                    percentile=100.0,
                    symmetric=False,
                    cmap_name="gray",
                )
                st.plotly_chart(fig_valid, width="stretch")

            st.subheader("Frustum samples in world")
            st.plotly_chart(
                build_frustum_samples_figure(
                    debug,
                    p3d_cameras=batch.p3d_cameras,
                    candidate_index=int(cand_idx),
                    grid_size=int(grid_size),
                    depths_m=depth_values,
                ),
                width="stretch",
            )

    with tab_evidence:
        _info_popover(
            "evidence",
            "Visualizes voxel evidence (occupancy, centerness, or scene-field "
            "channels) above a threshold. This exposes where the backbone sees "
            "occupied space versus free or unknown space.",
        )
        channel_labels = cfg.module_config.vin.scene_field_channels
        occ_threshold = float(
            st.slider(
                "Evidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            ),
        )
        evidence_figs = build_scene_field_evidence_figures(
            debug,
            channel_names=channel_labels,
            occ_threshold=occ_threshold,
            max_points=20000,
        )
        if not evidence_figs:
            evidence_figs = build_backbone_evidence_figures(
                debug,
                occ_threshold=occ_threshold,
                max_points=20000,
            )
        if not evidence_figs:
            st.info("No backbone evidence tensors found for plotting.")
        for key, fig in evidence_figs.items():
            st.plotly_chart(fig, width="stretch", key=f"vin_evidence_{key}")

    with tab_transforms:
        _info_popover(
            "transforms",
            "Roundtrip plots validate world-to-voxel transforms used to align "
            "candidate poses with the EVL grid. Large residuals indicate frame "
            "mismatch or incorrect voxel extents.",
        )
        log1p_roundtrip = st.checkbox(
            "Log1p roundtrip histogram counts",
            value=False,
            key="vin_roundtrip_log1p",
        )
        st.plotly_chart(
            build_voxel_roundtrip_figure(debug, log1p_counts=log1p_roundtrip),
            width="stretch",
        )
        _info_popover(
            "se3 closure",
            "Checks chain consistency of SE(3): "
            "T_world_cam vs T_world_rig_ref * T_rig_ref_cam. "
            "Near-zero translation/rotation residuals indicate that pose "
            "composition and inversion are consistent.",
        )
        st.plotly_chart(
            build_se3_closure_figure(
                batch.candidate_poses_world_cam,
                batch.reference_pose_world_rig,
            ),
            width="stretch",
        )
        _info_popover(
            "voxel in-bounds",
            "Transforms candidate centers into the voxel frame and reports "
            "the fraction inside the voxel extent. Normalized coordinate "
            "histograms show whether per-axis scaling stays within [-1, 1].",
        )
        st.plotly_chart(
            build_voxel_inbounds_figure(
                batch.candidate_poses_world_cam,
                debug.backbone_out.t_world_voxel,
                debug.backbone_out.voxel_extent,
            ),
            width="stretch",
        )
        vin_model = state.module.vin if state.module is not None else None
        if vin_model is not None and hasattr(vin_model, "_pos_grid_from_pts_world"):
            try:
                field_in = debug.field_in
                grid_shape = (
                    int(field_in.shape[-3]),
                    int(field_in.shape[-2]),
                    int(field_in.shape[-1]),
                )
                pos_grid = vin_model._pos_grid_from_pts_world(
                    debug.backbone_out.pts_world,
                    t_world_voxel=debug.backbone_out.t_world_voxel,
                    pose_world_rig_ref=batch.reference_pose_world_rig,
                    voxel_extent=debug.backbone_out.voxel_extent,
                    grid_shape=grid_shape,
                )
                _info_popover(
                    "pos grid linearity",
                    "Fits an affine map from voxel coordinates to pos_grid values "
                    "and reports R² per rig axis. High R² indicates that the "
                    "positional grid is a linear transform of voxel coords.",
                )
                st.plotly_chart(
                    build_pos_grid_linearity_figure(
                        pos_grid,
                        debug.backbone_out.voxel_extent,
                    ),
                    width="stretch",
                )
            except Exception as exc:  # pragma: no cover - optional diagnostics
                st.info(
                    f"Pos-grid linearity unavailable: {type(exc).__name__}: {exc}",
                )
        if has_tokens:
            pred_norm = _to_numpy(pred.expected_normalized.reshape(-1))
            st.plotly_chart(
                build_prediction_alignment_figure(debug, expected_normalized=pred_norm),
                width="stretch",
            )
        else:
            st.info("Prediction alignment plot is only available for VIN v1.")

    with tab_concept:
        _info_popover(
            "ff encodings",
            "This tab inspects the pose-encoding pathway used by VIN. "
            "VIN v1 relies on spherical features plus a learnable Fourier "
            "feature (LFF) block; VIN v2 uses LFF over the pose vector "
            "`[t_x, t_y, t_z, r6d_0..5]` and a positional grid for attention keys. "
            "The plots below show distributions and low-dimensional projections "
            "to diagnose scale, anisotropy, and feature collapse.",
        )
        pose_encoder_lff = state.module.vin.pose_encoder_lff if state.module is not None else None
        pose_enc = debug.pose_enc.reshape(-1, debug.pose_enc.shape[-1])

        if has_tokens:
            st.caption(
                "Plot Learnable Fourier Features for the actual encoded candidates.",
            )
            _info_popover(
                "lff diagnostics",
                "These plots visualize the LFF block used inside the pose encoder. "
                "Weight-space figures (Wr and its norms) show learned frequency "
                "directions, while candidate-space figures show the actual "
                "Fourier activations for the current batch. Look for mode collapse "
                "(features with near-zero variance) or extreme saturation.",
            )
            lmax = int(cfg.plot_lmax)
            sh_norm = str(cfg.plot_sh_normalization)
            freq_list = list(cfg.plot_radius_freqs)
            max_candidates = int(pose_enc.shape[0])
            max_pose_dims = int(pose_enc.shape[-1])
            max_sh_components = 64
            save_html = st.checkbox(
                "Save HTML to .logs/vin/streamlit",
                value=False,
                key="vin_save_html",
            )
            log1p_counts = st.checkbox(
                "Log1p histogram counts",
                value=False,
                key="vin_plot_log1p",
            )
            plot_btn = st.button("Generate encoding plots", key="vin_plot_btn")

            if plot_btn:
                figs = build_vin_encoding_figures(
                    debug,
                    lmax=int(lmax),
                    sh_normalization=str(sh_norm),
                    radius_freqs=freq_list,
                    pose_encoder_lff=pose_encoder_lff,
                    include_legacy_sh=False,
                    log1p_counts=log1p_counts,
                )
                actual_figs = build_candidate_encoding_figures(
                    debug,
                    lmax=int(lmax),
                    sh_normalization=str(sh_norm),
                    radius_freqs=freq_list,
                    pose_encoder_lff=pose_encoder_lff,
                    include_legacy_sh=False,
                    max_candidates=int(max_candidates),
                    max_sh_components=int(max_sh_components),
                    max_pose_dims=int(max_pose_dims),
                )
                if save_html:
                    out_dir = cfg.paths.resolve_under_root(
                        Path(".logs") / "vin" / "streamlit",
                    )
                    stem = f"{batch.scene_id}_{batch.snippet_id}".replace("/", "_")
                    save_vin_encoding_figures(
                        figs | actual_figs,
                        out_dir=out_dir,
                        file_stem_prefix=stem,
                    )
                for label, fig in figs.items():
                    st.plotly_chart(fig, width="stretch", key=f"vin_plot_{label}")
                for label, fig in actual_figs.items():
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        key=f"vin_plot_actual_{label}",
                    )
        else:
            st.caption("VIN v2 positional encodings (pose grid + LFF pose encoder).")
            vin_model = state.module.vin if state.module is not None else None
            if vin_model is None:
                st.info("VIN model not available.")
            else:
                pose_vec = debug.pose_vec
                if pose_vec is not None:
                    _info_popover(
                        "pose vector",
                        "Histogram of a single pose-vector component. "
                        "Translation entries reflect candidate displacement "
                        "in the reference rig frame; rotation entries are "
                        "the 6D rotation representation. Extreme ranges or "
                        "heavy skew indicate scaling issues before LFF.",
                    )
                    dim_labels = [
                        "t_x",
                        "t_y",
                        "t_z",
                        "r6d_0",
                        "r6d_1",
                        "r6d_2",
                        "r6d_3",
                        "r6d_4",
                        "r6d_5",
                    ]
                    dim_index = int(
                        st.selectbox(
                            "Pose input component",
                            options=list(range(len(dim_labels))),
                            format_func=lambda idx: dim_labels[idx],
                            key="vin_pose_dim",
                        ),
                    )
                    log1p_pose_counts = st.checkbox(
                        "Log1p pose histogram counts",
                        value=False,
                        key="vin_pose_vec_log1p",
                    )
                    st.plotly_chart(
                        build_pose_vec_histogram(
                            pose_vec,
                            dim_index=dim_index,
                            num_bins=60,
                            log1p_counts=log1p_pose_counts,
                        ),
                        width="stretch",
                    )

                    max_features = int(
                        st.slider(
                            "Max features",
                            min_value=16,
                            max_value=256,
                            value=96,
                        ),
                    )
                    hist_bins = int(
                        st.slider(
                            "LFF hist bins",
                            min_value=20,
                            max_value=200,
                            value=60,
                        ),
                    )
                    max_points = int(
                        st.slider(
                            "LFF max points",
                            min_value=1000,
                            max_value=20000,
                            value=8000,
                        ),
                    )
                    log1p_lff_counts = st.checkbox(
                        "Log1p LFF histogram counts",
                        value=False,
                        key="vin_lff_hist_log1p",
                    )

                    if pose_encoder_lff is not None:
                        _info_popover(
                            "lff empirical",
                            "Empirical histograms and PCA for the LFF block. "
                            "Fourier features are the raw sin/cos projection "
                            "of the pose vector; the MLP output is the learned "
                            "mixture. PCA helps spot anisotropy or dead features.",
                        )
                        lff_figs = build_lff_empirical_figures(
                            pose_vec,
                            pose_encoder_lff,
                            max_features=max_features,
                            hist_bins=hist_bins,
                            max_points=max_points,
                            log1p_counts=log1p_lff_counts,
                        )
                        st.plotly_chart(
                            lff_figs["lff_empirical_fourier_hist"],
                            width="stretch",
                        )
                        st.plotly_chart(
                            lff_figs["lff_empirical_mlp_hist"],
                            width="stretch",
                        )
                        st.plotly_chart(
                            lff_figs["lff_empirical_fourier_pca"],
                            width="stretch",
                        )
                        st.plotly_chart(
                            lff_figs["lff_empirical_mlp_pca"],
                            width="stretch",
                        )

                    color_mode = st.selectbox(
                        "Pose encoding color",
                        options=["translation_norm", "candidate_index"],
                        index=0,
                        key="vin_pose_enc_color",
                    )
                    if color_mode == "translation_norm":
                        color_values = torch.linalg.vector_norm(
                            pose_vec[..., :3],
                            dim=-1,
                        )
                        color_label = "|t|"
                    else:
                        color_values = torch.arange(
                            pose_vec.shape[1],
                            device=pose_vec.device,
                        ).view(1, -1)
                        color_label = "candidate idx"
                    _info_popover(
                        "pose enc pca",
                        "PCA of the final LFF pose encoding. Color can reflect "
                        "translation magnitude or candidate index. A smooth "
                        "gradient suggests that pose magnitude is well represented; "
                        "tight clumps can indicate collapsed embeddings.",
                    )
                    st.plotly_chart(
                        build_pose_enc_pca_figure(
                            debug.pose_enc,
                            color_values=color_values,
                            color_label=color_label,
                        ),
                        width="stretch",
                    )

                try:
                    field_in = debug.field_in
                    grid_shape = (
                        int(field_in.shape[-3]),
                        int(field_in.shape[-2]),
                        int(field_in.shape[-1]),
                    )
                    pos_grid = vin_model._pos_grid_from_pts_world(
                        debug.backbone_out.pts_world,
                        t_world_voxel=debug.backbone_out.t_world_voxel,
                        pose_world_rig_ref=batch.reference_pose_world_rig,
                        voxel_extent=debug.backbone_out.voxel_extent,
                        grid_shape=grid_shape,
                    )
                    axis = st.selectbox(
                        "Grid axis",
                        options=["D", "H", "W"],
                        index=0,
                        key="vin_grid_axis",
                    )
                    max_index = {
                        "D": grid_shape[0],
                        "H": grid_shape[1],
                        "W": grid_shape[2],
                    }[axis] - 1
                    slice_idx = int(
                        st.slider(
                            "Grid slice index",
                            min_value=0,
                            max_value=max_index,
                            value=max_index // 2,
                        ),
                    )
                    _info_popover(
                        "pos grid slices",
                        "Position grid slices show the normalized voxel centers "
                        "in the reference rig frame (pos_x/pos_y/pos_z). "
                        "Expect near-linear gradients across each axis; "
                        "distortions indicate mismatched voxel extents or frames.",
                    )
                    st.plotly_chart(
                        build_pose_grid_slices_figure(
                            pos_grid,
                            axis=axis,
                            index=slice_idx,
                        ),
                        width="stretch",
                    )
                    color_by = st.selectbox(
                        "Pos grid PCA color",
                        options=["radius", "x", "y", "z"],
                        index=0,
                        key="vin_grid_color",
                    )
                    show_axes = st.checkbox(
                        "Show rig axes",
                        value=True,
                        key="vin_grid_axes",
                    )
                    axis_scale = float(
                        st.slider(
                            "Axis scale",
                            min_value=0.1,
                            max_value=2.0,
                            value=0.5,
                            step=0.1,
                        ),
                    )
                    _info_popover(
                        "pos grid pca",
                        "PCA of the positional embeddings used as attention keys "
                        "for the global pool. The axis overlays show how the "
                        "learned projection aligns with x/y/z directions.",
                    )
                    st.plotly_chart(
                        build_pose_grid_pca_figure(
                            pos_grid,
                            pos_proj=vin_model.global_pooler.pos_proj,
                            max_points=8000,
                            color_by=color_by,
                            show_axes=show_axes,
                            axis_scale=axis_scale,
                        ),
                        width="stretch",
                    )
                except Exception as exc:  # pragma: no cover - optional diagnostics
                    st.info(
                        f"Positional grid plots unavailable: {type(exc).__name__}: {exc}",
                    )

    with tab_coral:
        _info_popover(
            "coral diagnostics",
            "CORAL models cumulative probabilities P(y > k) for ordinal bins. "
            "This panel visualizes threshold probabilities, marginal class "
            "probabilities, bin representatives, and per-candidate loss/entropy. "
            "Use it to validate bin calibration and monotonicity.",
        )
        binner = getattr(state.module, "_binner", None)
        head_coral = getattr(state.module.vin, "head_coral", None) if state.module is not None else None

        logits = pred.logits
        probs = pred.prob
        if logits.ndim == 3:
            logits = logits[0]
        if probs.ndim == 3:
            probs = probs[0]
        num_classes = int(probs.shape[-1])

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("RRI distribution + bin edges")
            if batch.rri is None:
                st.info("Oracle RRI values unavailable in this batch.")
            elif binner is None:
                st.info("RRI binner unavailable; cannot overlay bin edges.")
            else:
                rri_flat = batch.rri.reshape(-1).detach().cpu().numpy()
                edges = binner.edges.detach().cpu().numpy()
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=rri_flat,
                        nbinsx=60,
                        name="RRI",
                        marker_color="#5da5da",
                        opacity=0.75,
                    ),
                )
                for edge in edges.tolist():
                    fig_hist.add_vline(
                        x=edge,
                        line_dash="dot",
                        line_width=1,
                        line_color="gray",
                    )
                fig_hist.update_layout(
                    title=_pretty_label("Oracle RRI with bin edges"),
                    xaxis_title=_pretty_label("RRI"),
                    yaxis_title=_pretty_label("count"),
                    barmode="overlay",
                )
                st.plotly_chart(fig_hist, width="stretch")

        with col_right:
            st.subheader("Bin representatives")
            if binner is None:
                st.info("RRI binner unavailable.")
            else:
                bin_means = binner.bin_means
                midpoints = binner.class_midpoints()
                learned = None
                if head_coral is not None and getattr(head_coral, "has_bin_values", False):
                    learned = head_coral.bin_values.values().detach().cpu()

                fig_bins = go.Figure()
                if bin_means is not None:
                    fig_bins.add_trace(
                        go.Scatter(
                            x=list(range(num_classes)),
                            y=bin_means.detach().cpu().numpy(),
                            mode="lines+markers",
                            name="bin_mean",
                        ),
                    )
                if midpoints is not None:
                    fig_bins.add_trace(
                        go.Scatter(
                            x=list(range(num_classes)),
                            y=midpoints.detach().cpu().numpy(),
                            mode="lines+markers",
                            name="midpoint",
                        ),
                    )
                if learned is not None:
                    fig_bins.add_trace(
                        go.Scatter(
                            x=list(range(num_classes)),
                            y=learned.numpy(),
                            mode="lines+markers",
                            name="learned_u",
                        ),
                    )
                fig_bins.update_layout(
                    title=_pretty_label("Bin representatives (u_k)"),
                    xaxis_title=_pretty_label("bin index"),
                    yaxis_title=_pretty_label("RRI value"),
                )
                st.plotly_chart(fig_bins, width="stretch")

        st.subheader("Candidate-level CORAL outputs")
        cand_idx = st.slider(
            "Candidate index",
            min_value=0,
            max_value=max(0, num_candidates - 1),
            value=0,
            key="vin_coral_candidate",
        )
        cand_logits = logits[cand_idx]
        cand_probs = probs[cand_idx]
        cand_p_gt = torch.sigmoid(cand_logits).detach().cpu().numpy()
        cand_probs_np = cand_probs.detach().cpu().numpy()

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            fig_pgt = go.Figure()
            fig_pgt.add_trace(
                go.Scatter(
                    x=list(range(num_classes - 1)),
                    y=cand_p_gt,
                    mode="lines+markers",
                    name="P(y>k)",
                ),
            )
            fig_pgt.update_layout(
                title=_pretty_label("Cumulative probabilities"),
                xaxis_title=_pretty_label("threshold k"),
                yaxis_title=_pretty_label("P(y>k)"),
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig_pgt, width="stretch")

        with col_b:
            fig_pi = go.Figure()
            fig_pi.add_trace(
                go.Bar(
                    x=list(range(num_classes)),
                    y=cand_probs_np,
                    name="P(y=k)",
                ),
            )
            fig_pi.update_layout(
                title=_pretty_label("Marginal class probabilities"),
                xaxis_title=_pretty_label("class k"),
                yaxis_title=_pretty_label("P(y=k)"),
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig_pi, width="stretch")

        with col_c:
            st.subheader("Expected values")
            ordinal_expected = cand_p_gt.sum()
            st.metric("E[y] (ordinal)", f"{float(ordinal_expected):.3f}")
            if head_coral is not None and getattr(head_coral, "has_bin_values", False):
                pred_rri = float(head_coral.expected_from_probs(cand_probs).item())
                st.metric("E[RRI] (learned u_k)", f"{pred_rri:.4f}")
            elif binner is not None:
                pred_rri = float(binner.expected_from_probs(cand_probs).item())
                st.metric("E[RRI] (bin means)", f"{pred_rri:.4f}")
            else:
                st.info("No bin representatives available.")

        st.subheader("CORAL diagnostics across candidates")
        monotonicity = coral_monotonicity_violation_rate(logits).detach().cpu().numpy()
        fig_mono = _histogram_overlay(
            [("monotonicity_violation_rate", monotonicity)],
            bins=60,
            title=_pretty_label("Monotonicity violation rate"),
            xaxis_title=_pretty_label("fraction of violations"),
            log1p_counts=False,
        )
        st.plotly_chart(fig_mono, width="stretch")

        if batch.rri is not None and binner is not None:
            rri_flat = batch.rri.reshape(-1)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            mask = torch.isfinite(rri_flat)
            if mask.any():
                labels = binner.transform(rri_flat)
                loss_per = coral_loss(
                    logits_flat[mask],
                    labels[mask],
                    num_classes=int(binner.num_classes),
                    reduction="none",
                )
                fig_loss = _histogram_overlay(
                    [("coral_loss", loss_per.detach().cpu().numpy())],
                    bins=60,
                    title=_pretty_label("CORAL loss per candidate"),
                    xaxis_title=_pretty_label("loss"),
                    log1p_counts=False,
                )
                st.plotly_chart(fig_loss, width="stretch")
            else:
                st.info("No finite RRI labels available for loss diagnostics.")
        else:
            st.info("Loss diagnostics require oracle RRIs and a fitted binner.")


def render_rri_binning_page() -> None:
    """Render RRI binning diagnostics from saved fit data."""
    st.header("RRI Binning")
    st.caption(
        "Inspect the RRI distribution and quantile edges used for CORAL binning, "
        "loaded directly from saved binner artifacts.",
    )
    _info_popover(
        "rri binning",
        "The binner is fit on cached oracle RRIs stored in "
        "`rri_binner_fit_data.pt` (raw RRI samples) and "
        "`rri_binner.json` (quantile edges + optional per-bin stats). "
        "This view uses only those artifacts—no dataset reloading—to verify "
        "binning quality and the empirical bin means/stds used for expected-value "
        "computations."
        "\nTo refit the binner on updated data, run "
        "`uv run nbv-fit-binner [--config-path .configs/offline_only.toml]`.",
    )

    default_fit_path = Path(".logs") / "vin" / "rri_binner_fit_data.pt"
    default_edges_path = Path(".logs") / "vin" / "rri_binner.json"

    col_a, col_b = st.columns(2)
    with col_a:
        fit_path_str = st.text_input(
            "Fit data (.pt)",
            value=str(default_fit_path),
            key="rri_binner_fit_path",
        )
    with col_b:
        edges_path_str = st.text_input(
            "Binner edges (.json)",
            value=str(default_edges_path),
            key="rri_binner_edges_path",
        )

    log1p_counts = st.checkbox(
        "Log1p histogram counts",
        value=False,
        key="rri_binner_log1p",
    )
    bins = int(
        st.slider(
            "Histogram bins",
            min_value=20,
            max_value=200,
            value=80,
            step=10,
            key="rri_binner_bins",
        ),
    )

    try:
        fit_path = Path(fit_path_str).expanduser()
        edges_path = Path(edges_path_str).expanduser()
        if not fit_path.exists():
            st.warning(f"Fit data not found: {fit_path}")
            return
        if not edges_path.exists():
            st.warning(f"Binner JSON not found: {edges_path}")
            return

        rri = _load_rri_fit_data(fit_path)
        binner = _load_binner_data(edges_path)
    except Exception as exc:  # pragma: no cover - UI guard
        _report_exception(exc, context="Failed to load RRI binning data")
        return

    if rri.numel() == 0:
        st.info("Fit data contains no finite RRI values.")
        return

    edges = binner.edges
    num_classes = int(binner.num_classes or (edges.numel() + 1))
    rri_stats = {
        "samples": int(rri.numel()),
        "min": float(rri.min().item()),
        "max": float(rri.max().item()),
        "mean": float(rri.mean().item()),
        "median": float(rri.median().item()),
    }

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Samples", rri_stats["samples"])
    col2.metric("Mean RRI", f"{rri_stats['mean']:.4f}")
    col3.metric("Median RRI", f"{rri_stats['median']:.4f}")
    col4.metric("Min RRI", f"{rri_stats['min']:.4f}")
    col5.metric("Max RRI", f"{rri_stats['max']:.4f}")

    random_coral_loss = float(max(1, num_classes - 1) * math.log(2.0))
    col_a, col_b = st.columns(2)
    col_a.metric(
        "Random-guess CORAL loss",
        f"{random_coral_loss:.4f}",
        help="Assumes logits=0 (p=0.5) for each threshold; loss = (K-1)*log(2).",
    )
    random_probs = torch.full((num_classes,), 1.0 / float(num_classes), dtype=torch.float32)
    random_expected = float(binner.expected_from_probs(random_probs).item())
    col_b.metric(
        "Uniform-guess expected RRI",
        f"{random_expected:.4f}",
        help="Expected RRI from uniform class probabilities using bin means (or midpoints if means missing).",
    )

    rri_np = rri.cpu().numpy()
    series = [("rri", rri_np)]
    fig_hist = _histogram_overlay(
        series,
        bins=bins,
        title="Raw oracle RRI distribution",
        xaxis_title="rri",
        log1p_counts=log1p_counts,
    )
    for edge in edges.detach().cpu().numpy().tolist():
        fig_hist.add_vline(
            x=float(edge),
            line_width=1,
            line_dash="dash",
            line_color="black",
            opacity=0.3,
        )
    st.plotly_chart(fig_hist, width="stretch")

    labels = binner.transform(rri)
    counts = torch.bincount(labels.to(dtype=torch.int64), minlength=int(num_classes)).cpu().numpy()
    bin_means = binner.bin_means
    bin_stds = binner.bin_stds
    if bin_means is None or bin_stds is None:
        means = torch.empty(int(num_classes), dtype=rri.dtype)
        stds = torch.empty_like(means)
        midpoints = binner.class_midpoints().to(device=rri.device, dtype=rri.dtype)
        for idx in range(int(num_classes)):
            vals = rri[labels == idx]
            if vals.numel() == 0:
                means[idx] = midpoints[idx]
                stds[idx] = 0.0
            else:
                means[idx] = vals.mean()
                stds[idx] = vals.std(unbiased=False)
        bin_means = means.cpu()
        bin_stds = stds.cpu()

    midpoints = binner.class_midpoints().cpu()
    stats_df = pd.DataFrame(
        {
            "class": np.arange(int(num_classes)),
            "count": counts,
            "midpoint": midpoints.numpy(),
            "bin_mean": bin_means.numpy() if bin_means is not None else midpoints.numpy(),
            "bin_std": bin_stds.numpy() if bin_stds is not None else np.zeros_like(midpoints.numpy()),
        },
    )
    st.subheader("Per-bin statistics")
    st.dataframe(stats_df, width="stretch", height=260)

    fig_means = go.Figure()
    fig_means.add_trace(
        go.Bar(
            x=stats_df["class"],
            y=stats_df["bin_mean"],
            error_y={"type": "data", "array": stats_df["bin_std"], "visible": True},
            name="bin mean",
        ),
    )
    fig_means.add_trace(
        go.Scatter(
            x=stats_df["class"],
            y=stats_df["midpoint"],
            mode="lines+markers",
            name="midpoint",
        ),
    )
    fig_means.update_layout(
        title=_pretty_label("Bin means (±1 std) vs midpoints"),
        xaxis_title="class",
        yaxis_title="rri",
    )
    st.plotly_chart(fig_means, width="stretch")
    if log1p_counts:
        counts = np.log1p(counts)
        y_title = "log1p(count)"
    else:
        y_title = "count"
    fig_labels = px.bar(
        x=np.arange(int(num_classes)),
        y=counts,
        labels={"x": "label", "y": y_title},
        title=_pretty_label("Ordinal labels (K classes)"),
    )
    st.plotly_chart(fig_labels, width="stretch")


def render_wandb_analysis_page() -> None:
    """Render analytics derived from W&B run history."""
    st.header("W&B Run Analysis")
    _info_popover(
        "wandb analysis",
        "This panel pulls W&B run history and computes derived diagnostics "
        "beyond the default dashboards: train/val gaps, metric stability, "
        "cross-metric correlations, and calibration bias between predicted "
        "and oracle RRI. It also surfaces confusion matrices and label "
        "histograms logged by the Lightning module.",
    )

    if wandb is None:
        st.error("wandb is not available. Install it to use this panel.")
        return

    cache_key = "wandb_analysis_cache"
    cache = st.session_state.get(cache_key, {})

    default_entity = os.environ.get("WANDB_ENTITY", "")
    default_project = os.environ.get("WANDB_PROJECT", "aria-nbv")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    run_ref = col_a.text_input(
        "Run path / id / name",
        value=str(cache.get("run_ref", "")),
        placeholder="entity/project/runs/<run_id> or <run_id> or <display_name>",
    )
    entity = col_b.text_input(
        "Entity",
        value=str(cache.get("entity", default_entity)),
    )
    project = col_c.text_input(
        "Project",
        value=str(cache.get("project", default_project)),
    )

    col_d, col_e = st.columns([1, 1])
    max_rows = col_d.number_input(
        "History rows",
        min_value=100,
        max_value=50000,
        value=int(cache.get("max_rows", 2000)),
        step=100,
    )
    media_rows = col_e.number_input(
        "Media rows",
        min_value=50,
        max_value=5000,
        value=int(cache.get("media_rows", 400)),
        step=50,
    )

    load_btn = st.button("Load / refresh run")

    if load_btn:
        if not run_ref.strip():
            st.warning("Enter a run id, name, or full W&B path first.")
        else:
            try:
                api = wandb.Api()
                run = _resolve_wandb_run(
                    api=api,
                    run_ref=run_ref,
                    entity=entity.strip() or None,
                    project=project.strip() or None,
                )
                history = _load_wandb_history(
                    run,
                    keys=None,
                    max_rows=int(max_rows),
                )
                cache = {
                    "run": run,
                    "history": history,
                    "run_ref": run_ref,
                    "entity": entity,
                    "project": project,
                    "max_rows": int(max_rows),
                    "media_rows": int(media_rows),
                }
                st.session_state[cache_key] = cache
            except Exception as exc:  # pragma: no cover - API guard
                _report_exception(exc, context="W&B run load failed")
                return

    run = cache.get("run")
    history = cache.get("history")
    if run is None or history is None:
        st.info("Load a W&B run to view analytics.")
        return

    st.subheader("Run summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Run ID", getattr(run, "id", "n/a"))
    col2.metric("Name", getattr(run, "name", "n/a"))
    col3.metric("State", getattr(run, "state", "n/a"))
    col4.metric("Project", getattr(run, "project", "n/a"))
    st.caption(f"Run path: {getattr(run, 'path', 'n/a')}")

    with st.expander("Config", expanded=False):
        try:
            st.json(dict(run.config))
        except Exception:
            st.json({})

    with st.expander("Summary", expanded=False):
        try:
            st.json(dict(run.summary))
        except Exception:
            st.json({})

    history = history.copy()
    if history.empty:
        st.warning("Run history is empty.")
        return

    numeric_cols = [col for col in history.columns if pd.api.types.is_numeric_dtype(history[col])]
    x_candidates = [col for col in ("trainer/global_step", "_step", "global_step", "epoch") if col in history.columns]
    if not x_candidates:
        history["row"] = np.arange(len(history))
        x_candidates = ["row"]

    x_key = st.selectbox("X-axis", options=x_candidates, index=0)

    with st.expander("Metric coverage", expanded=False):
        coverage_rows = []
        for col in numeric_cols:
            series = history[col]
            coverage_rows.append(
                {
                    "metric": col,
                    "non_null": int(series.notna().sum()),
                    "min": float(series.min()) if series.notna().any() else float("nan"),
                    "max": float(series.max()) if series.notna().any() else float("nan"),
                },
            )
        coverage_df = pd.DataFrame(coverage_rows).sort_values(
            "non_null",
            ascending=False,
        )
        st.dataframe(coverage_df, width="stretch", height=240)

    st.subheader("Derived trend analysis")
    _info_popover(
        "trend analysis",
        "Overlay raw curves with an exponential moving average (EMA) to "
        "highlight slow dynamics and suppress minibatch noise.",
    )
    trend_metrics = st.multiselect(
        "Metrics to plot",
        options=numeric_cols,
        default=[m for m in numeric_cols if m.startswith("train/")][:3],
    )
    ema_alpha = st.slider(
        "EMA alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )
    show_raw = st.checkbox("Show raw curves", value=True)
    if trend_metrics:
        fig = go.Figure()
        for metric in trend_metrics:
            df_metric = history[[x_key, metric]].dropna()
            if df_metric.empty:
                continue
            df_metric = df_metric.sort_values(x_key)
            if show_raw:
                fig.add_trace(
                    go.Scatter(
                        x=df_metric[x_key],
                        y=df_metric[metric],
                        mode="lines",
                        name=f"{metric} (raw)",
                        line={"width": 1, "dash": "dot"},
                        opacity=0.6,
                    ),
                )
            if ema_alpha > 0.0:
                smooth = df_metric[metric].ewm(alpha=ema_alpha, adjust=False).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df_metric[x_key],
                        y=smooth,
                        mode="lines",
                        name=f"{metric} (ema)",
                    ),
                )
        fig.update_layout(
            title=_pretty_label("Trend explorer"),
            xaxis_title=_pretty_label(x_key),
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Train/Val gap")
    _info_popover(
        "gap",
        "Plots train minus validation values to diagnose generalization. "
        "Positive gaps on loss indicate overfitting; negative gaps on "
        "RRI metrics can indicate validation outperformance or data shift.",
    )
    pairs = _metric_pairs(list(history.columns))
    gap_bases = sorted(base for base, stages in pairs.items() if "train" in stages and "val" in stages)
    selected_gaps = st.multiselect(
        "Gap metrics",
        options=gap_bases,
        default=[b for b in gap_bases if "loss" in b][:1],
    )
    prefer_suffix = "epoch" if "epoch" in x_key else "step"
    for base in selected_gaps:
        train_key = _select_metric_key(pairs[base]["train"], prefer_suffix)
        val_key = _select_metric_key(pairs[base]["val"], prefer_suffix)
        if train_key is None or val_key is None:
            continue
        df_gap = history[[x_key, train_key, val_key]].dropna()
        if df_gap.empty:
            continue
        df_gap = df_gap.sort_values(x_key)
        gap = df_gap[train_key] - df_gap[val_key]
        fig_gap = go.Figure(
            go.Scatter(x=df_gap[x_key], y=gap, mode="lines", name=base),
        )
        fig_gap.update_layout(
            title=_pretty_label(f"Train - Val gap: {base}"),
            xaxis_title=_pretty_label(x_key),
            yaxis_title=_pretty_label("gap"),
        )
        st.plotly_chart(fig_gap, width="stretch")

    st.subheader("RRI calibration bias & variance")
    _info_popover(
        "calibration",
        "Compares predicted and oracle RRI means over time. The mean residual "
        "(bias) and the residual variance summarize systematic offset and "
        "stochastic error, respectively. Persistent bias indicates "
        "miscalibrated ordinal bins or feature scaling drift.",
    )
    stage_choice = st.selectbox("Stage", options=["train", "val"], index=0)
    pred_key = None
    oracle_key = None
    pred_key = _select_metric_key(
        pairs.get("pred_rri_mean", {}).get(stage_choice, {}),
        prefer_suffix,
    )
    oracle_key = _select_metric_key(
        pairs.get("rri_mean", {}).get(stage_choice, {}),
        prefer_suffix,
    )
    if pred_key and oracle_key:
        df_bias = history[[x_key, pred_key, oracle_key]].dropna()
        if not df_bias.empty:
            df_bias = df_bias.sort_values(x_key)
            residual = df_bias[pred_key] - df_bias[oracle_key]
            bias_mean = float(residual.mean())
            bias_abs = float(residual.abs().mean())
            bias_sq = float(bias_mean**2)
            var_resid = float(residual.var(ddof=0))
            mse_resid = float((residual**2).mean())

            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            col_b1.metric("Bias (mean)", f"{bias_mean:.4f}")
            col_b2.metric("|Bias| (mean)", f"{bias_abs:.4f}")
            col_b3.metric("Bias^2", f"{bias_sq:.4f}")
            col_b4.metric("Residual var", f"{var_resid:.4f}")

            fig_bias = go.Figure(
                go.Scatter(x=df_bias[x_key], y=residual, mode="lines"),
            )
            fig_bias.update_layout(
                title=_pretty_label(
                    f"{stage_choice} bias: pred_rri_mean - rri_mean",
                ),
                xaxis_title=_pretty_label(x_key),
                yaxis_title=_pretty_label("residual"),
            )
            st.plotly_chart(fig_bias, width="stretch")

            window = st.slider(
                "Variance window",
                min_value=5,
                max_value=200,
                value=25,
                step=5,
                key="wandb_bias_var_window",
            )
            roll_var = residual.rolling(window=window).var(ddof=0)
            roll_pred = df_bias[pred_key].rolling(window=window).var(ddof=0)
            roll_oracle = df_bias[oracle_key].rolling(window=window).var(ddof=0)
            fig_var = go.Figure()
            fig_var.add_trace(
                go.Scatter(
                    x=df_bias[x_key],
                    y=roll_var,
                    mode="lines",
                    name="Var(pred - oracle)",
                ),
            )
            fig_var.add_trace(
                go.Scatter(
                    x=df_bias[x_key],
                    y=roll_pred,
                    mode="lines",
                    name="Var(pred)",
                    line={"dash": "dot"},
                ),
            )
            fig_var.add_trace(
                go.Scatter(
                    x=df_bias[x_key],
                    y=roll_oracle,
                    mode="lines",
                    name="Var(oracle)",
                    line={"dash": "dot"},
                ),
            )
            fig_var.update_layout(
                title=_pretty_label(
                    f"{stage_choice} variance diagnostics (window={window})",
                ),
                xaxis_title=_pretty_label(x_key),
                yaxis_title=_pretty_label("variance"),
            )
            st.plotly_chart(fig_var, width="stretch")
            st.caption(f"Residual MSE: {mse_resid:.4f}")

            fig_scatter = px.scatter(
                df_bias,
                x=oracle_key,
                y=pred_key,
                title=_pretty_label(f"{stage_choice} predicted vs oracle RRI"),
                labels={
                    oracle_key: _pretty_label("oracle"),
                    pred_key: _pretty_label("predicted"),
                },
            )
            st.plotly_chart(fig_scatter, width="stretch")
    else:
        st.info("No paired pred_rri_mean and rri_mean metrics found.")

    st.subheader("Metric correlation")
    _info_popover(
        "correlation",
        "Pearson/Spearman correlations across metrics highlight coupling "
        "between optimization targets (loss, RRI, spearman). Strong off-diagonal "
        "structure can reveal redundant or conflicting objectives.",
    )
    corr_metrics = st.multiselect(
        "Metrics for correlation",
        options=numeric_cols,
        default=[m for m in numeric_cols if "loss" in m or "spearman" in m][:6],
    )
    corr_method = st.selectbox("Correlation method", options=["pearson", "spearman"])
    if corr_metrics:
        df_corr = history[corr_metrics].dropna()
        if not df_corr.empty:
            corr = df_corr.corr(method=corr_method)
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title=_pretty_label("Metric correlation"),
            )
            st.plotly_chart(fig_corr, width="stretch")

    st.subheader("Metric stability")
    _info_popover(
        "stability",
        "Rolling standard deviation quantifies short-term volatility. "
        "Spikes often indicate learning-rate changes or unstable gradients.",
    )
    if numeric_cols:
        volatility_metric = st.selectbox(
            "Volatility metric",
            options=numeric_cols,
            index=0,
        )
        window = st.slider(
            "Rolling window",
            min_value=5,
            max_value=200,
            value=25,
            step=5,
        )
        if volatility_metric:
            df_vol = history[[x_key, volatility_metric]].dropna()
            if not df_vol.empty:
                df_vol = df_vol.sort_values(x_key)
                roll_std = df_vol[volatility_metric].rolling(window=window).std()
                fig_vol = go.Figure(
                    go.Scatter(x=df_vol[x_key], y=roll_std, mode="lines"),
                )
                fig_vol.update_layout(
                    title=_pretty_label(
                        f"Rolling std ({window}) for {volatility_metric}",
                    ),
                    xaxis_title=_pretty_label(x_key),
                    yaxis_title=_pretty_label("std"),
                )
                st.plotly_chart(fig_vol, width="stretch")
    else:
        st.info("No numeric metrics available for stability analysis.")

    st.subheader("Logged confusion matrices & label histograms")
    _info_popover(
        "wandb media",
        "These images are logged by the Lightning module via W&B. "
        "They expose ordinal-class calibration and class imbalance over time.",
    )
    media_keys = [
        "train/confusion_matrix",
        "train/confusion_matrix_step",
        "val/confusion_matrix",
        "val/confusion_matrix_step",
        "train/label_histogram",
        "train/label_histogram_step",
        "val/label_histogram",
        "val/label_histogram_step",
    ]
    media_hist = _load_wandb_history(
        run,
        keys=media_keys,
        max_rows=int(media_rows),
    )
    cache_dir = PathConfig().wandb / "api_media" / str(getattr(run, "id", "run"))

    def _render_media_group(title: str, keys: list[str]) -> None:
        available = {key: _wandb_media_paths(media_hist, key) for key in keys}
        available = {key: paths for key, paths in available.items() if paths}
        if not available:
            st.info(f"No {title.lower()} images found in history.")
            return
        key_choice = st.selectbox(
            f"{title} key",
            options=list(available.keys()),
            index=0,
        )
        paths = available.get(key_choice, [])
        if not paths:
            st.info("No media paths available for this key.")
            return
        idx = st.slider(
            f"{title} index",
            min_value=0,
            max_value=max(0, len(paths) - 1),
            value=max(0, len(paths) - 1),
        )
        path = paths[int(idx)]
        local_path = _wandb_download_media(
            run,
            path=path,
            cache_dir=cache_dir,
        )
        if local_path and local_path.exists():
            st.image(str(local_path), caption=f"{key_choice} ({idx})")
        else:
            st.warning("Failed to download the selected media file.")

    col_left, col_right = st.columns(2)
    with col_left:
        _render_media_group(
            "Confusion matrix",
            [
                "train/confusion_matrix",
                "train/confusion_matrix_step",
                "val/confusion_matrix",
                "val/confusion_matrix_step",
            ],
        )
    with col_right:
        _render_media_group(
            "Label histogram",
            [
                "train/label_histogram",
                "train/label_histogram_step",
                "val/label_histogram",
                "val/label_histogram_step",
            ],
        )


def render_offline_stats_page() -> None:
    """Render offline cache statistics as a standalone page."""
    st.header("Offline Cache Statistics")
    st.caption(
        "Aggregate statistics over cached oracle batches without retaining full samples in memory.",
    )
    _info_popover(
        "offline stats",
        "Summaries are computed over cached oracle batches. This view helps "
        "validate label distributions, candidate counts, and backbone feature "
        "scales without loading full samples into memory.",
    )

    stats_key = "vin_offline_stats"
    stats_cache = st.session_state.get(stats_key, {})

    with st.sidebar.form("vin_offline_stats_form"):
        st.subheader("Offline stats")
        toml_path = st.text_input(
            "Experiment config TOML",
            value=str(Path(".configs") / "offline_only.toml"),
        )
        stage = st.selectbox(
            "Stage",
            options=[Stage.TRAIN, Stage.VAL, Stage.TEST],
            format_func=lambda s: s.value,
            key="vin_offline_stage",
        )
        cache_dir = st.text_input(
            "Offline cache dir",
            value=str(PathConfig().offline_cache_dir),
        )
        map_location = st.selectbox(
            "Cache map_location",
            options=["cpu", "cuda"],
            index=0,
            key="vin_offline_map_location",
        )
        max_samples = st.number_input(
            "Max samples (0 = all)",
            min_value=0,
            value=0,
            step=1,
            key="vin_offline_max_samples",
        )
        num_workers = st.number_input(
            "DataLoader workers (0 = use config)",
            min_value=0,
            value=0,
            step=1,
            key="vin_offline_num_workers",
        )
        train_val_split = st.number_input(
            "Train/val split",
            min_value=0.0,
            max_value=0.95,
            value=float(OracleRriCacheDatasetConfig().train_val_split),
            step=0.05,
            key="vin_offline_train_val_split",
        )
        run_stats = st.form_submit_button("Compute offline stats")

    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path is not None:

        def _count_index(path: Path) -> int:
            if not path.exists():
                return 0
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

        index_path = cache_path / "index.jsonl"
        train_index_path = cache_path / "train_index.jsonl"
        val_index_path = cache_path / "val_index.jsonl"
        samples_dir = cache_path / "samples"
        index_count = _count_index(index_path)
        train_count = _count_index(train_index_path)
        val_count = _count_index(val_index_path)
        sample_count = sum(1 for _ in samples_dir.glob("*.pt")) if samples_dir.exists() else 0
        if index_count or sample_count or train_count or val_count:
            counts_label = f"Index entries: {index_count} · Sample files: {sample_count}"
            if train_count or val_count:
                counts_label += f" · Train: {train_count} · Val: {val_count}"
            st.caption(counts_label)
        if sample_count > index_count:
            st.warning(
                "Cache index has fewer entries than sample files. "
                "Offline stats only read index.jsonl; rebuild the index to include all samples.",
            )
            split_seed = st.number_input(
                "Split RNG seed (-1 = random)",
                min_value=-1,
                value=-1,
                step=1,
                key="vin_offline_split_seed",
            )
            if st.button("Rebuild index from samples", key="vin_offline_rebuild_index"):
                with st.spinner("Rebuilding cache index..."):
                    rebuilt = rebuild_cache_index(
                        cache_dir=cache_path,
                        train_val_split=float(train_val_split),
                        rng_seed=None if split_seed < 0 else int(split_seed),
                    )
                st.success(f"Rebuilt index with {rebuilt} entries.")
                st.session_state.pop(stats_key, None)
                st.rerun()

    cfg_key = "|".join(
        [
            toml_path.strip(),
            stage.value,
            cache_dir.strip(),
            map_location,
            str(int(max_samples)),
            str(int(num_workers)),
            f"{float(train_val_split):.3f}",
        ],
    )

    if run_stats:
        try:
            with st.spinner("Collecting offline cache statistics..."):
                stats_cache = _collect_offline_cache_stats(
                    toml_path=toml_path.strip() or None,
                    stage=stage,
                    cache_dir=cache_dir.strip() or None,
                    map_location=map_location,
                    max_samples=int(max_samples),
                    num_workers=int(num_workers) if num_workers > 0 else None,
                    train_val_split=float(train_val_split),
                )
            stats_cache["key"] = cfg_key
            st.session_state[stats_key] = stats_cache
        except Exception as exc:  # pragma: no cover - UI guard
            _report_exception(exc, context="Offline stats failed")
            return

    if not stats_cache or stats_cache.get("key") != cfg_key:
        st.info("Run the offline stats to load summaries.")
        return

    summary = stats_cache["summary"]
    sample_df = stats_cache["sample_df"]
    backbone_df = stats_cache["backbone_df"]
    rri_values = stats_cache["rri_values"]
    pm_comp_after_values = stats_cache["pm_comp_after_values"]
    pm_acc_after_values = stats_cache["pm_acc_after_values"]
    num_valid_values = stats_cache["num_valid_values"]

    def _fmt(value: float) -> str:
        return f"{value:.4f}" if np.isfinite(value) else "n/a"

    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", summary["samples"])
    col2.metric("Total candidates", summary["total_candidates"])
    col3.metric("RRI mean", _fmt(summary["rri_mean"]))
    col4, col5, col6 = st.columns(3)
    col4.metric("RRI median", _fmt(summary["rri_median"]))
    col5.metric("pm_comp_after mean", _fmt(summary["pm_comp_after_mean"]))
    col6.metric("pm_acc_after mean", _fmt(summary["pm_acc_after_mean"]))

    log1p_counts = st.checkbox(
        "Log1p histogram counts",
        value=False,
        key="vin_offline_log1p",
    )

    if not sample_df.empty:
        st.subheader("Per-sample summary")
        st.dataframe(sample_df, width="stretch", height=240)

    with DEFAULT_PLOT_CFG.apply():
        st.subheader("Global RRI metrics")
        _info_popover(
            "rri distributions",
            "Histograms show the global distribution of RRI and its components across "
            "all cached candidates. Skewed or heavy-tailed distributions can indicate "
            "sampling bias or failure cases in the oracle pipeline.",
        )
        if rri_values:
            fig, ax = plt.subplots(figsize=(7, 3))
            _plot_hist_counts_mpl(
                rri_values,
                bins=60,
                log1p_counts=log1p_counts,
                ax=ax,
            )
            ax.set_title(_pretty_label("Oracle RRI distribution (all candidates)"))
            ax.set_xlabel(_pretty_label("RRI"))
            ax.set_ylabel(
                _pretty_label("log1p(count)" if log1p_counts else "count"),
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        if pm_comp_after_values:
            fig, ax = plt.subplots(figsize=(7, 3))
            _plot_hist_counts_mpl(
                pm_comp_after_values,
                bins=60,
                log1p_counts=log1p_counts,
                ax=ax,
            )
            ax.set_title(_pretty_label("pm_comp_after distribution (all candidates)"))
            ax.set_xlabel(_pretty_label("Mesh→point distance"))
            ax.set_ylabel(
                _pretty_label("log1p(count)" if log1p_counts else "count"),
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        if pm_acc_after_values:
            fig, ax = plt.subplots(figsize=(7, 3))
            _plot_hist_counts_mpl(
                pm_acc_after_values,
                bins=60,
                log1p_counts=log1p_counts,
                ax=ax,
            )
            ax.set_title(_pretty_label("pm_acc_after distribution (all candidates)"))
            ax.set_xlabel(_pretty_label("Point→mesh distance"))
            ax.set_ylabel(
                _pretty_label("log1p(count)" if log1p_counts else "count"),
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        st.subheader("RRI binning")
        _info_popover(
            "rri binning",
            "Fits quantile-based ordinal bins for CORAL training using the cached "
            "RRI distribution. Edges are recomputed from the selected offline "
            "samples; labels show how many candidates fall into each ordinal class.",
        )
        if rri_values:
            num_classes = int(
                st.slider(
                    "Num classes (K)",
                    min_value=2,
                    max_value=50,
                    value=15,
                    step=1,
                    key="vin_offline_binner_classes",
                ),
            )
            fit_binner = st.button(
                "Fit binner from offline RRI",
                key="vin_offline_binner_fit",
            )
            binner_key = f"{cfg_key}|{num_classes}"
            if st.session_state.get("vin_offline_binner_key") != binner_key or fit_binner:
                rri_tensor = torch.tensor(rri_values, dtype=torch.float32)
                binner = RriOrdinalBinner.fit_from_iterable(
                    [rri_tensor],
                    num_classes=num_classes,
                )
                labels = binner.transform(rri_tensor)
                st.session_state["vin_offline_binner_key"] = binner_key
                st.session_state["vin_offline_binner"] = binner
                st.session_state["vin_offline_binner_labels"] = labels

            binner = st.session_state.get("vin_offline_binner")
            labels = st.session_state.get("vin_offline_binner_labels")
            if binner is not None and labels is not None:
                fig, ax = plt.subplots(figsize=(8, 3.5))
                _plot_hist_counts_mpl(
                    rri_values,
                    bins=60,
                    log1p_counts=log1p_counts,
                    ax=ax,
                )
                for edge in binner.edges.detach().cpu().numpy().tolist():
                    ax.axvline(float(edge), color="black", linewidth=1.0, alpha=0.25)
                ax.set_title(_pretty_label("Raw oracle RRI + quantile edges"))
                ax.set_xlabel(_pretty_label("rri"))
                ax.set_ylabel(
                    _pretty_label("log1p(count)" if log1p_counts else "count"),
                )
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

                counts = (
                    torch.bincount(
                        labels.to(torch.int64),
                        minlength=int(binner.num_classes),
                    )
                    .cpu()
                    .numpy()
                )
                y_vals = np.log1p(counts) if log1p_counts else counts
                fig, ax = plt.subplots(figsize=(7, 3))
                sns.barplot(
                    x=np.arange(int(binner.num_classes)),
                    y=y_vals,
                    color="#285f82",
                    ax=ax,
                )
                ax.set_title(_pretty_label("Ordinal labels (K classes)"))
                ax.set_xlabel(_pretty_label("label"))
                ax.set_ylabel(
                    _pretty_label("log1p(count)" if log1p_counts else "count"),
                )
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
        else:
            st.info("No RRI values available to fit the binner.")

    if num_valid_values:
        st.subheader("Valid candidate counts")
        _info_popover(
            "valid counts",
            "Histogram of the number of valid candidates per snippet. Low counts "
            "often indicate aggressive rule filtering or challenging geometry.",
        )
        fig, ax = plt.subplots(figsize=(7, 3))
        _plot_hist_counts_mpl(
            num_valid_values,
            bins=30,
            log1p_counts=log1p_counts,
            ax=ax,
        )
        ax.set_title(_pretty_label("num_valid per snippet"))
        ax.set_xlabel(_pretty_label("num_valid"))
        ax.set_ylabel(
            _pretty_label("log1p(count)" if log1p_counts else "count"),
        )
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    candidate_offsets = stats_cache.get("candidate_offsets")
    candidate_yaw = stats_cache.get("candidate_yaw")
    candidate_pitch = stats_cache.get("candidate_pitch")
    candidate_roll = stats_cache.get("candidate_roll")
    candidate_rot_deg = stats_cache.get("candidate_rot_deg")
    if isinstance(candidate_offsets, np.ndarray) and candidate_offsets.size and candidate_offsets.shape[-1] == 3:
        st.subheader("Candidate pose distributions")
        _info_popover(
            "candidate distributions",
            "These distributions are computed over **all candidates** from the "
            "offline cache, expressed in the reference rig frame. Offsets are "
            "the candidate translations `t_rc`; yaw/pitch/roll are derived from "
            "the relative rotation `R_rc`; the rotation delta is the SO(3) angle "
            "between candidate and reference orientation (a measure of jitter).",
        )
        cand_bins = int(
            st.slider(
                "Candidate histogram bins",
                min_value=20,
                max_value=180,
                value=60,
                key="vin_offline_candidate_bins",
            ),
        )
        show_polar = st.checkbox(
            "Show azimuth/elevation heatmap",
            value=True,
            key="vin_offline_candidate_polar",
        )
        if show_polar:
            fig_polar = plot_position_polar(
                candidate_offsets,
                title=_pretty_label("Offset Azimuth/Elevation (Rig Frame)"),
                bins=cand_bins,
                fixed_ranges=True,
            )
            fig_polar.update_layout(
                title=_pretty_label("Offset Azimuth/Elevation (Rig Frame)"),
                xaxis_title=_pretty_label("azimuth (deg)"),
                yaxis_title=_pretty_label("elevation (deg)"),
            )
            st.plotly_chart(fig_polar, width="stretch")

        az = np.degrees(
            np.arctan2(candidate_offsets[:, 0], candidate_offsets[:, 2]),
        )
        el = np.degrees(
            np.arctan2(
                candidate_offsets[:, 1],
                np.linalg.norm(candidate_offsets[:, [0, 2]], axis=1) + 1e-8,
            ),
        )
        radius = np.linalg.norm(candidate_offsets, axis=1)
        col_a, col_b = st.columns(2)
        with col_a:
            fig_az = _histogram_overlay(
                [("azimuth", az)],
                bins=cand_bins,
                title="Offset azimuth distribution",
                xaxis_title="azimuth (deg)",
                log1p_counts=log1p_counts,
            )
            st.plotly_chart(fig_az, width="stretch")
        with col_b:
            fig_el = _histogram_overlay(
                [("elevation", el)],
                bins=cand_bins,
                title="Offset elevation distribution",
                xaxis_title="elevation (deg)",
                log1p_counts=log1p_counts,
            )
            st.plotly_chart(fig_el, width="stretch")

        fig_r = _histogram_overlay(
            [("radius", radius)],
            bins=cand_bins,
            title="Offset radius distribution",
            xaxis_title="radius (m)",
            log1p_counts=log1p_counts,
        )
        st.plotly_chart(fig_r, width="stretch")

        if (
            isinstance(candidate_yaw, np.ndarray)
            and isinstance(candidate_pitch, np.ndarray)
            and isinstance(candidate_roll, np.ndarray)
        ):
            fig_angles = _histogram_overlay(
                [
                    ("yaw", candidate_yaw),
                    ("pitch", candidate_pitch),
                    ("roll", candidate_roll),
                ],
                bins=cand_bins,
                title="Candidate orientation distribution (yaw/pitch/roll)",
                xaxis_title="angle (deg)",
                log1p_counts=log1p_counts,
            )
            st.plotly_chart(fig_angles, width="stretch")

        if isinstance(candidate_rot_deg, np.ndarray):
            fig_rot = _histogram_overlay(
                [("rotation_delta", candidate_rot_deg)],
                bins=cand_bins,
                title="Rotation delta distribution",
                xaxis_title="rotation delta (deg)",
                log1p_counts=log1p_counts,
            )
            st.plotly_chart(fig_rot, width="stretch")

    if not sample_df.empty:
        st.subheader("Scatter diagnostics")
        _info_popover(
            "scatter diagnostics",
            "Cross-plots relate mean RRI to accuracy/completeness and to the "
            "number of valid candidates. These plots help spot correlations and "
            "outliers in the oracle labels.",
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=sample_df,
            x="rri_mean",
            y="pm_comp_after_mean",
            ax=ax,
        )
        ax.set_title(_pretty_label("RRI mean vs pm_comp_after mean"))
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=sample_df,
            x="rri_mean",
            y="pm_acc_after_mean",
            ax=ax,
        )
        ax.set_title(_pretty_label("RRI mean vs pm_acc_after mean"))
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=sample_df, x="num_valid", y="rri_mean", ax=ax)
        ax.set_title(_pretty_label("num_valid vs RRI mean"))
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        if not backbone_df.empty:
            st.subheader("Backbone feature statistics")
            _info_popover(
                "backbone stats",
                "Per-field statistics of EVL backbone outputs. High variance or "
                "extreme sparsity can indicate scale mismatches or missing modalities.",
            )
            field_options = sorted(backbone_df["field"].dropna().unique().tolist())
            selected_fields = st.multiselect(
                "Backbone fields",
                options=field_options,
                default=field_options,
                key="vin_offline_backbone_fields",
            )
            if selected_fields:
                backbone_df = backbone_df.query("field in @selected_fields")
            if backbone_df.empty:
                st.info("No backbone fields selected.")
                return
            metric_cols = [col for col in ["mean", "std", "abs_mean", "nz_frac", "numel"] if col in backbone_df.columns]
            selected_cols = st.multiselect(
                "Backbone stats columns",
                options=metric_cols,
                default=metric_cols,
                key="vin_offline_backbone_cols",
            )
            if not selected_cols:
                st.info("Select at least one metric column to summarize.")
                return
            summary_df = backbone_df.groupby("field")[selected_cols].mean().reset_index()
            sort_metric = "std" if "std" in selected_cols else selected_cols[0]
            sort_choice = st.selectbox(
                "Sort metric",
                options=selected_cols,
                index=selected_cols.index(sort_metric),
                key="vin_offline_backbone_sort",
            )
            summary_df = summary_df.sort_values(sort_choice, ascending=False)
            st.dataframe(summary_df, width="stretch", height=280)

            plot_metric = st.selectbox(
                "Plot metric",
                options=selected_cols,
                index=selected_cols.index(sort_metric),
                key="vin_offline_backbone_plot_metric",
            )
            top_df = summary_df.head(10)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=top_df, x=plot_metric, y="field", ax=ax)
            ax.set_title(
                _pretty_label(f"Mean feature {plot_metric} (top 10)"),
            )
            ax.set_xlabel(_pretty_label(plot_metric))
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)


__all__ = [
    "render_candidates_page",
    "render_data_page",
    "render_depth_page",
    "render_rri_binning_page",
    "render_offline_stats_page",
    "render_rri_page",
    "render_vin_diagnostics_page",
    "render_wandb_analysis_page",
]
