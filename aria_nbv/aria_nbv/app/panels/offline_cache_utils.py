"""Offline cache helpers for panels."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch

from ...configs import PathConfig
from ...data import AseEfmDatasetConfig, EfmSnippetView, VinOracleCacheDatasetConfig
from ...data.offline_cache import OracleRriCacheConfig, OracleRriCacheDataset, OracleRriCacheDatasetConfig
from ...data.offline_cache_coverage import read_cache_index_entries
from ...data.vin_snippet_cache import VinSnippetCacheConfig, VinSnippetCacheDatasetConfig
from ...lightning.aria_nbv_experiment import AriaNBVExperimentConfig
from ...utils import Stage
from ..state_types import config_signature

BACKBONE_KEEP_FIELDS_FOR_STATS = [
    "t_world_voxel",
    "voxel_extent",
    "occ_pr",
    "occ_input",
    "counts",
    "cent_pr",
    "free_input",
    "pts_world",
]


def _tensor_nbytes(value: torch.Tensor) -> int:
    return int(value.numel()) * int(value.element_size())


def _estimate_nbytes(value: Any, *, _seen: set[int] | None = None) -> int:
    """Best-effort estimate of the memory footprint for nested tensor containers."""
    if value is None:
        return 0

    if _seen is None:
        _seen = set()

    obj_id = id(value)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)

    if torch.is_tensor(value):
        return _tensor_nbytes(value)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)

    tensor_fn = getattr(value, "tensor", None)
    if callable(tensor_fn):
        try:
            tensor = tensor_fn()
        except Exception:
            tensor = None
        if torch.is_tensor(tensor):
            return _tensor_nbytes(tensor)

    if is_dataclass(value):
        return sum(_estimate_nbytes(getattr(value, field.name), _seen=_seen) for field in fields(value))

    if isinstance(value, dict):
        return sum(_estimate_nbytes(item, _seen=_seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_estimate_nbytes(item, _seen=_seen) for item in value)

    if hasattr(value, "__dict__"):
        return sum(_estimate_nbytes(item, _seen=_seen) for item in vars(value).values())

    return 0


def _p3d_cameras_nbytes(value: Any) -> int:
    """Estimate size of the commonly-used PerspectiveCameras tensor fields."""
    if value is None:
        return 0
    total = 0
    for name in ("R", "T", "focal_length", "principal_point", "image_size"):
        tensor = getattr(value, name, None)
        if torch.is_tensor(tensor):
            total += _tensor_nbytes(tensor)
    return total


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
    paths: PathConfig,
    state: Any,
    stage: Stage | None,
    include_efm_snippet: bool,
    include_gt_mesh: bool,
) -> OracleRriCacheDataset | None:
    _ = (include_efm_snippet, include_gt_mesh)
    if cache_dir is None:
        return None
    split = "all"
    if stage is Stage.TRAIN:
        split = "train"
    elif stage in (Stage.VAL, Stage.TEST):
        split = "val"
    # Keep cache datasets snippet-free to avoid re-instantiation when toggling
    # attach-snippet controls; snippets are loaded on demand via helpers below.
    cache_cfg = OracleRriCacheDatasetConfig(
        cache=OracleRriCacheConfig(cache_dir=Path(cache_dir), paths=paths),
        load_backbone=True,
        split=split,
        include_efm_snippet=False,
        include_gt_mesh=False,
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
            left0_norm[degenerate] = alt_norm[degenerate]
        left0 = left0 / left0_norm.clamp_min(eps)

        up0 = torch.cross(forward, left0, dim=-1)
        up0 = _normalise(up0, eps=eps)
        up_cam = _normalise(up_cam, eps=eps)
        cosang = (up0 * up_cam).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cosang)
        sign = torch.sign((left0 * up_cam).sum(dim=-1))
        return angle * sign

    def _broadcast_ref_pose(
        ref_rot: torch.Tensor,
        ref_t: torch.Tensor,
        target_rot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast reference pose to match target leading dimensions."""
        if ref_rot.ndim == 2:
            ref_rot = ref_rot.unsqueeze(0)
        if ref_t.ndim == 1:
            ref_t = ref_t.unsqueeze(0)

        target_shape = target_rot.shape[:-2]
        ref_shape = ref_rot.shape[:-2]

        if len(ref_shape) < len(target_shape):
            pad = (1,) * (len(target_shape) - len(ref_shape))
            ref_rot = ref_rot.reshape(ref_shape + pad + (3, 3))
            ref_t = ref_t.reshape(ref_t.shape[:-1] + pad + (3,))
            ref_shape = ref_rot.shape[:-2]

        if len(ref_shape) != len(target_shape):
            raise ValueError(
                f"reference_pose_world_rig has incompatible batch dims {ref_shape} for target {target_shape}.",
            )

        expanded_shape = []
        for ref_dim, target_dim in zip(ref_shape, target_shape, strict=True):
            if ref_dim not in (1, target_dim):
                raise ValueError(
                    f"reference_pose_world_rig has incompatible batch dims {ref_shape} for target {target_shape}.",
                )
            expanded_shape.append(target_dim if ref_dim == 1 else ref_dim)

        ref_rot = ref_rot.expand(*expanded_shape, 3, 3)
        ref_t = ref_t.expand(*expanded_shape, 3)
        return ref_rot, ref_t

    def _as_tensor(value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            return value
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value)
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

    resolved_toml: Path | None = None
    if toml_path:
        try:
            resolved = PathConfig().resolve_config_toml_path(
                toml_path,
                must_exist=False,
            )
        except ValueError as exc:
            st.warning(f"Invalid config path ({toml_path}): {exc}")
        else:
            if not resolved.exists():
                st.warning(f"Config file not found: {resolved}")
            else:
                resolved_toml = resolved

    cfg = AriaNBVExperimentConfig.from_toml(resolved_toml) if resolved_toml is not None else AriaNBVExperimentConfig()
    cfg.run_mode = "summarize_vin"
    cfg.stage = stage
    cfg.trainer_config.use_wandb = False

    dm_cfg = cfg.datamodule_config

    paths = cfg.paths if isinstance(cfg.paths, PathConfig) else PathConfig()
    cache_root = cache_dir or str(
        paths.offline_cache_dir or (paths.data_root / "oracle_rri_cache"),
    )
    cache_cfg = OracleRriCacheDatasetConfig(
        cache=OracleRriCacheConfig(cache_dir=Path(cache_root), paths=paths),
        load_backbone=True,
        backbone_keep_fields=BACKBONE_KEEP_FIELDS_FOR_STATS,
        train_val_split=train_val_split,
    )
    dm_cfg.source = VinOracleCacheDatasetConfig(
        cache=cache_cfg,
        train_split="train",
        val_split="val",
    )
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
    mem_backbone_bytes: list[int] = []
    mem_rri_bytes: list[int] = []
    mem_vin_snippet_bytes: list[int] = []
    mem_pose_camera_bytes: list[int] = []
    mem_total_bytes: list[int] = []
    candidate_offsets: list[np.ndarray] = []
    candidate_yaw: list[np.ndarray] = []
    candidate_pitch: list[np.ndarray] = []
    candidate_roll: list[np.ndarray] = []
    candidate_rot_deg: list[np.ndarray] = []

    max_batches = None if max_samples in (None, 0) else int(max_samples)
    raw_shapes: dict[str, str] | None = None
    padded_shapes: dict[str, str] | None = None
    dataset = getattr(dataloader, "dataset", None)
    if isinstance(dataset, torch.utils.data.Dataset):
        try:
            raw_batch = dataset[0]
        except Exception:
            raw_batch = None
        if raw_batch is not None and hasattr(raw_batch, "shape_summary"):
            raw_shapes = raw_batch.shape_summary()

    progress = st.progress(0.0)
    for idx, batch in enumerate(dataloader):
        if max_batches is not None and idx >= max_batches:
            break
        if idx == 0 and hasattr(batch, "shape_summary"):
            padded_shapes = batch.shape_summary()

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

        backbone_bytes = _estimate_nbytes(batch.backbone_out)
        rri_bytes = sum(
            _estimate_nbytes(getattr(batch, name, None))
            for name in (
                "rri",
                "pm_dist_before",
                "pm_dist_after",
                "pm_acc_before",
                "pm_comp_before",
                "pm_acc_after",
                "pm_comp_after",
            )
        )
        vin_snippet_bytes = _estimate_nbytes(getattr(batch, "efm_snippet_view", None))
        pose_camera_bytes = (
            _estimate_nbytes(batch.candidate_poses_world_cam)
            + _estimate_nbytes(batch.reference_pose_world_rig)
            + _p3d_cameras_nbytes(batch.p3d_cameras)
        )
        total_bytes = backbone_bytes + rri_bytes + vin_snippet_bytes + pose_camera_bytes
        mem_backbone_bytes.append(backbone_bytes)
        mem_rri_bytes.append(rri_bytes)
        mem_vin_snippet_bytes.append(vin_snippet_bytes)
        mem_pose_camera_bytes.append(pose_camera_bytes)
        mem_total_bytes.append(total_bytes)

        mib = float(1024**2)
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
                "mem_backbone_mib": float(backbone_bytes) / mib,
                "mem_rri_mib": float(rri_bytes) / mib,
                "mem_vin_snippet_mib": float(vin_snippet_bytes) / mib,
                "mem_pose_camera_mib": float(pose_camera_bytes) / mib,
                "mem_total_mib": float(total_bytes) / mib,
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
            r_wr, t_wr = _broadcast_ref_pose(r_wr, t_wr, r_wc)

            r_rw = r_wr.transpose(-1, -2)
            t_rw = -(r_rw @ t_wr.unsqueeze(-1)).squeeze(-1)
            r_rc = r_rw @ r_wc
            t_rc = t_rw + (r_rw @ t_wc.unsqueeze(-1)).squeeze(-1)

            if t_rc.ndim > 2:
                t_rc_flat = t_rc.reshape(-1, 3)
                r_rc_flat = r_rc.reshape(-1, 3, 3)
            else:
                t_rc_flat = t_rc
                r_rc_flat = r_rc

            candidate_offsets.append(t_rc_flat.detach().cpu().numpy())

            fwd = r_rc_flat[:, :, 2]
            up = r_rc_flat[:, :, 1]
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

            trace = r_rc_flat[:, 0, 0] + r_rc_flat[:, 1, 1] + r_rc_flat[:, 2, 2]
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

    def _summarize_bytes(values: list[int]) -> dict[str, float]:
        if not values:
            return {"mean_mib": float("nan"), "median_mib": float("nan"), "p95_mib": float("nan")}
        arr = np.asarray(values, dtype=np.float64)
        mib = float(1024**2)
        return {
            "mean_mib": float(arr.mean() / mib),
            "median_mib": float(np.median(arr) / mib),
            "p95_mib": float(np.percentile(arr, 95) / mib),
        }

    mem_summary = {
        "backbone": _summarize_bytes(mem_backbone_bytes),
        "rri": _summarize_bytes(mem_rri_bytes),
        "vin_snippet": _summarize_bytes(mem_vin_snippet_bytes),
        "pose_camera": _summarize_bytes(mem_pose_camera_bytes),
        "total": _summarize_bytes(mem_total_bytes),
    }
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
        "memory_summary": mem_summary,
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
        "batch_shapes": {
            "raw": raw_shapes,
            "padded": padded_shapes,
        },
    }


def _collect_vin_snippet_cache_stats(
    *,
    cache_dir: str | None,
    map_location: str,
    max_samples: int | None,
    num_workers: int | None,
) -> dict[str, Any]:
    """Collect summary stats from a VIN snippet cache."""
    paths = PathConfig()
    cache_root = cache_dir or str(
        (paths.offline_cache_dir or (paths.data_root / "oracle_rri_cache")) / "vin_snippet_cache",
    )
    cache_cfg = VinSnippetCacheDatasetConfig(
        cache=VinSnippetCacheConfig(cache_dir=Path(cache_root), paths=paths),
        map_location=map_location,
        limit=None if max_samples in (None, 0) else int(max_samples),
    )
    dataset = cache_cfg.setup_target()

    entries = read_cache_index_entries(cache_cfg.cache.index_path)
    if cache_cfg.limit is not None:
        entries = entries[: int(cache_cfg.limit)]

    def _collate(items: list[Any]) -> Any:
        return items[0]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(num_workers or 0),
        persistent_workers=bool(num_workers),
        collate_fn=_collate,
    )

    def _finite_1d(values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values.reshape(0)
        flat = values.reshape(-1)
        return flat[torch.isfinite(flat)]

    def _point_stats(
        values: torch.Tensor,
        *,
        max_quantile_samples: int = 200_000,
        rng: torch.Generator | None = None,
    ) -> dict[str, float]:
        if values.numel() == 0:
            return {
                "count": 0.0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
            }
        vals = values.to(dtype=torch.float32)
        q_vals = vals
        if vals.numel() > max_quantile_samples:
            if rng is None:
                rng = torch.Generator(device=vals.device)
                rng.manual_seed(0)
            idx = torch.randint(
                low=0,
                high=vals.numel(),
                size=(max_quantile_samples,),
                generator=rng,
                device=vals.device,
            )
            q_vals = vals[idx]
        if vals.numel() == 1:
            p50 = float(vals.item())
            p95 = p50
        else:
            qs = torch.quantile(
                q_vals,
                torch.tensor([0.5, 0.95], device=q_vals.device, dtype=q_vals.dtype),
            )
            p50 = float(qs[0].item())
            p95 = float(qs[1].item())
        return {
            "count": float(vals.numel()),
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()),
            "min": float(vals.min().item()),
            "max": float(vals.max().item()),
            "p50": p50,
            "p95": p95,
        }

    sample_rows: list[dict[str, Any]] = []
    points_counts: list[int] = []
    traj_lengths: list[int] = []
    inv_std_values: list[float] = []
    obs_count_values: list[float] = []
    snippet_shapes: dict[str, str] | None = None
    has_obs_count = False

    progress = st.progress(0.0)
    total = len(dataset)
    for idx, snippet in enumerate(loader):
        if total:
            progress.progress(min(1.0, float(idx + 1) / float(total)))
        points_world = snippet.points_world
        traj = snippet.t_world_rig
        if idx == 0 and snippet_shapes is None:
            snippet_shapes = {
                "vin_snippet.points_world": str(tuple(points_world.shape)),
                "vin_snippet.t_world_rig": str(tuple(traj.tensor().shape)),
            }
        points_count = int(points_world.shape[0])
        traj_len = int(traj.shape[0])
        points_counts.append(points_count)
        traj_lengths.append(traj_len)

        if points_count > 0 and points_world.shape[-1] > 3:
            inv_vals = _finite_1d(points_world[:, 3])
            if inv_vals.numel() > 0:
                inv_std_values.extend(inv_vals.detach().cpu().tolist())

        if points_count > 0 and points_world.shape[-1] > 4:
            has_obs_count = True
            obs_vals = _finite_1d(points_world[:, 4])
            if obs_vals.numel() > 0:
                obs_count_values.extend(obs_vals.detach().cpu().tolist())

        scene_id = entries[idx].scene_id if idx < len(entries) else ""
        snippet_id = entries[idx].snippet_id if idx < len(entries) else ""
        sample_rows.append(
            {
                "scene_id": scene_id,
                "snippet_id": snippet_id,
                "points_count": points_count,
                "traj_len": traj_len,
            },
        )
    progress.empty()

    def _safe_mean(values: list[int | float]) -> float:
        if not values:
            return float("nan")
        return float(np.mean(values))

    def _safe_median(values: list[int | float]) -> float:
        if not values:
            return float("nan")
        return float(np.median(values))

    def _safe_max(values: list[int | float]) -> float:
        if not values:
            return float("nan")
        return float(np.max(values))

    inv_stats = (
        _point_stats(torch.tensor(inv_std_values), rng=torch.Generator().manual_seed(0))
        if inv_std_values
        else _point_stats(torch.empty(0))
    )
    obs_stats = (
        _point_stats(torch.tensor(obs_count_values), rng=torch.Generator().manual_seed(0))
        if obs_count_values
        else _point_stats(torch.empty(0))
    )

    summary = {
        "samples": len(points_counts),
        "points_mean": _safe_mean(points_counts),
        "points_median": _safe_median(points_counts),
        "points_max": _safe_max(points_counts),
        "traj_len_mean": _safe_mean(traj_lengths),
        "traj_len_median": _safe_median(traj_lengths),
        "inv_dist_std_mean": inv_stats["mean"],
        "inv_dist_std_std": inv_stats["std"],
        "inv_dist_std_p95": inv_stats["p95"],
        "inv_dist_std_min": inv_stats["min"],
        "inv_dist_std_max": inv_stats["max"],
        "obs_count_mean": obs_stats["mean"],
        "obs_count_std": obs_stats["std"],
        "obs_count_p95": obs_stats["p95"],
        "obs_count_min": obs_stats["min"],
        "obs_count_max": obs_stats["max"],
    }

    return {
        "summary": summary,
        "sample_df": pd.DataFrame(sample_rows),
        "points_counts": points_counts,
        "traj_lengths": traj_lengths,
        "inv_dist_std": inv_std_values,
        "obs_count": obs_count_values,
        "has_obs_count": has_obs_count,
        "snippet_shapes": snippet_shapes,
    }


def _collect_vin_batch_shape_preview(
    *,
    toml_path: str | None,
    stage: Stage,
    oracle_cache_dir: str | None,
    vin_cache_dir: str | None,
    train_val_split: float,
    num_workers: int | None,
) -> dict[str, dict[str, str] | None]:
    """Collect a lightweight shape preview for VIN batches."""
    if oracle_cache_dir is None or vin_cache_dir is None:
        return {"raw": None, "padded": None}

    resolved_toml: Path | None = None
    if toml_path:
        try:
            resolved = PathConfig().resolve_config_toml_path(
                toml_path,
                must_exist=False,
            )
        except ValueError:
            resolved = None
        else:
            if resolved.exists():
                resolved_toml = resolved

    cfg = AriaNBVExperimentConfig.from_toml(resolved_toml) if resolved_toml is not None else AriaNBVExperimentConfig()
    cfg.run_mode = "summarize_vin"
    cfg.stage = stage
    cfg.trainer_config.use_wandb = False

    dm_cfg = cfg.datamodule_config
    dm_cfg.source = VinOracleCacheDatasetConfig(
        cache=OracleRriCacheDatasetConfig(
            cache=OracleRriCacheConfig(cache_dir=Path(oracle_cache_dir), paths=cfg.paths),
            load_backbone=True,
            backbone_keep_fields=BACKBONE_KEEP_FIELDS_FOR_STATS,
            train_val_split=train_val_split,
            vin_snippet_cache=VinSnippetCacheConfig(cache_dir=Path(vin_cache_dir), paths=cfg.paths),
            vin_snippet_cache_mode="auto",
        ),
        train_split="train",
        val_split="val",
    )
    if num_workers is not None and num_workers > 0:
        dm_cfg.num_workers = int(num_workers)
    else:
        dm_cfg.num_workers = 0

    datamodule = dm_cfg.setup_target()
    dataloader = datamodule.train_dataloader() if stage is Stage.TRAIN else datamodule.val_dataloader()

    raw_shapes: dict[str, str] | None = None
    padded_shapes: dict[str, str] | None = None

    dataset = getattr(dataloader, "dataset", None)
    if isinstance(dataset, torch.utils.data.Dataset):
        try:
            raw_batch = dataset[0]
        except Exception:
            raw_batch = None
        if raw_batch is not None and hasattr(raw_batch, "shape_summary"):
            raw_shapes = raw_batch.shape_summary()

    try:
        batch = next(iter(dataloader))
    except StopIteration:
        batch = None
    if batch is not None and hasattr(batch, "shape_summary"):
        padded_shapes = batch.shape_summary()

    return {
        "raw": raw_shapes,
        "padded": padded_shapes,
    }


__all__ = [
    "_collect_offline_cache_stats",
    "_collect_vin_snippet_cache_stats",
    "_collect_vin_batch_shape_preview",
    "_load_efm_snippet_for_cache",
    "_prepare_offline_cache_dataset",
]
