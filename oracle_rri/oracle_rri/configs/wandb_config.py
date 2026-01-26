"""W&B configuration and helper utilities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field
from pytorch_lightning.loggers import WandbLogger

from ..utils import BaseConfig
from .path_config import PathConfig

if TYPE_CHECKING:
    import pandas as pd


class WandbConfig(BaseConfig):
    """Wrapper around Lightning's `WandbLogger`.

    References:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html
    """

    target: type[WandbLogger] = Field(default_factory=lambda: WandbLogger, exclude=True)

    name: str | None = Field(default=None, description="Display name for the run.")
    project: str = Field(default="aria-nbv", description="W&B project name.")
    entity: str | None = None
    offline: bool = Field(False, description="Enable offline logging.")
    log_model: bool | str = Field(
        default=False,
        description="Forward Lightning checkpoints to W&B artefacts.",
    )
    checkpoint_name: str | None = Field(default=None, description="Checkpoint artefact name.")
    tags: list[str] | None = Field(default=None, description="Optional list of tags.")
    group: str | None = Field(default=None, description="Group multiple related runs.")
    job_type: str | None = Field(default=None, description="Attach a W&B job_type label.")
    prefix: str | None = Field(default=None, description="Namespace prefix for metric keys.")

    def setup_target(self, **kwargs: Any) -> WandbLogger:
        """Instantiate a configured `WandbLogger`."""

        wandb_dir = PathConfig().wandb.as_posix()

        return WandbLogger(
            name=self.name,
            project=self.project,
            entity=self.entity,
            save_dir=wandb_dir,
            offline=self.offline,
            log_model=self.log_model,
            prefix=self.prefix,
            checkpoint_name=self.checkpoint_name,
            tags=self.tags,
            group=self.group,
            job_type=self.job_type,
            **(kwargs or {}),
        )


WANDB_STEP_KEYS: tuple[str, ...] = ("trainer/global_step", "global_step", "_step", "epoch")


def _flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten nested mappings to dot-separated keys, skipping private keys."""

    def _normalize_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, bool, int, float, np.number)):
            return value
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value)
        return str(value)

    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        key_str = str(key)
        if key_str.startswith("_"):
            continue
        full_key = f"{prefix}{sep}{key_str}" if prefix else key_str
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, prefix=full_key, sep=sep))
        else:
            flat[full_key] = _normalize_value(value)
    return flat


def _safe_mapping(raw: Any) -> dict[str, Any]:
    """Safely coerce a W&B mapping-like object to a dict."""
    if raw is None:
        return {}
    try:
        return dict(raw)
    except Exception:  # pragma: no cover - defensive guard
        return {}


def _list_entities(api: Any) -> list[str]:
    """Return available entities (user + teams) for the current API token."""
    entities: list[str] = []
    try:
        viewer = api.viewer()
    except Exception:  # pragma: no cover - API guard
        viewer = None
    if viewer is None:
        return []
    for attr in ("entity", "username", "name"):
        value = getattr(viewer, attr, None)
        if isinstance(value, str) and value:
            entities.append(value)
    teams = getattr(viewer, "teams", None)
    if teams:
        for team in teams:
            name = getattr(team, "name", None)
            if isinstance(name, str) and name:
                entities.append(name)
    return list(dict.fromkeys(entities))


def _list_projects(api: Any, *, entity: str) -> list[str]:
    """Return available projects for a given entity."""
    projects: list[str] = []
    if not entity:
        return projects
    try:
        iterator = api.projects(entity)
    except Exception:  # pragma: no cover - API guard
        return projects
    for project in iterator:
        name = getattr(project, "name", None)
        if isinstance(name, str) and name:
            projects.append(name)
        elif isinstance(project, dict):
            value = project.get("name")
            if isinstance(value, str) and value:
                projects.append(value)
    return list(dict.fromkeys(projects))


def _list_runs(
    api: Any,
    *,
    entity: str,
    project: str,
    max_runs: int,
) -> list[Any]:
    """Fetch up to max_runs from W&B, ordered by recency when supported."""
    try:
        iterator = api.runs(f"{entity}/{project}", order="-created_at")
    except TypeError:  # pragma: no cover - older API versions
        iterator = api.runs(f"{entity}/{project}")
    runs: list[Any] = []
    for run in iterator:
        runs.append(run)
        if max_runs and len(runs) >= max_runs:
            break
    return runs


def _extract_run_steps(run: Any, *, keys: tuple[str, ...] = WANDB_STEP_KEYS) -> float | None:
    """Extract the best-available step count from a run summary."""
    summary = _safe_mapping(getattr(run, "summary", None))
    for key in keys:
        value = summary.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _filter_runs(
    runs: list[Any],
    *,
    name_regex: re.Pattern[str] | None,
    states: list[str],
    tags: set[str],
    group: str,
    job_type: str,
    min_steps: float | None,
    max_steps: float | None,
) -> list[Any]:
    """Filter runs by state, regex, tags, group, job_type, and steps."""
    filtered: list[Any] = []
    for run in runs:
        state = str(getattr(run, "state", "") or "")
        if states and state not in states:
            continue
        name = str(getattr(run, "name", "") or "")
        run_id = str(getattr(run, "id", "") or "")
        if name_regex and not (name_regex.search(name) or name_regex.search(run_id)):
            continue
        run_group = str(getattr(run, "group", "") or "")
        if group and run_group != group:
            continue
        run_job_type = str(getattr(run, "job_type", "") or "")
        if job_type and run_job_type != job_type:
            continue
        run_tags = set(getattr(run, "tags", []) or [])
        if tags and not (run_tags & tags):
            continue
        if min_steps is not None or max_steps is not None:
            steps = _extract_run_steps(run)
            if steps is None:
                continue
            if min_steps is not None and steps < min_steps:
                continue
            if max_steps is not None and steps > max_steps:
                continue
        filtered.append(run)
    return filtered


def _load_wandb_history(
    run: Any,
    *,
    keys: list[str] | None,
    max_rows: int,
) -> "pd.DataFrame":
    import pandas as pd

    history = run.history(keys=keys, samples=int(max_rows))
    if isinstance(history, pd.DataFrame):
        return history
    return pd.DataFrame(history)


def _metric_pairs_with_pattern(
    columns: list[str],
    *,
    pattern: str,
) -> dict[str, dict[str, dict[str, str]]]:
    pairs: dict[str, dict[str, dict[str, str]]] = {}
    regex = re.compile(pattern)
    for name in columns:
        match = regex.match(name)
        if not match:
            continue
        stage, base, suffix = match.groups()
        suffix = suffix or "raw"
        pairs.setdefault(base, {}).setdefault(stage, {})[suffix] = name
    return pairs


def _metric_pairs(columns: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    return _metric_pairs_with_pattern(
        columns,
        pattern=r"^(train|val)/(.+?)(?:_(step|epoch))?$",
    )


def _select_metric_key(stage_map: dict[str, str], prefer: str) -> str | None:
    if prefer in stage_map:
        return stage_map[prefer]
    if "raw" in stage_map:
        return stage_map["raw"]
    if stage_map:
        return next(iter(stage_map.values()))
    return None


__all__ = [
    "WandbConfig",
    "WANDB_STEP_KEYS",
    "_extract_run_steps",
    "_filter_runs",
    "_flatten_mapping",
    "_list_entities",
    "_list_projects",
    "_list_runs",
    "_load_wandb_history",
    "_metric_pairs",
    "_metric_pairs_with_pattern",
    "_safe_mapping",
    "_select_metric_key",
]
