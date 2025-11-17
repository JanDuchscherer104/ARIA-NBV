"""Minimal ATEK WebDataset wrapper with GT mesh insertion and typed view."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import trimesh
from atek.data_loaders.atek_wds_dataloader import (
    load_atek_wds_dataset,
)
from pydantic import Field, ValidationInfo, field_validator, model_validator
from torch.utils.data import IterableDataset

from ..configs import PathConfig
from ..utils import BaseConfig, Console
from .views import TypedSample

SEQ_PATTERN = re.compile(r".*?(?P<scene_id>\d{4,})(?:_|-)(?P<snippet_id>[\w-]+)")


def _parse_sequence_name(sequence_name: str) -> tuple[str, str]:
    if not sequence_name:
        return "unknown", "unknown"
    parts = sequence_name.split("_")
    if len(parts) >= 3 and parts[-2].isdigit():
        return parts[-2], parts[-1]
    match = SEQ_PATTERN.match(sequence_name)
    if match:
        return match.group("scene_id"), match.group("snippet_id")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return sequence_name, "unknown"


def _infer_ids(flat_dict: Mapping[str, Any], sequence_name: str) -> tuple[str, str]:
    scene_id, snippet_id = _parse_sequence_name(sequence_name)
    if snippet_id != "unknown":
        return scene_id, snippet_id
    key = flat_dict.get("__key__")
    if isinstance(key, str):
        part = key.split("_")[-1]
        if part:
            snippet_id = part
    url = flat_dict.get("__url__")
    if isinstance(url, str):
        stem = Path(url).stem
        if stem:
            snippet_id = stem
        parent = Path(url).parent.name
        if parent.isdigit():
            scene_id = parent
    return scene_id, snippet_id


def _explode_batched_dict(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split a batched dict from WebDataset into per-item dicts.

    Supports leading-batch tensors and lists; non-batched items are copied to
    each output dict. This mirrors ATEK/efm3d loader behaviour where each
    field shares the same batch dimension.
    """

    batch_size: int | None = None
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            batch_size = value.shape[0]
            break
        if isinstance(value, list):
            batch_size = len(value)
            break
    if batch_size is None:
        return [dict(batch)]
    per_item: list[dict[str, Any]] = []
    for idx in range(batch_size):
        item: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                item[key] = value[idx]
            elif isinstance(value, list):
                item[key] = value[idx]
            else:
                item[key] = value
        per_item.append(item)
    return per_item


def ase_collate(batch: Sequence[TypedSample]) -> dict[str, Any]:
    return {
        "scene_id": [s.scene_id for s in batch],
        "snippet_id": [s.snippet_id for s in batch],
        "atek": list(batch),
        "efm": [s.to_efm_dict() for s in batch],
        "gt_mesh": [s.mesh for s in batch],
    }


class ASEDataset(IterableDataset[TypedSample]):
    """Iterable dataset yielding TypedSample with optional GT mesh pairing."""

    def __init__(self, config: ASEDatasetConfig):
        super().__init__()
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbose(config.verbose)

        self.console.log(f"Loading ATEK WebDataset from {len(config.tar_urls)} shards")
        self._atek_wds = load_atek_wds_dataset(
            urls=config.tar_urls,
            batch_size=config.batch_size,
            shuffle_flag=config.shuffle,
            repeat_flag=config.repeat,
        )
        self._mesh_cache: dict[str, trimesh.Trimesh] = {}
        self.console.log("ATEK loader ready")

    # How can we vectorize this for batching?
    def _load_mesh(self, scene_id: str) -> trimesh.Trimesh | None:
        if scene_id in self._mesh_cache:
            return self._mesh_cache[scene_id]
        mesh_path = self.config.scene_to_mesh.get(scene_id)
        if mesh_path is None or not mesh_path.exists():
            if self.config.require_mesh:
                raise FileNotFoundError(f"GT mesh for scene {scene_id} not found (require_mesh=True).")
            return None
        mesh = trimesh.load(str(mesh_path), process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Mesh for scene {scene_id} is not Trimesh")
        if self.config.mesh_simplify_ratio is not None:
            mesh = mesh.simplify_quadric_decimation(self.config.mesh_simplify_ratio)
            self.console.dbg(f"Simplified mesh {scene_id}")
        if self.config.cache_meshes:
            self._mesh_cache[scene_id] = mesh
        return mesh

    def _iter_flat_samples(self) -> Iterator[dict[str, Any]]:
        for raw in self._atek_wds:
            if isinstance(raw, Mapping):
                samples = _explode_batched_dict(raw) if self.config.batch_size else [dict(raw)]
                for sample in samples:
                    yield sample
            elif isinstance(raw, (list, tuple)):
                for sample in raw:
                    if not isinstance(sample, Mapping):
                        raise TypeError(f"Unexpected sample type {type(sample)}")
                    yield from (_explode_batched_dict(sample) if self.config.batch_size else [dict(sample)])
            else:
                raise TypeError(f"Unexpected sample type from loader: {type(raw)}")

    def __iter__(self) -> Iterator[TypedSample]:
        for flat_dict in self._iter_flat_samples():
            scene_id, snippet_id = _infer_ids(flat_dict, flat_dict.get("sequence_name", ""))
            mesh = self._load_mesh(scene_id) if self.config.load_meshes else None
            yield TypedSample(flat=flat_dict, scene_id=scene_id, snippet_id=snippet_id, mesh=mesh)


class ASEDatasetConfig(BaseConfig[ASEDataset]):
    """Configuration for ASEDataset.

    Fields mirror ATEK/ASE layout documented in docs/contents/ase_dataset.qmd.
    """

    target: type[ASEDataset] = Field(default=ASEDataset, exclude=True)
    paths: PathConfig = Field(default_factory=PathConfig)
    scene_ids: list[str] | None = Field(default=None, description="Optional list of scene ids to include.")
    atek_root: Path | None = Field(default=None, description="Override root directory containing scene shard folders.")
    atek_variant: str = Field(default="efm_eval", description="Subdirectory name under data_root for ATEK shards.")
    tar_urls: list[str] = Field(
        default_factory=list,
        description="List of ATEK WebDataset shard paths (auto-populated; external override disallowed).",
    )
    scene_to_mesh: dict[str, Path] = Field(default_factory=dict, description="Mapping scene_id -> GT mesh path.")

    batch_size: int | None = Field(
        default=None, description="Optional batch size for load_atek_wds_dataset; None yields single samples."
    )
    shuffle: bool = Field(default=False, description="Shuffle WebDataset shards.")
    repeat: bool = Field(default=False, description="Repeat shards indefinitely (streaming).")

    load_meshes: bool = Field(default=True, description="If True, attach meshes when available.")
    require_mesh: bool = Field(default=False, description="If True, raise when mesh for scene is missing.")
    mesh_simplify_ratio: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Optional quadric decimation ratio for meshes."
    )
    cache_meshes: bool = Field(default=True, description="Cache loaded meshes per scene id.")

    verbose: bool = Field(default=True, description="Enable verbose Console logging.")

    @field_validator("tar_urls", mode="before")
    @classmethod
    def _disallow_external_tar_urls(cls, value: Any) -> None:
        # Ignore any externally supplied value; tar_urls is populated from scene_ids.
        return value

    @field_validator("atek_root", mode="before")
    @classmethod
    def _normalize_atek_root(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        return path

    def _resolve_atek_root(self) -> Path:
        return self._resolve_atek_root_static(self.paths, self.atek_root, self.atek_variant)

    @staticmethod
    def _resolve_atek_root_static(paths: PathConfig, atek_root: Path | None, atek_variant: str) -> Path:
        if atek_root is not None:
            base = Path(atek_root).expanduser()
            if not base.is_absolute():
                base = (Path.cwd() / base).resolve()
            return base
        candidates = [
            paths.data_root / "ase_atek" / atek_variant,
            paths.data_root / f"ase_{atek_variant}",
            paths.data_root / "ase_efm_eval",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @field_validator("tar_urls", mode="before")
    @classmethod
    def _populate_tar_urls(cls, value: list[str] | None, info: ValidationInfo) -> list[str]:
        data = info.data
        paths: PathConfig = data.get("paths") or PathConfig()
        scene_ids: list[str] | None = data.get("scene_ids")
        atek_variant: str = data.get("atek_variant", "efm_eval")
        atek_root: Path | None = data.get("atek_root")

        if value:
            return value

        base = cls._resolve_atek_root_static(paths, atek_root, atek_variant)

        resolved: list[str] = []
        if scene_ids:
            for scene in scene_ids:
                resolved.extend(str(p) for p in sorted((base / scene).glob("*.tar")))
        else:
            resolved = [str(p) for p in sorted(base.glob("**/*.tar"))]

        if not resolved:
            raise ValueError(f"No tar files found under {base}; provide tar_urls or scene_ids.")
        return resolved

    @model_validator(mode="after")
    def _autofill_paths(self) -> ASEDatasetConfig:
        if not self.tar_urls:
            base = self._resolve_atek_root()
            self.tar_urls = [str(p) for scene in self.scene_ids for p in sorted((base / scene).glob("*.tar"))]
            if not self.tar_urls:
                raise ValueError(f"No tar files found under {base} for scenes: {self.scene_ids}")

        if not self.scene_ids:
            inferred: set[str] = set()
            for url in self.tar_urls:
                parent = Path(url).parent.name
                if parent.isdigit():
                    inferred.add(parent)
            self.scene_ids = sorted(inferred) if inferred else None

        if self.load_meshes and not self.scene_to_mesh:
            if self.scene_ids:
                self.scene_to_mesh = {scene: self.paths.resolve_mesh_path(scene) for scene in self.scene_ids}
            else:
                mesh_dir = self.paths.ase_meshes
                self.scene_to_mesh = {p.stem.replace("scene_ply_", ""): p for p in mesh_dir.glob("scene_ply_*.ply")}

        if self.load_meshes and not self.scene_to_mesh and self.require_mesh:
            raise ValueError("load_meshes=True but no meshes resolved.")

        Console.with_prefix(self.__class__.__name__, "config").set_verbose(self.verbose).log(
            f"Resolved {len(self.tar_urls)} tar shards" + (f" for scenes {self.scene_ids}" if self.scene_ids else "")
        )
        return self

    def setup_target(self) -> ASEDataset:  # type: ignore[override]
        console = Console.with_prefix(self.__class__.__name__, "setup_target").set_verbose(self.verbose)
        expanded_urls = [
            str(path)
            for url in self.tar_urls or []
            for path in (sorted(Path().glob(url)) if any(ch in url for ch in "*?[]") else [Path(url)])
        ]
        if not expanded_urls:
            raise FileNotFoundError("No tar files matched derived tar_urls")
        self.tar_urls = expanded_urls
        console.log(f"Preparing ASEDataset (tar URLs: {len(self.tar_urls)}, scenes: {len(self.scene_to_mesh)})")
        return self.target(self)


__all__ = [
    "ASEDataset",
    "ASEDatasetConfig",
    "TypedSample",
    "ase_collate",
]
