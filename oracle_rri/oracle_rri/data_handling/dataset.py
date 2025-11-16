"""Typed, PyTorch-friendly ASE/ATEK dataset wrapper with GT mesh pairing."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import trimesh
from atek.data_loaders.atek_wds_dataloader import (
    load_atek_wds_dataset,
    select_and_remap_dict_keys,
)

try:  # Prefer installed efm3d, otherwise fall back to vendored source.
    from efm3d.dataset.efm_model_adaptor import EfmModelAdaptor
except ModuleNotFoundError:  # pragma: no cover - exercised in CI fallback
    vendor_root = Path(__file__).resolve().parents[3] / "external" / "efm3d"
    sys.path.append(str(vendor_root))
    from efm3d.dataset.efm_model_adaptor import EfmModelAdaptor
from pydantic import Field, field_validator, model_validator
from torch.utils.data import IterableDataset

from ..configs import PathConfig
from ..utils import BaseConfig, Console
from .schemas import AtekSnippet, CameraLabel

SEQ_PATTERN = re.compile(r".*?(?P<scene_id>\d{4,})(?:_|-)(?P<snippet_id>[\w-]+)")


@dataclass(slots=True)
class ASESample:
    """Single ASE snippet with typed ATEK data and optional GT mesh."""

    scene_id: str
    snippet_id: str
    atek: AtekSnippet
    gt_mesh: trimesh.Trimesh | None

    @property
    def has_mesh(self) -> bool:
        return self.gt_mesh is not None

    @property
    def has_rgb(self) -> bool:
        return self.atek.camera_rgb is not None and self.atek.camera_rgb.images is not None

    @property
    def has_slam_points(self) -> bool:
        return self.atek.semidense is not None and bool(self.atek.semidense.points_world)

    @property
    def has_depth(self) -> bool:
        return self.atek.camera_rgb_depth is not None and self.atek.camera_rgb_depth.images is not None

    def to_flatten_dict(self) -> dict[str, Any]:
        """Return a flat dict compatible with EFM3D + mesh metadata."""
        flat = self.atek.to_flatten_dict()
        flat["scene_id"] = self.scene_id
        flat["snippet_id"] = self.snippet_id
        flat["gt_mesh"] = self.gt_mesh
        return flat

    def to_efm_dict(self, include_mesh: bool = True) -> dict[str, Any]:
        """Remap keys to the EFM3D schema using ``EfmModelAdaptor`` mappings."""
        remapped = select_and_remap_dict_keys(
            sample_dict=self.atek.to_flatten_dict(),
            key_mapping=EfmModelAdaptor.get_dict_key_mapping_all(),
        )
        remapped["scene_id"] = self.scene_id
        remapped["snippet_id"] = self.snippet_id
        if include_mesh:
            remapped["gt_mesh"] = self.gt_mesh
        return remapped

    def __repr__(self) -> str:
        mesh_info = (
            f"mesh=Trimesh(V={len(self.gt_mesh.vertices)},F={len(self.gt_mesh.faces)})"
            if self.gt_mesh is not None
            else "mesh=None"
        )
        cams = ", ".join(sorted([c.name for c in self.atek.cameras.keys()])) or "no-cams"
        semidense = (
            f"{len(self.atek.semidense.points_world)} frames"
            if self.atek.semidense and self.atek.semidense.points_world
            else "no-points"
        )
        return f"ASESample(scene={self.scene_id}, snippet={self.snippet_id}, cams={cams}, semidense={semidense}, {mesh_info})"


def _parse_sequence_name(sequence_name: str) -> tuple[str, str]:
    """Extract scene/snippet identifiers from a sequence name."""
    if not sequence_name:
        return "unknown", "unknown"
    match = SEQ_PATTERN.match(sequence_name)
    if match:
        return match.group("scene_id"), match.group("snippet_id")
    parts = sequence_name.split("_")
    if len(parts) >= 2:
        return parts[1], "_".join(parts[2:]) if len(parts) > 2 else "unknown"
    return sequence_name or "unknown", "unknown"


def _explode_batched_dict(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split a batched dict (B-first) into a list of per-sample dicts."""
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


def ase_collate(batch: Sequence[ASESample]) -> dict[str, Any]:
    """Collate function for ``torch.utils.data.DataLoader``."""
    return {
        "scene_id": [sample.scene_id for sample in batch],
        "snippet_id": [sample.snippet_id for sample in batch],
        "atek": [sample.atek for sample in batch],
        "efm": [sample.to_efm_dict() for sample in batch],
        "gt_mesh": [sample.gt_mesh for sample in batch],
    }


class ASEDataset(IterableDataset[ASESample]):
    """Iterable dataset yielding typed ASE snippets with paired GT meshes."""

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

    def _load_mesh(self, scene_id: str) -> trimesh.Trimesh | None:
        """Load and optionally cache the GT mesh for a scene."""
        if scene_id in self._mesh_cache:
            return self._mesh_cache[scene_id]

        mesh_path = self.config.scene_to_mesh.get(scene_id)
        if mesh_path is None or not mesh_path.exists():
            if self.config.require_mesh:
                raise FileNotFoundError(f"GT mesh for scene {scene_id} not found (config.require_mesh=True).")
            return None

        mesh = trimesh.load(str(mesh_path), process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Loaded mesh for scene {scene_id} is not a Trimesh: {type(mesh)}")

        if self.config.mesh_simplify_ratio is not None:
            original_faces = len(mesh.faces)
            mesh = mesh.simplify_quadric_decimation(self.config.mesh_simplify_ratio)
            self.console.dbg(f"Simplified mesh {scene_id}: {original_faces:,} -> {len(mesh.faces):,} faces")

        if self.config.cache_meshes:
            self._mesh_cache[scene_id] = mesh
        return mesh

    def _iter_flat_samples(self) -> Iterator[dict[str, Any]]:
        """Iterate over flattened ATEK dict samples (handling wds batches)."""
        for raw in self._atek_wds:
            if isinstance(raw, Mapping):
                samples = _explode_batched_dict(raw) if self.config.batch_size else [dict(raw)]
                for sample in samples:
                    yield sample
            elif isinstance(raw, (list, tuple)):
                for sample in raw:
                    if not isinstance(sample, Mapping):
                        raise TypeError(f"Unexpected sample type inside batch: {type(sample)}")
                    yield from (_explode_batched_dict(sample) if self.config.batch_size else [dict(sample)])
            else:
                raise TypeError(f"Unexpected sample type from ATEK loader: {type(raw)}")

    def __iter__(self) -> Iterator[ASESample]:
        """Yield typed samples that can be consumed by a PyTorch DataLoader."""
        for flat_dict in self._iter_flat_samples():
            atek = AtekSnippet.from_flat(flat_dict)
            scene_id, snippet_id = _parse_sequence_name(atek.sequence_name)
            mesh = self._load_mesh(scene_id) if self.config.load_meshes else None

            yield ASESample(
                scene_id=scene_id,
                snippet_id=snippet_id,
                atek=atek,
                gt_mesh=mesh,
            )


class ASEDatasetConfig(BaseConfig[ASEDataset]):
    """Configuration for :class:`ASEDataset`."""

    target: type[ASEDataset] = Field(default=ASEDataset, exclude=True)
    paths: PathConfig = Field(default_factory=PathConfig)

    tar_urls: list[str] = Field(default_factory=list)
    scene_to_mesh: dict[str, Path] = Field(default_factory=dict)
    atek_variant: str = Field(default="efm")
    scene_ids: list[str] | None = Field(default=None)
    atek_root: Path | None = Field(
        default=None, description="Override root directory that contains per-scene ATEK shards."
    )

    batch_size: int | None = Field(
        default=None, description="WebDataset batch size; prefer None and use DataLoader batching."
    )
    shuffle: bool = Field(default=False)
    repeat: bool = Field(default=False)

    load_meshes: bool = Field(default=True)
    require_mesh: bool = Field(default=False)
    mesh_simplify_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    cache_meshes: bool = Field(default=True)

    is_debug: bool = False
    verbose: bool = Field(default=True)

    @field_validator("tar_urls", mode="before")
    @classmethod
    def _normalize_tar_urls(cls, value: list[str | Path] | str | Path) -> list[str]:
        if isinstance(value, (str, Path)):
            value = [value]
        return [str(v) for v in value]

    @field_validator("atek_root", mode="before")
    @classmethod
    def _normalize_atek_root(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        return path

    def _resolve_atek_root(self) -> Path:
        """Resolve the directory that stores per-scene tar shards."""
        if self.atek_root is not None:
            base = self.atek_root
            if not base.is_absolute():
                base = self.paths.data_root / base
            return base

        candidates = [
            self.paths.data_root / "ase_atek" / self.atek_variant,
            self.paths.data_root / f"ase_{self.atek_variant}",
            self.paths.data_root / "ase_efm",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @model_validator(mode="after")
    def _autofill_paths(self) -> ASEDatasetConfig:
        """Populate tar paths and mesh mapping from scene IDs when unset."""
        base = self._resolve_atek_root()

        if not self.tar_urls:
            if self.scene_ids:
                resolved: list[str] = []
                for scene in self.scene_ids:
                    resolved.extend(str(p) for p in sorted((base / scene).glob("*.tar")))
                self.tar_urls = resolved
            else:
                self.tar_urls = [str(p) for p in sorted(base.glob("**/*.tar"))]

        if self.load_meshes and not self.scene_to_mesh:
            if self.scene_ids:
                self.scene_to_mesh = {scene: self.paths.resolve_mesh_path(scene) for scene in self.scene_ids}
            else:
                mesh_dir = self.paths.ase_meshes
                self.scene_to_mesh = {
                    p.stem.replace("scene_ply_", ""): p for p in mesh_dir.glob("scene_ply_*.ply")
                }

        if self.is_debug and self.mesh_simplify_ratio is None:
            self.mesh_simplify_ratio = 0.1

        if not self.tar_urls:
            raise ValueError("No tar files configured. Provide `tar_urls` or `scene_ids`.")
        return self

    def setup_target(self) -> ASEDataset:  # type: ignore[override]
        console = Console.with_prefix(self.__class__.__name__, "setup_target").set_verbose(self.verbose)
        expanded_urls: list[str] = []
        for url in self.tar_urls:
            if any(ch in url for ch in "*?[]"):
                matches = sorted(Path().glob(url))
                if not matches:
                    raise FileNotFoundError(f"No tar files matched glob: {url}")
                expanded_urls.extend(str(p) for p in matches)
            else:
                expanded_urls.append(url)

        self.tar_urls = expanded_urls

        console.log(f"Preparing ASEDataset (tar URLs: {len(self.tar_urls)}, scenes: {len(self.scene_to_mesh)})")
        return self.target(self)

    def __repr__(self) -> str:
        return (
            f"ASEDatasetConfig(tars={len(self.tar_urls)}, meshes={len(self.scene_to_mesh)}, "
            f"batch_size={self.batch_size}, shuffle={self.shuffle}, repeat={self.repeat}, "
            f"load_meshes={self.load_meshes}, simplify={self.mesh_simplify_ratio}, "
            f"atek_variant='{self.atek_variant}')"
        )


__all__ = [
    "ASEDataset",
    "ASEDatasetConfig",
    "ASESample",
    "AtekSnippet",
    "CameraLabel",
    "ase_collate",
]
