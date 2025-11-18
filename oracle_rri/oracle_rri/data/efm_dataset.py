"""EFM-formatted ASE dataset wrapper."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

import torch
import trimesh
from efm3d.dataset.efm_model_adaptor import load_atek_wds_dataset_as_efm
from pydantic import Field, ValidationInfo, field_validator, model_validator
from torch.utils.data import IterableDataset

from ..configs import PathConfig
from ..utils import BaseConfig, Console
from .efm_views import EfmSnippetView


def _explode_batched_dict(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split a batched dict from WebDataset into per-item dicts."""

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


def _infer_ids(efm_dict: Mapping[str, Any], sequence_name: str | None = None) -> tuple[str, str]:
    """Infer scene and snippet ids from keys/url."""

    scene_id = str(sequence_name or efm_dict.get("sequence_name", "unknown"))
    snippet_id = efm_dict.get("__key__") or efm_dict.get("__url__")
    if isinstance(snippet_id, str):
        snippet_id = Path(snippet_id).stem
    else:
        snippet_id = "unknown"
    if scene_id == "unknown" and "__url__" in efm_dict:
        parent = Path(str(efm_dict["__url__"])).parent.name
        if parent.isdigit():
            scene_id = parent
    return scene_id, str(snippet_id)


class AseEfmDataset(IterableDataset[EfmSnippetView]):
    """Iterable dataset yielding :class:`EfmSnippetView` with optional GT mesh."""

    def __init__(self, config: "AseEfmDatasetConfig"):
        super().__init__()
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbose(config.verbose)

        self.console.log(f"Loading EFM-formatted ATEK WDS from {len(config.tar_urls)} shards")
        self._efm_wds = load_atek_wds_dataset_as_efm(
            urls=config.tar_urls,
            freq=config.freq_hz,
            snippet_length_s=config.snippet_length_s,
            semidense_points_pad_to_num=config.semidense_points_pad,
            atek_to_efm_taxonomy_mapping_file=str(config.taxonomy_csv) if config.taxonomy_csv else None,
            batch_size=config.batch_size,
            collation_fn=None,
        )
        self._mesh_cache: dict[str, trimesh.Trimesh] = {}
        self.console.log("EFM loader ready")

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
        if self.config.mesh_simplify_ratio not in (None, 0):
            mesh = mesh.simplify_quadric_decimation(self.config.mesh_simplify_ratio)
            self.console.dbg(f"Simplified mesh {scene_id}")
        if self.config.cache_meshes:
            self._mesh_cache[scene_id] = mesh
        return mesh

    def _iter_efm_samples(self) -> Iterator[dict[str, Any]]:
        for raw in self._efm_wds:
            if isinstance(raw, Mapping):
                samples = _explode_batched_dict(raw) if self.config.batch_size else [dict(raw)]
                yield from samples
            elif isinstance(raw, (list, tuple)):
                for sample in raw:
                    if not isinstance(sample, Mapping):
                        raise TypeError(f"Unexpected sample type {type(sample)}")
                    yield from (_explode_batched_dict(sample) if self.config.batch_size else [dict(sample)])
            else:
                raise TypeError(f"Unexpected sample type from loader: {type(raw)}")

    def __iter__(self) -> Iterator[EfmSnippetView]:
        for efm_dict in self._iter_efm_samples():
            scene_id, snippet_id = _infer_ids(efm_dict, efm_dict.get("sequence_name", ""))
            mesh = self._load_mesh(scene_id) if self.config.load_meshes else None
            yield EfmSnippetView(efm=efm_dict, scene_id=scene_id, snippet_id=snippet_id, mesh=mesh)


class AseEfmDatasetConfig(BaseConfig[AseEfmDataset]):
    """Configuration for :class:`AseEfmDataset`."""

    target: type[AseEfmDataset] = Field(default=AseEfmDataset, exclude=True)
    paths: PathConfig = Field(default_factory=PathConfig)
    atek_variant: Literal["efm", "efm_eval", "cubercnn", "cubercnn_eval"] = Field(default="efm")
    scene_ids: list[str] = Field(default_factory=list)
    tar_urls: list[str] = Field(default_factory=list)
    scene_to_mesh: dict[str, Path] = Field(default_factory=dict)

    taxonomy_csv_filename: str = Field(default="atek_to_efm.csv")
    batch_size: int | None = Field(default=None)
    snippet_length_s: float = Field(default=2.0, gt=0)
    freq_hz: int = Field(default=10, gt=0)
    semidense_points_pad: int = Field(default=50000, gt=0)

    load_meshes: bool = Field(default=True)
    require_mesh: bool = Field(default=False)
    mesh_simplify_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    cache_meshes: bool = Field(default=True)
    verbose: bool = Field(default=True)

    @field_validator("taxonomy_csv_filename", mode="before")
    @classmethod
    def _strip_taxonomy(cls, value: str | Path) -> str:
        return Path(value).name

    @field_validator("tar_urls", mode="before")
    @classmethod
    def _populate_tar_urls(cls, value: list[str] | None, info: ValidationInfo) -> list[str]:
        data = info.data
        paths: PathConfig = data.get("paths") or PathConfig()
        scene_ids: list[str] = data.get("scene_ids")  # type: ignore[assignment]
        atek_variant: str = data.get("atek_variant")  # type: ignore[assignment]

        base = paths.resolve_atek_data_dir(atek_variant)

        resolved: list[str] = []
        if scene_ids:
            for scene in scene_ids:
                resolved.extend(str(p) for p in sorted((base / scene).glob("*.tar")))
        else:
            resolved = [str(p) for p in sorted(base.glob("**/*.tar"))]

        if not resolved:
            raise ValueError(f"No tar files found under {base}; provide tar_urls or scene_ids.")
        return resolved

    @property
    def taxonomy_csv(self) -> Path:
        """Resolved taxonomy mapping CSV path."""

        return (
            self.paths.root
            / self.paths.external_dir
            / "efm3d"
            / "efm3d"
            / "config"
            / "taxonomy"
            / self.taxonomy_csv_filename
        )

    @model_validator(mode="after")
    def _autofill_paths(self) -> "AseEfmDatasetConfig":
        if not self.tar_urls:
            base = self.paths.resolve_atek_data_dir(self.atek_variant)
            self.tar_urls = [str(p) for scene in self.scene_ids for p in sorted((base / scene).glob("*.tar"))]
            if not self.tar_urls:
                raise ValueError(f"No tar files found under {base} for scenes: {self.scene_ids}")

        if not self.scene_ids:
            inferred: set[str] = set()
            for url in self.tar_urls:
                parent = Path(url).parent.name
                if parent.isdigit():
                    inferred.add(parent)

            if not inferred:
                raise ValueError("scene_ids not provided and could not be inferred from tar_urls.")

            self.scene_ids = sorted(inferred)

        if self.load_meshes and not self.scene_to_mesh:
            if self.scene_ids:
                self.scene_to_mesh = {scene: self.paths.resolve_mesh_path(scene) for scene in self.scene_ids}
            else:
                mesh_dir = self.paths.ase_meshes
                self.scene_to_mesh = {p.stem.replace("scene_ply_", ""): p for p in mesh_dir.glob("scene_ply_*.ply")}

        if self.load_meshes and not self.scene_to_mesh and self.require_mesh:
            raise ValueError("load_meshes=True but no meshes resolved.")

        if self.taxonomy_csv and not self.taxonomy_csv.exists():
            raise FileNotFoundError(f"Taxonomy CSV not found at {self.taxonomy_csv}")

        Console.with_prefix(self.__class__.__name__, "config").set_verbose(self.verbose).log(
            f"Resolved {len(self.tar_urls)} tar shards"
            + (f" for scenes {self.scene_ids}" if self.scene_ids else "")
            + f" | taxonomy={self.taxonomy_csv.name}"
        )
        return self

    def setup_target(self) -> AseEfmDataset:  # type: ignore[override]
        console = Console.with_prefix(self.__class__.__name__, "setup_target").set_verbose(self.verbose)
        expanded_urls = [
            str(path)
            for url in self.tar_urls or []
            for path in (sorted(Path().glob(url)) if any(ch in url for ch in "*?[]") else [Path(url)])
        ]
        if not expanded_urls:
            raise FileNotFoundError("No tar files matched derived tar_urls")
        self.tar_urls = expanded_urls
        console.log(f"Preparing AseEfmDataset (tar URLs: {len(self.tar_urls)}, scenes: {len(self.scene_to_mesh)})")
        return self.target(self)


__all__ = ["AseEfmDataset", "AseEfmDatasetConfig"]
