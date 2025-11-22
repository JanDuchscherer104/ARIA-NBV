"""EFM-formatted ASE dataset wrapper."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import torch
import trimesh
from efm3d.aria.aria_constants import ARIA_POINTS_VOL_MAX, ARIA_POINTS_VOL_MIN, ARIA_POINTS_WORLD
from efm3d.dataset.efm_model_adaptor import load_atek_wds_dataset_as_efm
from pydantic import Field, ValidationInfo, field_validator, model_validator
from torch.utils.data import IterableDataset

from ..configs import PathConfig
from ..utils import BaseConfig, Console
from .efm_views import EfmSnippetView


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


def _tensor3(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor) and value.numel() == 3:
        return value
    return None


def infer_semidense_bounds(efm_dict: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Infer snippet world-space AABB from semidense metadata or points.

    Preference order:
    1) ``ARIA_POINTS_VOL_MIN`` / ``ARIA_POINTS_VOL_MAX`` (or legacy keys ``points/vol_min``, ``points/vol_max``)
    2) Axis-aligned bounds of finite semidense points (ignoring padded/NaN entries)

    Returns:
        Tuple of ``(min, max)`` tensors on CPU if finite bounds are available, otherwise ``None``.
    """

    vol_min = _tensor3(efm_dict.get(ARIA_POINTS_VOL_MIN))
    if vol_min is None:
        vol_min = _tensor3(efm_dict.get("points/vol_min"))

    vol_max = _tensor3(efm_dict.get(ARIA_POINTS_VOL_MAX))
    if vol_max is None:
        vol_max = _tensor3(efm_dict.get("points/vol_max"))
    vol_bounds: tuple[torch.Tensor, torch.Tensor] | None = None
    if vol_min is not None and vol_max is not None and torch.isfinite(vol_min).all() and torch.isfinite(vol_max).all():
        vol_bounds = (vol_min.detach().cpu(), vol_max.detach().cpu())

    points = efm_dict.get(ARIA_POINTS_WORLD)
    point_bounds: tuple[torch.Tensor, torch.Tensor] | None = None
    if isinstance(points, torch.Tensor) and points.shape[-1] == 3:
        lengths = efm_dict.get("msdpd#points_world_lengths") or efm_dict.get("points/lengths")
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to(dtype=torch.long)

        mask_valid = torch.isfinite(points).all(dim=-1)
        if isinstance(lengths, torch.Tensor) and lengths.shape[0] == points.shape[0]:
            idx = torch.arange(points.shape[1], device=points.device).unsqueeze(0) < lengths.unsqueeze(-1)
            mask_valid &= idx

        if mask_valid.any():
            valid_points = points[mask_valid]
            min_vec = valid_points.min(dim=0).values.detach().cpu()
            max_vec = valid_points.max(dim=0).values.detach().cpu()
            if torch.isfinite(min_vec).all() and torch.isfinite(max_vec).all():
                point_bounds = (min_vec, max_vec)

    if vol_bounds is None and point_bounds is None:
        return None
    if vol_bounds is None:
        return point_bounds
    if point_bounds is None:
        return vol_bounds

    vol_extent = (vol_bounds[1] - vol_bounds[0]).clamp_min(1e-6)
    point_extent = (point_bounds[1] - point_bounds[0]).clamp_min(1e-6)
    vol_volume = torch.prod(vol_extent)
    point_volume = torch.prod(point_extent)
    return point_bounds if point_volume < vol_volume else vol_bounds


def crop_mesh_with_bounds(
    mesh: trimesh.Trimesh,
    bounds: tuple[torch.Tensor, torch.Tensor],
    margin_m: float,
    *,
    max_faces: int | None = None,
    console: Console | None = None,
) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` cropped to an AABB with optional margin."""

    bounds_min, bounds_max = bounds
    margin = float(margin_m)
    lo = (bounds_min - margin).numpy()
    hi = (bounds_max + margin).numpy()

    verts = mesh.vertices
    in_bounds = np.logical_and(verts >= lo, verts <= hi).all(axis=1)
    if not np.any(in_bounds):
        if console:
            console.warn("Mesh cropping skipped: no vertices inside snippet bounds.")
        return mesh

    face_mask = np.any(in_bounds[mesh.faces], axis=1)
    if not np.any(face_mask):
        if console:
            console.warn("Mesh cropping skipped: no faces intersect snippet bounds.")
        return mesh

    cropped = mesh.submesh([face_mask], append=True)
    if max_faces is not None and cropped.faces.shape[0] > max_faces:
        cropped = cropped.simplify_quadric_decimation(face_count=max_faces)
        if console:
            console.dbg(f"Decimated cropped mesh to {max_faces} faces.")

    return cropped


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
            atek_to_efm_taxonomy_mapping_file=config.taxonomy_csv.as_posix() if config.taxonomy_csv else None,
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
        mesh = self._decimate_mesh(mesh)
        if self.config.cache_meshes:
            self._mesh_cache[scene_id] = mesh
        return mesh

    def _decimate_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        faces_before = mesh.faces.shape[0]
        target_faces = faces_before

        if self.config.mesh_simplify_ratio not in (None, 0):
            target_faces = min(target_faces, max(4, int(faces_before * self.config.mesh_simplify_ratio)))

        # TODO remove mesh_max_faces, only use mesh_simplify_ratio!
        if self.config.mesh_max_faces is not None:
            target_faces = min(target_faces, int(self.config.mesh_max_faces))

        if target_faces < faces_before:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            self.console.dbg(f"Decimated mesh from {faces_before} to {mesh.faces.shape[0]} faces.")

        return mesh

    def _iter_efm_samples(self) -> Iterator[dict[str, Any]]:
        """Yield per-snippet EFM dicts from the WebDataset loader.

        ``load_atek_wds_dataset_as_efm`` already returns a list of snippet dicts
        when ``batch_size`` is set, so we simply iterate that list and avoid any
        further explosion along the frame dimension (which corrupted shapes).
        """

        for raw in self._efm_wds:
            if isinstance(raw, Mapping):
                yield dict(raw)
                continue

            if isinstance(raw, (list, tuple)):
                for sample in raw:
                    if not isinstance(sample, Mapping):
                        raise TypeError(f"Unexpected sample type {type(sample)} from batched list")
                    yield dict(sample)
                continue

            raise TypeError(f"Unexpected sample type from loader: {type(raw)}")

    def __iter__(self) -> Iterator[EfmSnippetView]:
        for efm_dict in self._iter_efm_samples():
            # second inspection here. efm_dict is corrupted. i.e. efm_dict["points/p3s_world"].shape gives torch.Size([50000, 3])
            scene_id, snippet_id = _infer_ids(efm_dict, efm_dict.get("sequence_name", ""))
            mesh = self._load_mesh(scene_id) if self.config.load_meshes else None
            if mesh is not None and self.config.mesh_crop_margin_m is not None:
                bounds = infer_semidense_bounds(efm_dict)
                if bounds is not None:
                    mesh = crop_mesh_with_bounds(
                        mesh,
                        bounds,
                        self.config.mesh_crop_margin_m,
                        max_faces=self.config.mesh_max_faces,
                        console=self.console if self.config.verbose else None,
                    )
                elif self.config.verbose:
                    self.console.dbg(f"No semidense bounds available for snippet {snippet_id}; skipping mesh crop.")
            yield EfmSnippetView(efm=efm_dict, scene_id=scene_id, snippet_id=snippet_id, mesh=mesh)


class AseEfmDatasetConfig(BaseConfig[AseEfmDataset]):
    """Configuration for :class:`AseEfmDataset`."""

    DEBUG_DEFAULTS: dict = Field({"batch_size": 1, "mesh_simplify_ratio": 0.1, "verbose": True}, exclude=True)

    target: type[AseEfmDataset] = Field(default=AseEfmDataset, exclude=True)
    paths: PathConfig = Field(default_factory=PathConfig)
    atek_variant: Literal["efm", "efm_eval", "cubercnn", "cubercnn_eval"] = Field(default="efm")
    scene_ids: list[str] = Field(default_factory=list)
    tar_urls: list[str] = Field(default_factory=list)
    scene_to_mesh: dict[str, Path] = Field(default_factory=dict)

    taxonomy_csv_filename: str = Field(default="atek_to_efm.csv")
    batch_size: int | None = Field(default=2)
    snippet_length_s: float = Field(default=2.0, gt=0)
    freq_hz: int = Field(default=10, gt=0)
    semidense_points_pad: int = Field(default=50000, gt=0)

    load_meshes: bool = Field(default=True)
    require_mesh: bool = Field(default=False)
    mesh_simplify_ratio: float | None = Field(default=0.1, ge=0.0, le=1.0)
    """Fraction of faces to keep when simplifying meshes. ``None`` disables."""
    mesh_max_faces: int | None = Field(default=None, gt=0)
    """Absolute face cap after simplification; applied once per scene mesh."""
    mesh_crop_margin_m: float | None = Field(default=0.5, ge=0.0)
    """Margin added to semidense bounds when cropping meshes. ``None`` disables cropping."""
    cache_meshes: bool = Field(default=True)

    verbose: bool = Field(default=True)
    is_debug: bool = Field(default=False)

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

        resolved_tar_urls: list[str] = []
        if scene_ids:
            for scene in scene_ids:
                resolved_tar_urls.extend(str(p) for p in sorted((base / scene).glob("*.tar")))
        else:
            resolved_tar_urls = [str(p) for p in sorted(base.glob("**/*.tar"))]

        if not resolved_tar_urls:
            raise ValueError(f"No tar files found under {base}.")

        return resolved_tar_urls

    @property
    def taxonomy_csv(self) -> Path:
        """Resolved taxonomy mapping CSV path."""
        taxonomy_pth = (
            self.paths.root
            / self.paths.external_dir
            / "efm3d"
            / "efm3d"
            / "config"
            / "taxonomy"
            / self.taxonomy_csv_filename
        )
        if not taxonomy_pth.exists():
            raise FileNotFoundError(f"Taxonomy CSV not found at {taxonomy_pth}")
        return taxonomy_pth

    @model_validator(mode="after")
    def _autofill_paths(self) -> "AseEfmDatasetConfig":
        tar_urls = list(self.tar_urls)
        if not tar_urls:
            base = self.paths.resolve_atek_data_dir(self.atek_variant)
            tar_urls = [str(p) for scene in self.scene_ids for p in sorted((base / scene).glob("*.tar"))]
            if not tar_urls:
                raise ValueError(f"No tar files found under {base} for scenes: {self.scene_ids}")

        scene_ids = list(self.scene_ids)
        if not scene_ids:
            inferred: set[str] = set()
            for url in tar_urls:
                parent = Path(url).parent.name
                if parent.isdigit():
                    inferred.add(parent)

            if not inferred:
                raise ValueError("scene_ids not provided and could not be inferred from tar_urls.")

            scene_ids = sorted(inferred)

        scene_to_mesh = dict(self.scene_to_mesh)
        if self.load_meshes and not scene_to_mesh:
            if scene_ids:
                scene_to_mesh = {scene: self.paths.resolve_mesh_path(scene) for scene in scene_ids}
            else:
                mesh_dir = self.paths.ase_meshes
                scene_to_mesh = {p.stem.replace("scene_ply_", ""): p for p in mesh_dir.glob("scene_ply_*.ply")}

        if self.load_meshes and not scene_to_mesh and self.require_mesh:
            raise ValueError("load_meshes=True but no meshes resolved.")

        if self.taxonomy_csv and not self.taxonomy_csv.exists():
            raise FileNotFoundError(f"Taxonomy CSV not found at {self.taxonomy_csv}")

        object.__setattr__(self, "tar_urls", tar_urls)
        object.__setattr__(self, "scene_ids", scene_ids)
        object.__setattr__(self, "scene_to_mesh", scene_to_mesh)

        Console.with_prefix(self.__class__.__name__, "config").set_verbose(self.verbose).log(
            f"Resolved {len(self.tar_urls)} tar shards"
            + (f" for scenes {self.scene_ids}" if self.scene_ids else "")
            + f" | taxonomy={self.taxonomy_csv.name}"
        )
        return self

    @model_validator(mode="after")
    def _set_debug(self) -> Self:
        if self.is_debug:
            console = Console.with_prefix(self.__class__.__name__, "config").set_verbose(True).set_debug(True)
            console.log("Debug mode enabled in config. Setting debug defaults.")
            console.plog(self.DEBUG_DEFAULTS)

            for key, value in self.DEBUG_DEFAULTS.items():
                object.__setattr__(self, key, value)

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


__all__ = ["AseEfmDataset", "AseEfmDatasetConfig", "infer_semidense_bounds", "crop_mesh_with_bounds"]
