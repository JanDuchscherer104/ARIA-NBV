"""VIN snippet cache v2 backed by the shared canonical VIN adapter.

This module implements the v2 cache that stores compact VIN-ready snippet
payloads derived from raw EFM snippets.

Contents:
- config models for VIN cache readers and writers,
- metadata read/write and migration helpers,
- a build dataset and writer for generating VIN payloads from oracle indices,
- a map-style dataset for loading cached VIN snippet views,
- index repair and rebuild helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from efm3d.aria.pose import PoseTW
from pydantic import Field, ValidationInfo, field_validator
from torch.utils.data import DataLoader, Dataset

from ..configs import PathConfig
from ..utils import BaseConfig, Console, Verbosity
from .cache_contracts import OracleRriCacheEntry, VinSnippetCacheEntry, VinSnippetCacheMetadata
from .cache_index import load_json_metadata, read_index, write_index, write_json_metadata
from .efm_snippet_loader import EfmSnippetLoader
from .efm_views import VinSnippetView
from .offline_cache_store import _extract_snippet_token, _format_sample_key, _unique_sample_path
from .offline_cache_store import _read_metadata as _read_oracle_metadata
from .vin_adapter import (
    DEFAULT_VIN_SNIPPET_PAD_POINTS,
    build_vin_snippet_view,
    vin_snippet_cache_config_hash,
)

if TYPE_CHECKING:
    from .oracle_cache import OracleRriCacheConfig

VIN_SNIPPET_CACHE_VERSION = 2
VIN_SNIPPET_PAD_POINTS = DEFAULT_VIN_SNIPPET_PAD_POINTS


def _default_source_cache() -> "OracleRriCacheConfig":
    """Return the default oracle-cache config used by VIN writer configs."""
    from .oracle_cache import OracleRriCacheConfig

    return OracleRriCacheConfig()


@dataclass(slots=True)
class VinSnippetCacheBuildResult:
    """Result of building a VIN snippet payload."""

    entry: OracleRriCacheEntry
    """Oracle-cache entry currently being materialized."""

    payload: dict[str, Any] | None
    """Serialized VIN payload, or ``None`` when building failed."""

    error: str | None = None
    """Optional stringified build error captured for DataLoader workers."""


def _single_item_collate(batch: list[VinSnippetCacheBuildResult]) -> VinSnippetCacheBuildResult:
    """Collapse a unit-size DataLoader batch to its single build result."""
    return batch[0]


class VinSnippetCacheConfig(BaseConfig):
    """Filesystem configuration for VIN snippet cache artifacts."""

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache_dir: Path = Field(
        default_factory=lambda: PathConfig().offline_cache_dir / "vin_snippets",
    )
    """Root directory containing cached VIN snippet samples."""

    index_filename: str = "index.jsonl"
    """Filename for the cache index (JSON lines)."""

    metadata_filename: str = "metadata.json"
    """Filename for the cache metadata JSON."""

    samples_dirname: str = "samples"
    """Subdirectory name that stores sample payloads."""

    pad_points: int = VIN_SNIPPET_PAD_POINTS
    """Fixed point count used for padded VIN cache samples."""

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _resolve_cache_dir(cls, value: str | Path, info: ValidationInfo) -> Path:
        """Resolve relative VIN cache directories against the configured data roots."""
        paths: PathConfig = info.data.get("paths") or PathConfig()
        return paths.resolve_cache_artifact_dir(value)

    @field_validator("pad_points")
    @classmethod
    def _validate_pad_points(cls, value: int) -> int:
        """Validate that the configured padding length is non-negative."""
        pad_points = int(value)
        if pad_points < 0:
            raise ValueError("pad_points must be >= 0.")
        return pad_points

    @property
    def samples_dir(self) -> Path:
        """Return the directory containing VIN payload files."""
        return self.cache_dir / self.samples_dirname

    @property
    def index_path(self) -> Path:
        """Return the path to the VIN cache JSONL index."""
        return self.cache_dir / self.index_filename

    @property
    def metadata_path(self) -> Path:
        """Return the path to the VIN cache metadata JSON."""
        return self.cache_dir / self.metadata_filename


class VinSnippetCacheWriterConfig(BaseConfig["VinSnippetCacheWriter"]):
    """Configuration for building a VIN snippet cache from an oracle cache."""

    @property
    def target(self) -> type["VinSnippetCacheWriter"]:
        """Return the writer factory target."""
        return VinSnippetCacheWriter

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache: VinSnippetCacheConfig = Field(default_factory=VinSnippetCacheConfig)
    """Output VIN snippet cache configuration."""

    source_cache: Any = Field(default_factory=_default_source_cache)
    """Offline oracle cache used to enumerate snippets."""

    dataset: Any = None
    """Optional dataset config override used when oracle metadata lacks dataset config."""

    map_location: torch.device = Field(default="cpu")
    """Device string for loading EFM snippet tensors (defaults to CPU)."""

    semidense_max_points: int | None = None
    """Optional cap on the number of collapsed semidense points."""

    include_inv_dist_std: bool = True
    """Whether to keep inv_dist_std in collapsed points."""

    include_obs_count: bool = False
    """Whether to append per-point observation counts."""

    split: Literal["all", "train", "val"] = "all"
    """Which oracle-cache split to scan for snippet IDs."""

    max_samples: int | None = None
    """Optional cap on number of snippets to process."""

    overwrite: bool = False
    """Allow overwriting an existing cache index."""

    resume: bool = True
    """Reuse existing cache entries when the index already exists."""

    num_failures_allowed: int = 40
    """Maximum number of sample failures before aborting."""

    use_dataloader: bool = False
    """Whether to use a DataLoader to build snippets."""

    num_workers: int = 0
    """Number of DataLoader workers for snippet building."""

    persistent_workers: bool = False
    """Keep DataLoader workers alive between batches when num_workers > 0."""

    prefetch_factor: int | None = None
    """Optional DataLoader prefetch factor."""

    skip_missing_snippets: bool = True
    """Whether to skip snippets that are missing from the source dataset."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Verbosity level for logging progress."""

    @field_validator("include_inv_dist_std")
    @classmethod
    def _validate_inv_dist_std(cls, value: bool) -> bool:
        """Validate that cached VIN snippets keep inverse-distance uncertainty."""
        if not value:
            raise ValueError("VinSnippetView requires include_inv_dist_std=True.")
        return value

    @field_validator("map_location", mode="before")
    @classmethod
    def _validate_map_location(cls, value: str | torch.device) -> torch.device:
        """Normalize the configured torch device for snippet loading."""
        return cls._resolve_device(value)

    @field_validator("num_workers")
    @classmethod
    def _validate_num_workers(cls, value: int) -> int:
        """Validate that DataLoader worker counts are non-negative."""
        if value < 0:
            raise ValueError("num_workers must be >= 0.")
        return int(value)

    @field_validator("prefetch_factor")
    @classmethod
    def _validate_prefetch_factor(cls, value: int | None) -> int | None:
        """Validate that optional DataLoader prefetch factors are positive."""
        if value is None:
            return None
        if value <= 0:
            raise ValueError("prefetch_factor must be >= 1 when provided.")
        return int(value)


class VinSnippetCacheDatasetConfig(BaseConfig["VinSnippetCacheDataset"]):
    """Configuration for loading cached VIN snippet samples."""

    @property
    def target(self) -> type["VinSnippetCacheDataset"]:
        """Return the dataset factory target."""
        return VinSnippetCacheDataset

    cache: VinSnippetCacheConfig = Field(default_factory=VinSnippetCacheConfig)
    """Cache path configuration."""

    map_location: torch.device = Field(default="cpu")
    """Device string for loading cached tensors."""

    limit: int | None = None
    """Optional cap on number of cached samples loaded."""

    @field_validator("map_location", mode="before")
    @classmethod
    def _validate_map_location(cls, value: str | torch.device) -> torch.device:
        """Normalize the configured torch device for cache reads."""
        return cls._resolve_device(value)


def _build_metadata(
    *,
    dataset_config: dict[str, Any] | None,
    source_cache_dir: Path | None,
    source_cache_hash: str | None,
    include_inv_dist_std: bool,
    include_obs_count: bool,
    semidense_max_points: int | None,
    pad_points: int | None,
    version: int,
) -> VinSnippetCacheMetadata:
    """Build VIN cache metadata for a new writer run."""
    config_hash = vin_snippet_cache_config_hash(
        dataset_config=dataset_config,
        include_inv_dist_std=include_inv_dist_std,
        include_obs_count=include_obs_count,
        semidense_max_points=semidense_max_points,
        pad_points=pad_points,
    )
    return VinSnippetCacheMetadata(
        version=version,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        source_cache_dir=str(source_cache_dir) if source_cache_dir is not None else None,
        source_cache_hash=source_cache_hash,
        dataset_config=dataset_config,
        include_inv_dist_std=include_inv_dist_std,
        include_obs_count=include_obs_count,
        semidense_max_points=semidense_max_points,
        pad_points=pad_points,
        config_hash=config_hash,
        num_samples=None,
    )


def _write_metadata(path: Path, meta: VinSnippetCacheMetadata) -> None:
    """Serialize VIN cache metadata to ``metadata.json``."""
    write_json_metadata(
        path,
        {
            "version": meta.version,
            "created_at": meta.created_at,
            "source_cache_dir": meta.source_cache_dir,
            "source_cache_hash": meta.source_cache_hash,
            "dataset_config": meta.dataset_config,
            "include_inv_dist_std": meta.include_inv_dist_std,
            "include_obs_count": meta.include_obs_count,
            "semidense_max_points": meta.semidense_max_points,
            "pad_points": meta.pad_points,
            "config_hash": meta.config_hash,
            "num_samples": meta.num_samples,
        },
    )


def _read_metadata(path: Path) -> VinSnippetCacheMetadata:
    """Read VIN cache metadata from ``metadata.json``."""
    payload = load_json_metadata(path)
    return VinSnippetCacheMetadata(
        version=int(payload["version"]),
        created_at=str(payload["created_at"]),
        source_cache_dir=payload.get("source_cache_dir"),
        source_cache_hash=payload.get("source_cache_hash"),
        dataset_config=payload.get("dataset_config"),
        include_inv_dist_std=bool(payload.get("include_inv_dist_std", True)),
        include_obs_count=bool(payload.get("include_obs_count", False)),
        semidense_max_points=payload.get("semidense_max_points"),
        pad_points=payload.get("pad_points"),
        config_hash=payload.get("config_hash"),
        num_samples=payload.get("num_samples"),
    )


def read_vin_snippet_cache_metadata(path: Path) -> VinSnippetCacheMetadata:
    """Read VIN snippet cache metadata from a directory or metadata path."""
    meta_path = path / "metadata.json" if path.is_dir() else path
    return _read_metadata(meta_path)


def migrate_vin_snippet_cache_inplace(
    *,
    cache: VinSnippetCacheConfig,
    pad_points: int | None = None,
) -> None:
    """Upgrade an existing VIN snippet cache in-place to include lengths + padding."""
    console = Console.with_prefix("VinSnippetCacheMigration")
    meta_path = cache.metadata_path
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing VIN snippet metadata at {meta_path}")

    meta = _read_metadata(meta_path)
    target_pad_points = int(pad_points if pad_points is not None else cache.pad_points)
    console.log(f"Upgrading VIN snippet cache at {cache.cache_dir} to pad_points={target_pad_points}")

    entries = read_index(cache.index_path, entry_type=VinSnippetCacheEntry)
    for entry in entries:
        payload_path = cache.cache_dir / entry.path
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        points_world = payload.get("points_world")
        if points_world is None:
            continue
        points_world = points_world.to(dtype=torch.float32)
        finite = torch.isfinite(points_world[:, :3]).all(dim=-1)
        finite_count = int(finite.sum().item())
        num_points = finite_count
        lengths = payload.get("points_length")
        if lengths is not None:
            lengths = torch.as_tensor(lengths).reshape(-1)
            if lengths.numel() > 0:
                num_points = min(int(lengths[0].item()), finite_count)

        if num_points > target_pad_points:
            points_world = points_world[:target_pad_points]
            num_points = target_pad_points
        if num_points < target_pad_points:
            pad = torch.full(
                (target_pad_points - num_points, points_world.shape[1]),
                float("nan"),
                dtype=points_world.dtype,
                device=points_world.device,
            )
            points_world = torch.cat([points_world[:num_points], pad], dim=0)

        payload["points_world"] = points_world
        payload["points_length"] = torch.tensor([num_points], dtype=torch.int64)
        torch.save(payload, payload_path)

    meta.version = VIN_SNIPPET_CACHE_VERSION
    meta.pad_points = target_pad_points
    meta.config_hash = vin_snippet_cache_config_hash(
        dataset_config=meta.dataset_config,
        include_inv_dist_std=bool(meta.include_inv_dist_std),
        include_obs_count=bool(meta.include_obs_count),
        semidense_max_points=meta.semidense_max_points,
        pad_points=target_pad_points,
    )
    _write_metadata(meta_path, meta)
    console.log("VIN snippet cache migration complete.")


def _read_oracle_cache_index(path: Path) -> list[OracleRriCacheEntry]:
    """Read oracle-cache index entries used to build VIN snippets."""
    return read_index(path, entry_type=OracleRriCacheEntry)


def _build_vin_payload(
    *,
    loader: EfmSnippetLoader,
    entry: OracleRriCacheEntry,
    device: torch.device,
    max_points: int | None,
    include_inv_dist_std: bool,
    include_obs_count: bool,
    pad_points: int | None,
) -> dict[str, Any]:
    """Build one serialized VIN payload from a raw EFM snippet load."""
    efm_snippet = loader.load(scene_id=entry.scene_id, snippet_id=entry.snippet_id)
    vin_snippet = build_vin_snippet_view(
        efm_snippet,
        device=device,
        max_points=max_points,
        include_inv_dist_std=include_inv_dist_std,
        include_obs_count=include_obs_count,
        pad_points=pad_points,
    )
    return {
        "scene_id": entry.scene_id,
        "snippet_id": entry.snippet_id,
        "points_world": vin_snippet.points_world.detach().cpu(),
        "points_length": vin_snippet.lengths.detach().cpu().reshape(-1),
        "t_world_rig": vin_snippet.t_world_rig.tensor().detach().cpu(),
    }


class VinSnippetCacheBuildDataset(Dataset[VinSnippetCacheBuildResult]):
    """Map-style dataset that builds VIN snippet payloads."""

    def __init__(
        self,
        *,
        entries: list[OracleRriCacheEntry],
        dataset_payload: dict[str, Any],
        map_location: torch.device,
        paths: PathConfig,
        semidense_max_points: int | None,
        include_inv_dist_std: bool,
        include_obs_count: bool,
        pad_points: int | None,
        include_gt_mesh: bool = False,
    ) -> None:
        """Initialize the map-style builder used by the writer DataLoader."""
        super().__init__()
        self._entries = list(entries)
        self._dataset_payload = dict(dataset_payload)
        self._map_location = str(map_location)
        self._paths = paths
        self._semidense_max_points = semidense_max_points
        self._include_inv_dist_std = include_inv_dist_std
        self._include_obs_count = include_obs_count
        self._pad_points = pad_points
        self._include_gt_mesh = include_gt_mesh
        self._device = torch.device(self._map_location)
        self._loader: EfmSnippetLoader | None = None

    def __len__(self) -> int:
        """Return the number of oracle entries assigned to this build dataset."""
        return len(self._entries)

    def __getitem__(self, idx: int) -> VinSnippetCacheBuildResult:
        """Build one VIN payload and capture any exception as a result object."""
        entry = self._entries[idx]
        try:
            payload = self._build_payload(entry)
            return VinSnippetCacheBuildResult(entry=entry, payload=payload, error=None)
        except Exception as exc:  # pragma: no cover - defensive for worker failures
            return VinSnippetCacheBuildResult(
                entry=entry,
                payload=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _ensure_loader(self) -> EfmSnippetLoader:
        """Create or reuse the worker-local EFM snippet loader."""
        if self._loader is None:
            self._loader = EfmSnippetLoader(
                dataset_payload=self._dataset_payload,
                device=self._map_location,
                paths=self._paths,
                include_gt_mesh=self._include_gt_mesh,
            )
        return self._loader

    def _build_payload(self, entry: OracleRriCacheEntry) -> dict[str, Any]:
        """Materialize one VIN payload from the requested oracle entry."""
        loader = self._ensure_loader()
        return _build_vin_payload(
            loader=loader,
            entry=entry,
            device=self._device,
            max_points=self._semidense_max_points,
            include_inv_dist_std=self._include_inv_dist_std,
            include_obs_count=self._include_obs_count,
            pad_points=self._pad_points,
        )


class VinSnippetCacheWriter:
    """Build a cache of minimal VIN snippets from an oracle cache index."""

    def __init__(self, config: VinSnippetCacheWriterConfig) -> None:
        """Initialize the VIN cache writer and resolve its metadata inputs."""
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__).set_verbosity(
            self.config.verbosity,
        )
        self.console.log(f"Writing VIN snippets to: {self.config.cache.cache_dir}")

        self._loader: EfmSnippetLoader | None = None
        self._dataset_payload = self._resolve_dataset_payload()
        self._meta = self._prepare_metadata()
        self._config_hash = self._meta.config_hash or "unknown"

    def run(self) -> list[VinSnippetCacheEntry]:
        """Iterate oracle cache entries and persist minimal snippets."""
        cache_dir = self.config.cache.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = self.config.cache.samples_dir
        samples_dir.mkdir(parents=True, exist_ok=True)

        index_path = self.config.cache.index_path
        existing_entries: list[VinSnippetCacheEntry] = []
        existing_pairs: set[tuple[str, str]] = set()
        if self.config.overwrite:
            removed_samples = 0
            for sample_path in samples_dir.glob("*.pt"):
                sample_path.unlink()
                removed_samples += 1
            if index_path.exists():
                index_path.unlink()
            if removed_samples:
                self.console.log(
                    f"Cleared {removed_samples} existing VIN snippet payloads before overwrite rebuild.",
                )
        elif index_path.exists():
            if not self.config.overwrite and not self.config.resume:
                raise FileExistsError(
                    f"VIN snippet cache index already exists at {index_path} "
                    "(set overwrite=True to replace or resume=True to append).",
                )
            if not self.config.overwrite and self.config.cache.metadata_path.exists():
                existing_meta = _read_metadata(self.config.cache.metadata_path)
                expected_hash = self._meta.config_hash
                if existing_meta.config_hash and expected_hash and existing_meta.config_hash != expected_hash:
                    raise ValueError("VIN snippet cache config hash mismatch; set overwrite=True to rebuild.")
            existing_entries = read_index(index_path, entry_type=VinSnippetCacheEntry, allow_missing=True)
            valid_entries: list[VinSnippetCacheEntry] = []
            for entry in existing_entries:
                entry_path = self.config.cache.cache_dir / entry.path
                if entry_path.exists():
                    valid_entries.append(entry)
                    existing_pairs.add((entry.scene_id, entry.snippet_id))
                else:
                    self.console.warn(
                        "Index entry missing sample file; it will be regenerated if encountered "
                        f"(scene={entry.scene_id} snippet={entry.snippet_id}).",
                    )
            existing_entries = valid_entries
            if existing_entries:
                self.console.log(
                    f"Found {len(existing_entries)} existing snippets; skipping duplicates by scene/snippet.",
                )

        _write_metadata(self.config.cache.metadata_path, self._meta)
        entries: list[VinSnippetCacheEntry] = []
        max_samples = self.config.max_samples
        count = len(existing_entries)
        missing = 0
        fail_count = 0

        oracle_entries = self._load_oracle_entries()
        pending_entries = [
            entry for entry in oracle_entries if (entry.scene_id, entry.snippet_id) not in existing_pairs
        ]
        skipped = len(oracle_entries) - len(pending_entries)
        if max_samples is not None:
            remaining = max_samples - count
            if remaining <= 0:
                pending_entries = []
            elif remaining < len(pending_entries):
                pending_entries = pending_entries[:remaining]

        use_dataloader = self.config.use_dataloader or self.config.num_workers > 0
        if use_dataloader and pending_entries:
            dataset = VinSnippetCacheBuildDataset(
                entries=pending_entries,
                dataset_payload=self._dataset_payload,
                map_location=self.config.map_location,
                paths=self.config.paths,
                semidense_max_points=self.config.semidense_max_points,
                include_inv_dist_std=self.config.include_inv_dist_std,
                include_obs_count=self.config.include_obs_count,
                pad_points=self.config.cache.pad_points,
            )
            loader_kwargs: dict[str, Any] = {
                "batch_size": 1,
                "collate_fn": _single_item_collate,
                "num_workers": self.config.num_workers,
                "shuffle": False,
            }
            if self.config.num_workers > 0:
                loader_kwargs["persistent_workers"] = self.config.persistent_workers
                if self.config.prefetch_factor is not None:
                    loader_kwargs["prefetch_factor"] = self.config.prefetch_factor
            iterable: list[OracleRriCacheEntry] | DataLoader = DataLoader(dataset, **loader_kwargs)
        else:
            iterable = pending_entries

        for item in iterable:
            entry: OracleRriCacheEntry | None = None
            try:
                if use_dataloader:
                    result = item
                    if isinstance(result, list):
                        if len(result) != 1:
                            raise ValueError("Unexpected batch size when building VIN snippets.")
                        result = result[0]
                    entry = result.entry
                    if result.payload is None:
                        error_msg = result.error or "Unknown error"
                        if self.config.skip_missing_snippets and error_msg.startswith("FileNotFoundError"):
                            missing += 1
                            self.console.warn(
                                f"Skipping missing snippet scene={entry.scene_id} snippet={entry.snippet_id}: "
                                f"{error_msg}",
                            )
                            continue
                        self.console.error(
                            f"Failed to cache scene={entry.scene_id} snippet={entry.snippet_id}: {error_msg}",
                        )
                        fail_count += 1
                        continue
                    cache_entry = self._write_payload(entry, result.payload, samples_dir)
                else:
                    entry = item
                    cache_entry = self._write_sample(entry, samples_dir)
                entries.append(cache_entry)
                count += 1
            except Exception as exc:
                if self.config.skip_missing_snippets and isinstance(exc, FileNotFoundError):
                    missing += 1
                    self.console.warn(
                        f"Skipping missing snippet scene={entry.scene_id} snippet={entry.snippet_id}: {exc}"
                        if entry is not None
                        else f"Skipping missing snippet: {exc}",
                    )
                    continue
                self.console.error(
                    f"Failed to cache scene={entry.scene_id} snippet={entry.snippet_id}: {exc}"
                    if entry is not None
                    else f"Failed to cache VIN snippet: {exc}",
                )
                fail_count += 1
            finally:
                if fail_count > self.config.num_failures_allowed:
                    raise RuntimeError(
                        f"Exceeded maximum allowed failures ({self.config.num_failures_allowed}). Aborting.",
                    )

        final_entries = [*existing_entries, *entries]
        write_index(index_path, final_entries)
        if skipped:
            self.console.log(f"Skipped {skipped} duplicate snippets already in cache.")
        if missing:
            self.console.log(f"Skipped {missing} snippets missing from the source dataset.")
        self._meta.num_samples = count
        _write_metadata(self.config.cache.metadata_path, self._meta)
        self.console.log(
            f"VIN snippet cache now contains {count} samples (added {len(entries)} new) in {cache_dir}",
        )
        return entries

    def _resolve_dataset_payload(self) -> dict[str, Any]:
        """Resolve the raw-dataset payload used for live snippet loading."""
        if self.config.dataset is not None:
            return self.config.dataset.model_dump_cache(exclude_none=True)
        meta = _read_oracle_metadata(self.config.source_cache.metadata_path)
        dataset_config = getattr(meta, "dataset_config", None)
        if dataset_config is None:
            raise ValueError(
                "Oracle cache metadata does not include dataset_config; "
                "provide VinSnippetCacheWriterConfig.dataset to proceed.",
            )
        return dict(dataset_config)

    def _prepare_metadata(self) -> VinSnippetCacheMetadata:
        """Build the VIN cache metadata for the current writer configuration."""
        meta = _read_oracle_metadata(self.config.source_cache.metadata_path)
        source_hash = getattr(meta, "config_hash", None)
        dataset_cfg = self._dataset_payload
        return _build_metadata(
            dataset_config=dataset_cfg,
            source_cache_dir=self.config.source_cache.cache_dir,
            source_cache_hash=source_hash,
            include_inv_dist_std=self.config.include_inv_dist_std,
            include_obs_count=self.config.include_obs_count,
            semidense_max_points=self.config.semidense_max_points,
            pad_points=self.config.cache.pad_points,
            version=VIN_SNIPPET_CACHE_VERSION,
        )

    def _load_oracle_entries(self) -> list[OracleRriCacheEntry]:
        """Load the base or split-specific oracle entries to process."""
        if self.config.split == "train":
            index_path = self.config.source_cache.train_index_path
        elif self.config.split == "val":
            index_path = self.config.source_cache.val_index_path
        else:
            index_path = self.config.source_cache.index_path
        return _read_oracle_cache_index(index_path)

    def _build_payload(self, entry: OracleRriCacheEntry) -> dict[str, Any]:
        """Build one VIN payload using the writer-owned loader instance."""
        if self._loader is None:
            self._loader = EfmSnippetLoader(
                dataset_payload=self._dataset_payload,
                device=str(self.config.map_location),
                paths=self.config.paths,
                include_gt_mesh=False,
            )
        return _build_vin_payload(
            loader=self._loader,
            entry=entry,
            device=torch.device(str(self.config.map_location)),
            max_points=self.config.semidense_max_points,
            include_inv_dist_std=self.config.include_inv_dist_std,
            include_obs_count=self.config.include_obs_count,
            pad_points=self.config.cache.pad_points,
        )

    def _write_payload(
        self,
        entry: OracleRriCacheEntry,
        payload: dict[str, Any],
        samples_dir: Path,
    ) -> VinSnippetCacheEntry:
        """Persist one VIN payload and return its new index entry."""
        base_key = _format_sample_key(entry.scene_id, entry.snippet_id, self._config_hash)
        sample_key, sample_path = _unique_sample_path(samples_dir, base_key)
        torch.save(payload, sample_path)
        return VinSnippetCacheEntry(
            key=sample_key,
            scene_id=entry.scene_id,
            snippet_id=entry.snippet_id,
            path=str(sample_path.relative_to(self.config.cache.cache_dir)),
        )

    def _write_sample(
        self,
        entry: OracleRriCacheEntry,
        samples_dir: Path,
    ) -> VinSnippetCacheEntry:
        """Build and persist one VIN payload from an oracle entry."""
        payload = self._build_payload(entry)
        return self._write_payload(entry, payload, samples_dir)


class VinSnippetCacheDataset(Dataset[VinSnippetView]):
    """Map-style dataset that reads cached VIN snippet samples."""

    def __init__(self, config: VinSnippetCacheDatasetConfig) -> None:
        """Initialize the VIN cache reader and eagerly load its index."""
        super().__init__()
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self._index = read_index(self.config.cache.index_path, entry_type=VinSnippetCacheEntry)
        self._len = self._resolve_len()
        self._by_scene_snippet: dict[tuple[str, str], VinSnippetCacheEntry] = {}
        for entry in self._index:
            self._by_scene_snippet.setdefault((entry.scene_id, entry.snippet_id), entry)

    def _resolve_len(self) -> int:
        """Resolve the exposed dataset length after applying ``limit``."""
        limit = self.config.limit
        if limit is not None:
            return min(len(self._index), int(limit))
        return len(self._index)

    def __len__(self) -> int:
        """Return the number of readable VIN snippets exposed by this dataset."""
        return self._len

    def __getitem__(self, idx: int) -> VinSnippetView:
        """Return one cached VIN snippet by positional index."""
        if idx < 0 or idx >= self._len:
            raise IndexError("VIN snippet cache index out of range.")
        entry = self._index[idx]
        return self._load_entry(entry, map_location=str(self.config.map_location))

    def get_by_scene_snippet(
        self,
        *,
        scene_id: str,
        snippet_id: str,
        map_location: str | None = None,
    ) -> VinSnippetView | None:
        """Look up a cached VIN snippet by ``(scene_id, snippet_id)``."""
        entry = self._by_scene_snippet.get((scene_id, snippet_id))
        if entry is None:
            snippet_token = _extract_snippet_token(snippet_id)
            entry = self._by_scene_snippet.get((scene_id, snippet_token))
        if entry is None:
            return None
        return self._load_entry(entry, map_location=map_location)

    def _load_entry(self, entry: VinSnippetCacheEntry, *, map_location: str | None) -> VinSnippetView:
        """Decode one cached VIN payload into a :class:`VinSnippetView`."""
        map_loc = map_location or str(self.config.map_location)
        payload = torch.load(
            self.config.cache.cache_dir / entry.path,
            map_location=map_loc,
            weights_only=False,
        )
        device = torch.device(map_loc)
        points_world = payload["points_world"].to(device=device, dtype=torch.float32)
        lengths = payload.get("points_length")
        finite = torch.isfinite(points_world[:, :3]).all(dim=-1)
        finite_count = finite.sum().to(device=device, dtype=torch.int64)
        if lengths is None or finite_count.numel() == 0:
            lengths = finite_count.reshape(1)
        else:
            lengths = lengths.to(device=device, dtype=torch.int64).reshape(-1)
            if lengths.numel() == 0:
                lengths = finite_count.reshape(1)
            else:
                lengths = torch.minimum(lengths, finite_count.reshape(-1))
        t_world_rig = PoseTW(payload["t_world_rig"].to(device=device, dtype=torch.float32))
        return VinSnippetView(points_world=points_world, lengths=lengths, t_world_rig=t_world_rig)


def repair_vin_snippet_cache_index(*, cache: VinSnippetCacheConfig) -> int:
    """Repair the VIN cache index by scanning cached payloads."""
    samples_dir = cache.samples_dir
    if not samples_dir.exists():
        write_index(cache.index_path, [])
        return 0
    entries: list[VinSnippetCacheEntry] = []
    for sample_path in sorted(samples_dir.glob("*.pt")):
        payload = torch.load(sample_path, map_location="cpu", weights_only=False)
        scene_id = str(payload.get("scene_id", ""))
        snippet_id = str(payload.get("snippet_id", ""))
        entries.append(
            VinSnippetCacheEntry(
                key=sample_path.stem,
                scene_id=scene_id,
                snippet_id=snippet_id,
                path=str(sample_path.relative_to(cache.cache_dir)),
            ),
        )
    write_index(cache.index_path, entries)
    meta_path = cache.metadata_path
    if meta_path.exists():
        meta = _read_metadata(meta_path)
        meta.num_samples = len(entries)
        if meta.pad_points is None:
            meta.pad_points = cache.pad_points
        _write_metadata(meta_path, meta)
    return len(entries)


def rebuild_vin_snippet_cache_index(*, cache_dir: Path, pad_points: int | None = None) -> int:
    """Convenience wrapper to repair a VIN cache index from its payload files."""
    cache = VinSnippetCacheConfig(cache_dir=cache_dir, pad_points=pad_points or VIN_SNIPPET_PAD_POINTS)
    return repair_vin_snippet_cache_index(cache=cache)


__all__ = [
    "VIN_SNIPPET_CACHE_VERSION",
    "VIN_SNIPPET_PAD_POINTS",
    "VinSnippetCacheBuildResult",
    "VinSnippetCacheConfig",
    "VinSnippetCacheDataset",
    "VinSnippetCacheDatasetConfig",
    "VinSnippetCacheEntry",
    "VinSnippetCacheMetadata",
    "VinSnippetCacheWriter",
    "VinSnippetCacheWriterConfig",
    "migrate_vin_snippet_cache_inplace",
    "read_vin_snippet_cache_metadata",
    "rebuild_vin_snippet_cache_index",
    "repair_vin_snippet_cache_index",
]
