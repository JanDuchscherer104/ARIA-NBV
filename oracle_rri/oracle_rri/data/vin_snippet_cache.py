"""Minimal VIN snippet cache utilities.

This cache stores only the data needed for VIN v2 training:
- collapsed semidense points (XYZ + optional extras like inv_dist_std/obs_count), and
- historical rig trajectory poses.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from efm3d.aria.pose import PoseTW
from pydantic import Field, ValidationInfo, field_validator
from torch.utils.data import DataLoader, Dataset

from ..configs import PathConfig
from ..utils import BaseConfig, Console, Verbosity
from .efm_dataset import AseEfmDatasetConfig
from .efm_snippet_loader import EfmSnippetLoader
from .efm_views import VinSnippetView

if TYPE_CHECKING:
    from .offline_cache import OracleRriCacheConfig
from .offline_cache_store import _format_sample_key, _unique_sample_path
from .offline_cache_store import _read_metadata as _read_oracle_metadata
from .offline_cache_types import OracleRriCacheEntry, OracleRriCacheMetadata
from .vin_snippet_utils import build_vin_snippet_view, vin_snippet_cache_config_hash

VIN_SNIPPET_CACHE_VERSION = 1


def _default_source_cache() -> "OracleRriCacheConfig":
    from .offline_cache import OracleRriCacheConfig

    return OracleRriCacheConfig()


@dataclass(slots=True)
class VinSnippetCacheMetadata:
    """Top-level metadata for a VIN snippet cache directory."""

    version: int
    created_at: str
    source_cache_dir: str | None
    source_cache_hash: str | None
    dataset_config: dict[str, Any] | None
    include_inv_dist_std: bool
    include_obs_count: bool
    semidense_max_points: int | None
    config_hash: str | None = None
    num_samples: int | None = None


@dataclass(slots=True)
class VinSnippetCacheEntry:
    """Index entry describing a cached VIN snippet."""

    key: str
    scene_id: str
    snippet_id: str
    path: str


@dataclass(slots=True)
class VinSnippetCacheBuildResult:
    """Result of building a VIN snippet payload."""

    entry: OracleRriCacheEntry
    payload: dict[str, Any] | None
    error: str | None = None


def _single_item_collate(batch: list[VinSnippetCacheBuildResult]) -> VinSnippetCacheBuildResult:
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

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _resolve_cache_dir(cls, value: str | Path, info: ValidationInfo) -> Path:
        paths: PathConfig = info.data.get("paths") or PathConfig()
        path = Path(value)
        if path.is_absolute():
            return path.expanduser().resolve()
        base_dir = paths.offline_cache_dir or paths.data_root
        if path.parts:
            if path.parts[0] == paths.data_root.name or (
                paths.offline_cache_dir is not None and path.parts[0] == paths.offline_cache_dir.name
            ):
                base_dir = paths.root
        return paths.resolve_under_root(path, base_dir=base_dir)

    @property
    def samples_dir(self) -> Path:
        return self.cache_dir / self.samples_dirname

    @property
    def index_path(self) -> Path:
        return self.cache_dir / self.index_filename

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / self.metadata_filename


class VinSnippetCacheWriterConfig(BaseConfig["VinSnippetCacheWriter"]):
    """Configuration for building a VIN snippet cache from an oracle cache."""

    target: type["VinSnippetCacheWriter"] = Field(
        default_factory=lambda: VinSnippetCacheWriter,
        exclude=True,
    )

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache: VinSnippetCacheConfig = Field(default_factory=VinSnippetCacheConfig)
    """Output VIN snippet cache configuration."""

    source_cache: Any = Field(default_factory=_default_source_cache)
    """Offline oracle cache used to enumerate snippets."""

    dataset: AseEfmDatasetConfig | None = None
    """Optional dataset config override (used when oracle cache metadata lacks dataset config)."""

    map_location: torch.device = Field(default="cpu")
    """Device string for loading EFM snippet tensors (defaults to CPU)."""

    semidense_max_points: int | None = None
    """Optional cap on the number of collapsed semidense points."""

    include_inv_dist_std: bool = True
    """Whether to keep inv_dist_std in collapsed points (required by VinSnippetView)."""

    include_obs_count: bool = False
    """Whether to append per-point observation counts to the collapsed points."""

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
    """Whether to use a DataLoader to build snippets (enables multiprocessing)."""

    num_workers: int = 0
    """Number of DataLoader workers for snippet building (0 = main process)."""

    persistent_workers: bool = False
    """Keep DataLoader workers alive between batches when num_workers > 0."""

    prefetch_factor: int | None = None
    """Optional DataLoader prefetch factor (only when num_workers > 0)."""

    skip_missing_snippets: bool = True
    """Whether to skip snippets that are missing from the source dataset."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Verbosity level for logging progress."""

    @field_validator("include_inv_dist_std")
    @classmethod
    def _validate_inv_dist_std(cls, value: bool) -> bool:
        if not value:
            raise ValueError("VinSnippetView requires include_inv_dist_std=True.")
        return value

    @field_validator("map_location", mode="before")
    @classmethod
    def _validate_map_location(cls, value: str | torch.device) -> torch.device:
        return cls._resolve_device(value)

    @field_validator("num_workers")
    @classmethod
    def _validate_num_workers(cls, value: int) -> int:
        if value < 0:
            raise ValueError("num_workers must be >= 0.")
        return int(value)

    @field_validator("prefetch_factor")
    @classmethod
    def _validate_prefetch_factor(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("prefetch_factor must be >= 1 when provided.")
        return int(value)


class VinSnippetCacheDatasetConfig(BaseConfig["VinSnippetCacheDataset"]):
    """Configuration for loading cached VIN snippet samples."""

    target: type["VinSnippetCacheDataset"] = Field(
        default_factory=lambda: VinSnippetCacheDataset,
        exclude=True,
    )

    cache: VinSnippetCacheConfig = Field(default_factory=VinSnippetCacheConfig)
    """Cache path configuration."""

    map_location: torch.device = Field(default="cpu")
    """Device string for loading cached tensors (e.g., 'cpu', 'cuda')."""

    limit: int | None = None
    """Optional cap on number of cached samples loaded."""

    @field_validator("map_location", mode="before")
    @classmethod
    def _validate_map_location(cls, value: str) -> torch.device:
        return cls._resolve_device(value)


def _build_metadata(
    *,
    dataset_config: dict[str, Any] | None,
    source_cache_dir: Path | None,
    source_cache_hash: str | None,
    include_inv_dist_std: bool,
    include_obs_count: bool,
    semidense_max_points: int | None,
    version: int,
) -> VinSnippetCacheMetadata:
    config_hash = vin_snippet_cache_config_hash(
        dataset_config=dataset_config,
        include_inv_dist_std=include_inv_dist_std,
        include_obs_count=include_obs_count,
        semidense_max_points=semidense_max_points,
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
        config_hash=config_hash,
        num_samples=None,
    )


def _write_metadata(path: Path, meta: VinSnippetCacheMetadata) -> None:
    payload = {
        "version": meta.version,
        "created_at": meta.created_at,
        "source_cache_dir": meta.source_cache_dir,
        "source_cache_hash": meta.source_cache_hash,
        "dataset_config": meta.dataset_config,
        "include_inv_dist_std": meta.include_inv_dist_std,
        "include_obs_count": meta.include_obs_count,
        "semidense_max_points": meta.semidense_max_points,
        "config_hash": meta.config_hash,
        "num_samples": meta.num_samples,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_metadata(path: Path) -> VinSnippetCacheMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return VinSnippetCacheMetadata(
        version=int(payload["version"]),
        created_at=str(payload["created_at"]),
        source_cache_dir=payload.get("source_cache_dir"),
        source_cache_hash=payload.get("source_cache_hash"),
        dataset_config=payload.get("dataset_config"),
        include_inv_dist_std=bool(payload.get("include_inv_dist_std", True)),
        include_obs_count=bool(payload.get("include_obs_count", False)),
        semidense_max_points=payload.get("semidense_max_points"),
        config_hash=payload.get("config_hash"),
        num_samples=payload.get("num_samples"),
    )


def read_vin_snippet_cache_metadata(path: Path) -> VinSnippetCacheMetadata:
    """Public helper to read VIN snippet cache metadata."""
    return _read_metadata(path)


def _read_cache_index(path: Path, *, allow_missing: bool = False) -> list[VinSnippetCacheEntry]:
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Missing VIN snippet cache index at {path}")
    entries: list[VinSnippetCacheEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        entries.append(
            VinSnippetCacheEntry(
                key=item["key"],
                scene_id=item["scene_id"],
                snippet_id=item["snippet_id"],
                path=item["path"],
            ),
        )
    return entries


def _read_oracle_cache_index(path: Path) -> list[OracleRriCacheEntry]:
    entries: list[OracleRriCacheEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        entries.append(
            OracleRriCacheEntry(
                key=item["key"],
                scene_id=item["scene_id"],
                snippet_id=item["snippet_id"],
                path=item["path"],
            ),
        )
    return entries


def _build_vin_payload(
    *,
    loader: EfmSnippetLoader,
    entry: OracleRriCacheEntry,
    device: torch.device,
    max_points: int | None,
    include_inv_dist_std: bool,
    include_obs_count: bool,
) -> dict[str, Any]:
    efm_snippet = loader.load(scene_id=entry.scene_id, snippet_id=entry.snippet_id)
    vin_snippet = build_vin_snippet_view(
        efm_snippet,
        device=device,
        max_points=max_points,
        include_inv_dist_std=include_inv_dist_std,
        include_obs_count=include_obs_count,
    )
    return {
        "scene_id": entry.scene_id,
        "snippet_id": entry.snippet_id,
        "points_world": vin_snippet.points_world.detach().cpu(),
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
        include_gt_mesh: bool = False,
    ) -> None:
        super().__init__()
        self._entries = list(entries)
        self._dataset_payload = dict(dataset_payload)
        self._map_location = str(map_location)
        self._paths = paths
        self._semidense_max_points = semidense_max_points
        self._include_inv_dist_std = include_inv_dist_std
        self._include_obs_count = include_obs_count
        self._include_gt_mesh = include_gt_mesh
        self._device = torch.device(self._map_location)
        self._loader: EfmSnippetLoader | None = None

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> VinSnippetCacheBuildResult:
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
        if self._loader is None:
            self._loader = EfmSnippetLoader(
                dataset_payload=self._dataset_payload,
                device=self._map_location,
                paths=self._paths,
                include_gt_mesh=self._include_gt_mesh,
            )
        return self._loader

    def _build_payload(self, entry: OracleRriCacheEntry) -> dict[str, Any]:
        loader = self._ensure_loader()
        return _build_vin_payload(
            loader=loader,
            entry=entry,
            device=self._device,
            max_points=self._semidense_max_points,
            include_inv_dist_std=self._include_inv_dist_std,
            include_obs_count=self._include_obs_count,
        )


class VinSnippetCacheWriter:
    """Build a cache of minimal VIN snippets from an oracle cache index."""

    def __init__(self, config: VinSnippetCacheWriterConfig) -> None:
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
        if index_path.exists():
            if not self.config.overwrite and not self.config.resume:
                raise FileExistsError(
                    f"VIN snippet cache index already exists at {index_path} "
                    "(set overwrite=True to replace or resume=True to append).",
                )
            if not self.config.overwrite and self.config.cache.metadata_path.exists():
                existing_meta = _read_metadata(self.config.cache.metadata_path)
                expected_hash = self._meta.config_hash
                if existing_meta.config_hash and expected_hash and existing_meta.config_hash != expected_hash:
                    raise ValueError(
                        "VIN snippet cache config hash mismatch; set overwrite=True to rebuild.",
                    )
            existing_entries = _read_cache_index(index_path, allow_missing=True)
            valid_entries: list[VinSnippetCacheEntry] = []
            had_missing = False
            for entry in existing_entries:
                entry_path = self.config.cache.cache_dir / entry.path
                if entry_path.exists():
                    valid_entries.append(entry)
                    existing_pairs.add((entry.scene_id, entry.snippet_id))
                else:
                    had_missing = True
                    self.console.warn(
                        f"Index entry missing sample file; will regenerate if encountered "
                        f"(scene={entry.scene_id} snippet={entry.snippet_id}).",
                    )
            existing_entries = valid_entries
            if existing_entries:
                self.console.log(
                    f"Found {len(existing_entries)} existing snippets; skipping duplicates by scene/snippet.",
                )

            if self.config.overwrite or had_missing:
                index_path.unlink()
                if existing_entries:
                    index_path.write_text(
                        "\n".join(json.dumps(asdict(entry)) for entry in existing_entries) + "\n",
                        encoding="utf-8",
                    )
                else:
                    index_path.touch()

        _write_metadata(self.config.cache.metadata_path, self._meta)
        entries: list[VinSnippetCacheEntry] = []
        max_samples = self.config.max_samples
        count = len(existing_entries)
        skipped = 0
        missing = 0
        fail_count = 0
        if not index_path.exists():
            if existing_entries:
                index_path.write_text(
                    "\n".join(json.dumps(asdict(entry)) for entry in existing_entries) + "\n",
                    encoding="utf-8",
                )
            else:
                index_path.touch()

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
            iterable: list[OracleRriCacheEntry] | DataLoader = DataLoader(
                dataset,
                **loader_kwargs,
            )
        else:
            iterable = pending_entries
        with index_path.open("a", encoding="utf-8") as index_f:
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
                    index_f.write(json.dumps(asdict(cache_entry)) + "\n")
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
        if self.config.dataset is not None:
            return self.config.dataset.model_dump_cache(exclude_none=True)
        meta = _read_oracle_metadata(self.config.source_cache.metadata_path)
        if not isinstance(meta, OracleRriCacheMetadata):
            raise ValueError("Failed to read oracle cache metadata.")
        if meta.dataset_config is None:
            raise ValueError(
                "Oracle cache metadata does not include dataset_config; "
                "provide VinSnippetCacheWriterConfig.dataset to proceed.",
            )
        return dict(meta.dataset_config)

    def _prepare_metadata(self) -> VinSnippetCacheMetadata:
        meta = _read_oracle_metadata(self.config.source_cache.metadata_path)
        source_hash = meta.config_hash if isinstance(meta, OracleRriCacheMetadata) else None
        dataset_cfg = self._dataset_payload
        return _build_metadata(
            dataset_config=dataset_cfg,
            source_cache_dir=self.config.source_cache.cache_dir,
            source_cache_hash=source_hash,
            include_inv_dist_std=self.config.include_inv_dist_std,
            include_obs_count=self.config.include_obs_count,
            semidense_max_points=self.config.semidense_max_points,
            version=VIN_SNIPPET_CACHE_VERSION,
        )

    def _load_oracle_entries(self) -> list[OracleRriCacheEntry]:
        if self.config.split == "train":
            index_path = self.config.source_cache.train_index_path
        elif self.config.split == "val":
            index_path = self.config.source_cache.val_index_path
        else:
            index_path = self.config.source_cache.index_path
        return _read_oracle_cache_index(index_path)

    def _build_payload(self, entry: OracleRriCacheEntry) -> dict[str, Any]:
        if self._loader is None:
            self._loader = EfmSnippetLoader(
                dataset_payload=self._dataset_payload,
                device=str(self.config.map_location),
                paths=self.config.paths,
                include_gt_mesh=False,
            )
        device = torch.device(str(self.config.map_location))
        return _build_vin_payload(
            loader=self._loader,
            entry=entry,
            device=device,
            max_points=self.config.semidense_max_points,
            include_inv_dist_std=self.config.include_inv_dist_std,
            include_obs_count=self.config.include_obs_count,
        )

    def _write_payload(
        self,
        entry: OracleRriCacheEntry,
        payload: dict[str, Any],
        samples_dir: Path,
    ) -> VinSnippetCacheEntry:
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
        payload = self._build_payload(entry)
        return self._write_payload(entry, payload, samples_dir)


class VinSnippetCacheDataset(Dataset[VinSnippetView]):
    """Map-style dataset that reads cached VIN snippet samples."""

    def __init__(self, config: VinSnippetCacheDatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        self._index = _read_cache_index(self.config.cache.index_path)
        self._len = self._resolve_len()
        self._by_scene_snippet: dict[tuple[str, str], VinSnippetCacheEntry] = {}
        for entry in self._index:
            self._by_scene_snippet.setdefault((entry.scene_id, entry.snippet_id), entry)

    def _resolve_len(self) -> int:
        limit = self.config.limit
        if limit is not None:
            return min(len(self._index), int(limit))
        return len(self._index)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> VinSnippetView:
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
        entry = self._by_scene_snippet.get((scene_id, snippet_id))
        if entry is None:
            return None
        return self._load_entry(entry, map_location=map_location)

    def _load_entry(self, entry: VinSnippetCacheEntry, *, map_location: str | None) -> VinSnippetView:
        map_loc = map_location or str(self.config.map_location)
        payload = torch.load(
            self.config.cache.cache_dir / entry.path,
            map_location=map_loc,
            weights_only=False,
        )
        device = torch.device(map_loc)
        points_world = payload["points_world"].to(device=device, dtype=torch.float32)
        t_world_rig = PoseTW(payload["t_world_rig"].to(device=device, dtype=torch.float32))
        return VinSnippetView(points_world=points_world, t_world_rig=t_world_rig)


__all__ = [
    "VinSnippetCacheConfig",
    "VinSnippetCacheDataset",
    "VinSnippetCacheDatasetConfig",
    "VinSnippetCacheEntry",
    "VinSnippetCacheMetadata",
    "VinSnippetCacheWriter",
    "VinSnippetCacheWriterConfig",
    "read_vin_snippet_cache_metadata",
]
