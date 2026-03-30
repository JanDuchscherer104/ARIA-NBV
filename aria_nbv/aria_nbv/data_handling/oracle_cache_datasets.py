"""Dataset readers for oracle-cache payloads and VIN-ready batches.

This module contains the map-style dataset implementations layered on top of
the oracle-cache filesystem contract:
- `OracleRriCacheDataset` decodes cached oracle payloads into cache samples or
  VIN-ready batches, and
- `OracleRriCacheVinDataset` constrains that reader to always emit
  `VinOracleBatch` values for training-facing code.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from ..rendering.candidate_depth_renderer import CandidateDepths
from ..rri_metrics.types import RriResult
from ..utils import Console
from ..vin.types import EvlBackboneOutput
from . import oracle_cache as oracle_cache_runtime
from .cache_contracts import OracleRriCacheEntry, OracleRriCacheSample
from .cache_index import read_index, validate_oracle_split_indices
from .efm_snippet_loader import EfmSnippetLoader
from .efm_views import EfmSnippetView, VinSnippetView
from .offline_cache_store import _read_metadata
from .oracle_cache import OracleRriCacheDatasetConfig, _read_vin_snippet_cache_index
from .vin_adapter import DEFAULT_VIN_SNIPPET_PAD_POINTS, empty_vin_snippet
from .vin_provider import (
    EfmSnippetProvider,
    VinSnippetCacheProvider,
    VinSnippetProviderChain,
    expected_vin_snippet_cache_hash,
)

if TYPE_CHECKING:
    from .vin_oracle_types import VinOracleBatch


class OracleRriCacheDataset(Dataset[OracleRriCacheSample]):
    """Map-style dataset that reads cached oracle outputs."""

    def __init__(self, config: OracleRriCacheDatasetConfig) -> None:
        """Initialize the oracle-cache reader and eagerly load its index."""
        super().__init__()
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__)
        if self.config.include_gt_mesh and not self.config.include_efm_snippet:
            self.console.warn("include_gt_mesh=True has no effect unless include_efm_snippet=True.")
        self._index = self._load_index()
        self.metadata = _read_metadata(self.config.cache.metadata_path)
        self._len = self._resolve_len()
        self._efm_loader_by_device: dict[str, EfmSnippetLoader] = {}
        self._vin_snippet_provider: VinSnippetProviderChain | None = None
        self._vin_snippet_expected_hash = self._compute_vin_snippet_expected_hash()

    def _resolve_len(self) -> int:
        """Resolve the exposed dataset length after simplification and limits."""
        base_len = len(self._index)
        simplification = self.config.simplification
        if simplification is not None:
            base_len = int(base_len * simplification)
        limit = self.config.limit
        if limit is not None:
            return min(base_len, int(limit))
        return base_len

    def __getstate__(self) -> dict[str, Any]:
        """Drop worker-local loader state before pickling the dataset."""
        state = self.__dict__.copy()
        state["console"] = None
        state["_efm_loader_by_device"] = {}
        state["_vin_snippet_provider"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore worker-local loader state after unpickling the dataset."""
        self.__dict__.update(state)
        if self.__dict__.get("console") is None:
            self.console = Console.with_prefix(self.__class__.__name__)
        if self.__dict__.get("_efm_loader_by_device") is None:
            self._efm_loader_by_device = {}
        if self.__dict__.get("_vin_snippet_provider") is None:
            self._vin_snippet_provider = None
        if self.__dict__.get("_vin_snippet_expected_hash") is None:
            self._vin_snippet_expected_hash = self._compute_vin_snippet_expected_hash()

    def _resolve_vin_pad_points(self) -> int:
        """Return the VIN padding size implied by the configured snippet source."""
        if self.config.vin_snippet_cache is not None:
            return int(self.config.vin_snippet_cache.pad_points)
        return DEFAULT_VIN_SNIPPET_PAD_POINTS

    def _compute_vin_snippet_expected_hash(self) -> str | None:
        """Compute the expected VIN-cache compatibility hash for this reader."""
        if self.config.vin_snippet_cache is None:
            return None
        dataset_cfg = self.metadata.dataset_config if hasattr(self, "metadata") else None
        if dataset_cfg is None:
            meta = _read_metadata(self.config.cache.metadata_path)
            dataset_cfg = meta.dataset_config
        if dataset_cfg is None:
            return None
        return expected_vin_snippet_cache_hash(
            dataset_config=dataset_cfg,
            include_inv_dist_std=True,
            include_obs_count=self.config.semidense_include_obs_count,
            semidense_max_points=self.config.semidense_max_points,
            pad_points=self._resolve_vin_pad_points(),
        )

    def _ensure_vin_snippet_provider(self) -> VinSnippetProviderChain | None:
        """Build and cache the provider chain used for VIN snippet materialization."""
        if self._vin_snippet_provider is not None:
            return self._vin_snippet_provider
        if not self.config.include_efm_snippet:
            return None

        providers: list[VinSnippetCacheProvider | EfmSnippetProvider] = []
        mode = self.config.vin_snippet_cache_mode
        if mode == "required" and self.config.vin_snippet_cache is None:
            raise ValueError("vin_snippet_cache_mode='required' requires vin_snippet_cache configuration.")
        if self.config.vin_snippet_cache is not None and mode != "disabled":
            cache_provider = VinSnippetCacheProvider(
                cache=self.config.vin_snippet_cache,
                expected_config_hash=self._vin_snippet_expected_hash,
                mode=mode,
                console=self.console,
            )
            providers.append(cache_provider)

        if mode != "required":
            efm_provider = EfmSnippetProvider(
                dataset_payload=self.metadata.dataset_config,
                paths=self.config.paths,
                include_gt_mesh=self.config.include_gt_mesh,
                semidense_max_points=self.config.semidense_max_points,
                include_inv_dist_std=True,
                include_obs_count=self.config.semidense_include_obs_count,
                efm_keep_keys=set(self.config.efm_keep_keys or []) or None,
                pad_points=self._resolve_vin_pad_points(),
            )
            providers.append(efm_provider)

        if not providers:
            return None
        self._vin_snippet_provider = VinSnippetProviderChain(providers=providers)
        return self._vin_snippet_provider

    def __len__(self) -> int:
        """Return the number of readable cache entries exposed by this dataset."""
        return self._len

    def __getitem__(self, idx: int) -> OracleRriCacheSample | "VinOracleBatch":  # type: ignore[override]
        """Decode one cached oracle sample or VIN batch by positional index."""
        if idx < 0 or idx >= self._len:
            raise IndexError("Cache index out of range.")
        entry = self._index[idx]
        path = self.config.cache.cache_dir / entry.path
        map_location = "cpu"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            payload = torch.load(path, map_location=map_location, weights_only=False)
        device = torch.device("cpu")

        require_cache_sample = self.config.return_format == "cache_sample"
        require_depths = self.config.return_format in {"cache_sample", "vin_batch"}
        require_candidates = require_cache_sample
        require_pcs = require_cache_sample

        if require_candidates and not self.config.load_candidates:
            raise ValueError("load_candidates must be True when return_format='cache_sample'.")
        if require_depths and not self.config.load_depths:
            raise ValueError("load_depths must be True when return_format requires depths.")
        if require_pcs and not self.config.load_candidate_pcs:
            raise ValueError("load_candidate_pcs must be True when return_format='cache_sample'.")

        candidates = None
        if self.config.load_candidates:
            candidates = oracle_cache_runtime.decode_candidates(payload["candidates"], device=None)
        depths = None
        if self.config.load_depths and payload.get("depths") is not None:
            depths = oracle_cache_runtime.decode_depths(payload["depths"], device=device)
        pcs = None
        if self.config.load_candidate_pcs and payload.get("candidate_pcs") is not None:
            pcs = oracle_cache_runtime.decode_candidate_pcs(payload["candidate_pcs"], device=device)
        rri = oracle_cache_runtime.decode_rri(payload["rri"], device=device)

        if require_depths and depths is None:
            raise ValueError("Cached sample missing depths; re-generate with include_depths=True.")
        if require_pcs and pcs is None:
            raise ValueError("Cached sample missing candidate point clouds; re-generate with include_pointclouds=True.")

        backbone_out = None
        if self.config.load_backbone and payload.get("backbone") is not None:
            keep_fields = set(self.config.backbone_keep_fields) if self.config.backbone_keep_fields else None
            backbone_out = oracle_cache_runtime.decode_backbone(
                payload["backbone"],
                device=device,
                include_fields=keep_fields,
            )

        scene_id = payload["scene_id"]
        snippet_id = payload["snippet_id"]

        vin_snippet = None
        if self.config.return_format == "vin_batch" and self.config.include_efm_snippet:
            provider = self._ensure_vin_snippet_provider()
            if provider is not None:
                try:
                    vin_snippet = provider.get(
                        scene_id=scene_id,
                        snippet_id=snippet_id,
                        map_location=map_location,
                    )
                except Exception as exc:
                    if self.config.vin_snippet_cache_mode == "required":
                        raise
                    self.console.warn(
                        "Failed to load VIN snippet (cache/EFM) for "
                        f"scene={scene_id} snippet={snippet_id}: {exc}. "
                        "Using empty snippet; consider repairing the VIN snippet cache.",
                    )
            if vin_snippet is None:
                if self.config.vin_snippet_cache_mode == "required":
                    raise FileNotFoundError(
                        "vin_snippet_cache_mode='required' but snippet could not be loaded for "
                        f"scene={scene_id} snippet={snippet_id}.",
                    )
                extra_dim = 1 + int(self.config.semidense_include_obs_count)
                vin_snippet = empty_vin_snippet(
                    device,
                    extra_dim=extra_dim,
                    pad_points=self._resolve_vin_pad_points(),
                )

        efm_snippet = None
        if self.config.include_efm_snippet and self.config.return_format == "cache_sample":
            try:
                loader = self._efm_loader_by_device.get(map_location)
                if loader is None:
                    loader = EfmSnippetLoader(
                        dataset_payload=self.metadata.dataset_config,
                        device=map_location,
                        paths=self.config.paths,
                        include_gt_mesh=self.config.include_gt_mesh,
                    )
                    self._efm_loader_by_device[map_location] = loader
                efm_snippet = loader.load(scene_id=scene_id, snippet_id=snippet_id)
                if self.config.include_gt_mesh and efm_snippet.mesh is None:
                    self.console.warn(f"GT mesh missing for scene={scene_id} snippet={snippet_id}.")
                if self.config.efm_keep_keys is not None:
                    efm_snippet = efm_snippet.prune_efm(set(self.config.efm_keep_keys))
            except Exception as exc:
                self.console.warn(
                    f"Failed to load EFM snippet for scene={scene_id} snippet={snippet_id}: {exc}",
                )
                efm_snippet = None

        if self.config.return_format == "vin_batch":
            return self._to_vin_batch_from_parts(
                efm_snippet=vin_snippet if vin_snippet is not None else efm_snippet,
                depths=depths,
                rri=rri,
                scene_id=scene_id,
                snippet_id=snippet_id,
                backbone_out=backbone_out,
            )

        return OracleRriCacheSample(
            key=entry.key,
            scene_id=scene_id,
            snippet_id=snippet_id,
            candidates=candidates,
            depths=depths,
            candidate_pcs=pcs,
            rri=rri,
            backbone_out=backbone_out,
            efm_snippet_view=efm_snippet,
        )

    def _to_vin_batch_from_parts(
        self,
        *,
        efm_snippet: EfmSnippetView | VinSnippetView | None,
        depths: CandidateDepths,
        rri: RriResult,
        scene_id: str,
        snippet_id: str,
        backbone_out: EvlBackboneOutput | None,
    ) -> "VinOracleBatch":
        """Assemble a VIN batch from decoded cache payload parts."""
        from .vin_oracle_types import VinOracleBatch

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
            scene_id=scene_id,
            snippet_id=snippet_id,
            backbone_out=backbone_out,
        )

    def _filter_index_for_vin_snippet_cache(
        self,
        entries: list[OracleRriCacheEntry],
    ) -> list[OracleRriCacheEntry]:
        """Restrict oracle entries to the subset available in a VIN snippet cache."""
        if not self.config.vin_snippet_cache_allow_subset:
            return entries
        if self.config.return_format != "vin_batch":
            return entries
        if not self.config.include_efm_snippet:
            return entries
        if self.config.vin_snippet_cache_mode == "disabled":
            return entries
        if self.config.vin_snippet_cache is None:
            self.console.warn(
                "vin_snippet_cache_allow_subset=True but vin_snippet_cache is None; skipping VIN snippet filtering.",
            )
            return entries
        available = _read_vin_snippet_cache_index(
            self.config.vin_snippet_cache.index_path,
            allow_missing=True,
        )
        if not available:
            self.console.warn(
                "vin_snippet_cache_allow_subset=True but VIN snippet cache index is empty; "
                "dataset will be empty until cache entries are created.",
            )
            return []
        filtered = [entry for entry in entries if (entry.scene_id, entry.snippet_id) in available]
        if len(filtered) != len(entries):
            self.console.log(
                f"Filtered cache entries to VIN snippet cache subset: {len(filtered)}/{len(entries)} available.",
            )
        return filtered

    def _load_index(self) -> list[OracleRriCacheEntry]:
        """Load and validate the base or split-specific oracle-cache index."""
        if self.config.split == "all":
            entries = read_index(self.config.cache.index_path, entry_type=OracleRriCacheEntry)
            return self._filter_index_for_vin_snippet_cache(entries)

        base_entries = read_index(self.config.cache.index_path, entry_type=OracleRriCacheEntry)
        train_entries, val_entries = validate_oracle_split_indices(
            base_entries=base_entries,
            train_index_path=self.config.cache.train_index_path,
            val_index_path=self.config.cache.val_index_path,
        )
        entries = train_entries if self.config.split == "train" else val_entries
        return self._filter_index_for_vin_snippet_cache(entries)


class OracleRriCacheVinDataset(Dataset["VinOracleBatch"]):
    """VIN-focused wrapper over the oracle cache that always yields VinOracleBatch."""

    is_map_style: bool = True
    """Whether the wrapped dataset supports random access and batching."""

    def __init__(self, config: OracleRriCacheDatasetConfig) -> None:
        """Wrap the oracle-cache reader in VIN-batch-only mode."""
        if not config.load_depths:
            raise ValueError("OracleRriCacheVinDataset requires load_depths=True.")
        cfg = config.model_copy(deep=True)
        cfg.return_format = "vin_batch"
        cfg.load_candidates = False
        cfg.load_candidate_pcs = False
        self.config = cfg
        self._dataset = OracleRriCacheDataset(cfg)

    def __len__(self) -> int:
        """Return the number of VIN batches exposed by the wrapped dataset."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> "VinOracleBatch":
        """Return one VIN batch from the wrapped oracle-cache reader."""
        batch = self._dataset[idx]
        return batch  # type: ignore[return-value]

    def __iter__(self) -> Iterator["VinOracleBatch"]:
        """Iterate VIN batches from the wrapped oracle-cache reader."""
        for idx in range(len(self)):
            yield self[idx]


__all__ = ["OracleRriCacheDataset", "OracleRriCacheVinDataset"]
