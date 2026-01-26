"""Providers for minimal VIN snippet views."""

from __future__ import annotations

from typing import Any, Literal, Protocol

import torch

from ..configs import PathConfig
from ..utils import Console
from .efm_snippet_loader import EfmSnippetLoader
from .efm_views import VinSnippetView
from .offline_cache_store import _extract_snippet_token
from .vin_snippet_cache import (
    VinSnippetCacheConfig,
    VinSnippetCacheDataset,
    VinSnippetCacheDatasetConfig,
    read_vin_snippet_cache_metadata,
)
from .vin_snippet_utils import build_vin_snippet_view, vin_snippet_cache_config_hash


class VinSnippetProvider(Protocol):
    """Protocol for loading minimal VIN snippets."""

    def get(
        self,
        *,
        scene_id: str,
        snippet_id: str,
        map_location: str,
    ) -> VinSnippetView | None:
        """Return a VIN snippet view or None if unavailable."""


class VinSnippetCacheProvider:
    """VIN snippet provider backed by a precomputed cache."""

    def __init__(
        self,
        *,
        cache: VinSnippetCacheConfig,
        expected_config_hash: str | None,
        mode: Literal["auto", "required", "disabled"],
        console: Console,
    ) -> None:
        self._cache = cache
        self._expected_config_hash = expected_config_hash
        self._mode = mode
        self._console = console
        self._datasets: dict[str, VinSnippetCacheDataset] = {}
        self._disabled = mode == "disabled"
        self._validated = False

    def _ensure_dataset(self, map_location: str) -> VinSnippetCacheDataset | None:
        if self._disabled:
            return None
        dataset = self._datasets.get(map_location)
        if dataset is None:
            cfg = VinSnippetCacheDatasetConfig(
                cache=self._cache,
                map_location=map_location,
            )
            dataset = cfg.setup_target()
            self._datasets[map_location] = dataset
        return dataset

    def _validate_metadata(self) -> None:
        if self._validated or self._disabled:
            return
        meta = read_vin_snippet_cache_metadata(self._cache.metadata_path)
        expected = self._expected_config_hash
        meta_hash = meta.config_hash or vin_snippet_cache_config_hash(
            dataset_config=meta.dataset_config,
            include_inv_dist_std=bool(meta.include_inv_dist_std),
            include_obs_count=bool(meta.include_obs_count),
            semidense_max_points=meta.semidense_max_points,
            pad_points=meta.pad_points,
        )
        if expected is not None and meta_hash != expected:
            msg = "VIN snippet cache config hash mismatch; cache may be stale or built with different settings."
            if self._mode == "required":
                raise ValueError(msg)
            self._console.warn(f"{msg} Disabling vin_snippet_cache.")
            self._disabled = True
        self._validated = True

    def get(
        self,
        *,
        scene_id: str,
        snippet_id: str,
        map_location: str,
    ) -> VinSnippetView | None:
        if self._disabled:
            if self._mode == "required":
                raise FileNotFoundError(
                    "vin_snippet_cache is required but disabled or missing.",
                )
            return None
        try:
            self._validate_metadata()
            dataset = self._ensure_dataset(map_location)
        except FileNotFoundError as exc:
            if self._mode == "required":
                raise
            self._console.warn(
                f"vin_snippet_cache missing files at {self._cache.cache_dir}: {exc}. "
                "Falling back to EFM snippet loading.",
            )
            self._disabled = True
            return None
        if dataset is None:
            return None
        result = dataset.get_by_scene_snippet(
            scene_id=scene_id,
            snippet_id=snippet_id,
            map_location=map_location,
        )
        if result is None:
            snippet_token = _extract_snippet_token(snippet_id)
            if snippet_token != snippet_id:
                result = dataset.get_by_scene_snippet(
                    scene_id=scene_id,
                    snippet_id=snippet_token,
                    map_location=map_location,
                )
        if result is None and self._mode == "required":
            snippet_token = _extract_snippet_token(snippet_id)
            token_suffix = ""
            if snippet_token != snippet_id:
                token_suffix = f" (token={snippet_token})"
            raise FileNotFoundError(
                f"vin_snippet_cache missing entry for scene={scene_id} snippet={snippet_id} "
                f"(cache_dir={self._cache.cache_dir}){token_suffix}.",
            )
        return result


class EfmSnippetProvider:
    """VIN snippet provider that builds views from EFM snippets."""

    def __init__(
        self,
        *,
        dataset_payload: dict[str, Any] | None,
        paths: PathConfig,
        include_gt_mesh: bool,
        semidense_max_points: int | None,
        include_inv_dist_std: bool,
        include_obs_count: bool,
        efm_keep_keys: set[str] | None,
    ) -> None:
        self._dataset_payload = dict(dataset_payload or {})
        self._paths = paths
        self._include_gt_mesh = include_gt_mesh
        self._semidense_max_points = semidense_max_points
        self._include_inv_dist_std = include_inv_dist_std
        self._include_obs_count = include_obs_count
        self._efm_keep_keys = efm_keep_keys
        self._loaders: dict[str, EfmSnippetLoader] = {}

    def _ensure_loader(self, map_location: str) -> EfmSnippetLoader:
        loader = self._loaders.get(map_location)
        if loader is None:
            loader = EfmSnippetLoader(
                dataset_payload=self._dataset_payload,
                device=map_location,
                paths=self._paths,
                include_gt_mesh=self._include_gt_mesh,
            )
            self._loaders[map_location] = loader
        return loader

    def get(
        self,
        *,
        scene_id: str,
        snippet_id: str,
        map_location: str,
    ) -> VinSnippetView | None:
        loader = self._ensure_loader(map_location)
        efm_snippet = loader.load(scene_id=scene_id, snippet_id=snippet_id)
        if self._efm_keep_keys is not None:
            efm_snippet = efm_snippet.prune_efm(self._efm_keep_keys)
        return build_vin_snippet_view(
            efm_snippet,
            device=torch.device(map_location),
            max_points=self._semidense_max_points,
            include_inv_dist_std=self._include_inv_dist_std,
            include_obs_count=self._include_obs_count,
        )


class VinSnippetProviderChain:
    """Try multiple snippet providers in order."""

    def __init__(self, providers: list[VinSnippetProvider]) -> None:
        self._providers = providers

    def get(
        self,
        *,
        scene_id: str,
        snippet_id: str,
        map_location: str,
    ) -> VinSnippetView | None:
        for provider in self._providers:
            result = provider.get(
                scene_id=scene_id,
                snippet_id=snippet_id,
                map_location=map_location,
            )
            if result is not None:
                return result
        return None


def expected_vin_snippet_cache_hash(
    *,
    dataset_config: dict[str, Any] | None,
    include_inv_dist_std: bool,
    include_obs_count: bool,
    semidense_max_points: int | None,
    pad_points: int | None = None,
) -> str:
    """Compute the expected snippet cache hash for compatibility checks."""
    if pad_points is None:
        from .vin_snippet_cache import VIN_SNIPPET_PAD_POINTS

        pad_points = VIN_SNIPPET_PAD_POINTS
    return vin_snippet_cache_config_hash(
        dataset_config=dataset_config,
        include_inv_dist_std=include_inv_dist_std,
        include_obs_count=include_obs_count,
        semidense_max_points=semidense_max_points,
        pad_points=pad_points,
    )


__all__ = [
    "EfmSnippetProvider",
    "VinSnippetProvider",
    "VinSnippetCacheProvider",
    "VinSnippetProviderChain",
    "expected_vin_snippet_cache_hash",
]
