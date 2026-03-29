"""VIN oracle dataset variants and config factories."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypeAlias

import torch
from pydantic import Field
from torch.utils.data import IterableDataset

from ..configs import PathConfig
from ..pipelines.oracle_rri_labeler import OracleRriLabeler, OracleRriLabelerConfig
from ..utils import BaseConfig, Console, Stage, Verbosity
from .efm_dataset import AseEfmDatasetConfig
from .efm_views import EfmSnippetView
from .offline_cache import OracleRriCacheDatasetConfig, OracleRriCacheVinDataset
from .vin_oracle_types import VinOracleBatch


class VinOracleOnlineDataset(IterableDataset[VinOracleBatch]):
    """Iterable dataset yielding :class:`VinOracleBatch` with online oracle labels."""

    is_map_style: bool = False

    def __init__(
        self,
        *,
        base: IterableDataset[EfmSnippetView],
        labeler: OracleRriLabeler,
        max_attempts_per_batch: int,
        verbosity: Verbosity,
        efm_keep_keys: set[str] | None,
    ) -> None:
        super().__init__()
        self._base = base
        self._labeler = labeler
        self._max_attempts = int(max_attempts_per_batch)
        self._console = Console.with_prefix(self.__class__.__name__).set_verbosity(
            verbosity,
        )
        self._efm_keep_keys = efm_keep_keys

    def __iter__(self) -> Iterator[VinOracleBatch]:
        base_iter = iter(self._base)
        attempts = 0
        while True:
            sample = next(base_iter)
            try:
                label_batch = self._labeler.run(sample)
            except ValueError as exc:
                attempts += 1
                self._console.warn(
                    f"skip: scene={sample.scene_id} snip={sample.snippet_id} err={exc}",
                )
                if attempts >= self._max_attempts:
                    raise RuntimeError(
                        f"Exceeded max_attempts_per_batch={self._max_attempts} without a valid oracle label batch.",
                    ) from exc
                continue

            attempts = 0
            oracle_rri = label_batch.rri.rri.detach()
            if oracle_rri.numel() == 0 or not torch.isfinite(oracle_rri).any():
                self._console.warn(
                    f"skip: empty/non-finite rri scene={sample.scene_id} snip={sample.snippet_id}",
                )
                continue

            yield VinOracleBatch.from_label(
                label_batch,
                efm_keep_keys=self._efm_keep_keys,
            )


def _default_online_train_ds() -> AseEfmDatasetConfig:
    return AseEfmDatasetConfig(
        load_meshes=True,
        require_mesh=True,
        batch_size=1,
        verbosity=Verbosity.QUIET,
        is_debug=False,
        wds_shuffle=True,
    )


class VinOracleOnlineDatasetConfig(BaseConfig[VinOracleOnlineDataset]):
    """Configuration for online oracle VIN datasets."""

    kind: Literal["online"] = "online"
    """Discriminator for online datasets."""

    @property
    def target(self) -> type[VinOracleOnlineDataset]:
        """Factory target for :meth:`BaseConfig.setup_target`."""
        return VinOracleOnlineDataset

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    dataset: AseEfmDatasetConfig = Field(default_factory=_default_online_train_ds)
    """EFM dataset configuration (must provide meshes for oracle labels)."""

    train_overrides: dict[str, Any] | None = None
    """Optional field overrides applied for the train split."""

    val_overrides: dict[str, Any] | None = Field(
        default_factory=lambda: {"wds_shuffle": False, "wds_repeat": False},
    )
    """Optional field overrides applied for val/test splits."""

    labeler: OracleRriLabelerConfig = Field(default_factory=OracleRriLabelerConfig)
    """Oracle labeler configuration (candidates, rendering, RRI)."""

    max_attempts_per_batch: int = 50
    """Maximum oracle attempts before raising."""

    efm_keep_keys: list[str] | None = None
    """Optional allowlist of EFM keys to keep in VIN batches."""

    prune_efm_snippet: bool = True
    """Whether to prune EFM snippets before returning VIN batches."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Verbosity level for dataset/labeler diagnostics."""

    def setup_target(self, *, split: Stage) -> VinOracleOnlineDataset:  # type: ignore[override]
        dataset_cfg = self._resolve_dataset_cfg(split)
        base = dataset_cfg.setup_target()
        labeler = self.labeler.setup_target()
        keep_keys = None
        if self.prune_efm_snippet and self.efm_keep_keys:
            keep_keys = {key for key in self.efm_keep_keys if key}
        return VinOracleOnlineDataset(
            base=base,
            labeler=labeler,
            max_attempts_per_batch=self.max_attempts_per_batch,
            verbosity=self.verbosity,
            efm_keep_keys=keep_keys,
        )

    def _resolve_dataset_cfg(self, split: Stage) -> AseEfmDatasetConfig:
        dataset_cfg = self.dataset.model_copy(deep=True)
        dataset_cfg.paths = self.paths
        overrides = self.train_overrides if split is Stage.TRAIN else self.val_overrides
        if overrides:
            dataset_cfg = dataset_cfg.model_copy(deep=True, update=overrides)
        return dataset_cfg

    @property
    def is_map_style(self) -> bool:
        return False


class VinOracleCacheDatasetConfig(BaseConfig[OracleRriCacheVinDataset]):
    """Configuration for offline cached VIN datasets."""

    kind: Literal["offline_cache"] = "offline_cache"
    """Discriminator for offline cache datasets."""

    @property
    def target(self) -> type[OracleRriCacheVinDataset]:
        """Factory target for :meth:`BaseConfig.setup_target`."""
        return OracleRriCacheVinDataset

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver."""

    cache: OracleRriCacheDatasetConfig = Field(default_factory=OracleRriCacheDatasetConfig)
    """Oracle cache dataset configuration (return_format is forced to VIN batches)."""

    train_split: Literal["train", "val", "all"] = "all"
    """Cache split to use for training."""

    val_split: Literal["train", "val", "all"] = "all"
    """Cache split to use for validation/testing."""

    def setup_target(self, *, split: Stage) -> OracleRriCacheVinDataset:  # type: ignore[override]
        cache_split = self.train_split if split is Stage.TRAIN else self.val_split
        cache_cfg = self.cache.model_copy(deep=True)
        cache_cfg.paths = self.paths
        cache_cfg.cache.paths = self.paths
        cache_cfg.split = cache_split
        return OracleRriCacheVinDataset(cache_cfg)

    @property
    def is_map_style(self) -> bool:
        return True


VinOracleDatasetConfig: TypeAlias = Annotated[
    VinOracleOnlineDatasetConfig | VinOracleCacheDatasetConfig,
    Field(discriminator="kind"),
]


__all__ = [
    "VinOracleCacheDatasetConfig",
    "VinOracleDatasetConfig",
    "VinOracleOnlineDataset",
    "VinOracleOnlineDatasetConfig",
]
