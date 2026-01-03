"""LightningDataModule for VIN training with online or cached oracle labels.

The training data-flow mirrors `oracle_rri/scripts/train_vin.py`:

EFM snippet → candidate generation → depth rendering → backprojection → oracle RRI → VIN (CORAL).

This module keeps the expensive oracle labeler in the data pipeline by default,
but can switch to cached oracle outputs for fast parallel reading.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from efm3d.aria.aria_constants import (
    ARIA_POINTS_DIST_STD,
    ARIA_POINTS_INV_DIST_STD,
    ARIA_POINTS_TIME_NS,
    ARIA_POINTS_VOL_MAX,
    ARIA_POINTS_VOL_MIN,
    ARIA_POINTS_WORLD,
    ARIA_POSE_T_WORLD_RIG,
    ARIA_POSE_TIME_NS,
)
from efm3d.aria.pose import PoseTW
from pydantic import Field, model_validator
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,  # type: ignore[import-untyped]
)
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..configs import PathConfig
from ..data import AseEfmDatasetConfig, EfmSnippetView
from ..data.offline_cache import (
    OracleRriCacheAppender,
    OracleRriCacheAppenderConfig,
    OracleRriCacheDataset,
    OracleRriCacheDatasetConfig,
)
from ..pipelines.oracle_rri_labeler import (
    OracleRriLabelBatch,
    OracleRriLabeler,
    OracleRriLabelerConfig,
)
from ..utils import BaseConfig, Console, Stage, Verbosity
from ..vin.backbone_evl import EvlBackbone, EvlBackboneConfig
from ..vin.types import EvlBackboneOutput

Tensor = torch.Tensor


@dataclass(slots=True)
class VinOracleBatch:
    """Single-snippet VIN training batch produced from an oracle label run.

    Attributes:
        efm_snippet_view: Typed EFM snippet view (None when loading from cache).
        candidate_poses_world_cam: ``PoseTW["N 12"]`` candidate poses as world←camera for the rendered subset.
        reference_pose_world_rig: ``PoseTW["12"]`` reference pose as world←rig_reference for the snippet.
        rri: ``Tensor["N", float32]`` oracle RRI per candidate (same ordering as candidates).
        pm_dist_before: ``Tensor["N", float32]`` Chamfer-style point↔mesh distance before (broadcasted).
        pm_dist_after: ``Tensor["N", float32]`` Chamfer-style point↔mesh distance after (per-candidate).
        pm_acc_before: ``Tensor["N", float32]`` point→mesh accuracy distance before (broadcasted).
        pm_comp_before: ``Tensor["N", float32]`` mesh→point completeness distance before (broadcasted).
        pm_acc_after: ``Tensor["N", float32]`` point→mesh accuracy distance after (per-candidate).
        pm_comp_after: ``Tensor["N", float32]`` mesh→point completeness distance after (per-candidate).
        p3d_cameras: PyTorch3D cameras used for depth rendering/unprojection (same ordering as candidates).
        scene_id: ASE scene id for diagnostics.
        snippet_id: Snippet id (tar key/url stem) for diagnostics.
        backbone_out: Optional cached EVL backbone outputs.
    """

    efm_snippet_view: EfmSnippetView | None
    """Optional typed snippet view (None when loading from cache)."""
    candidate_poses_world_cam: PoseTW
    reference_pose_world_rig: PoseTW
    rri: Tensor
    pm_dist_before: Tensor
    pm_dist_after: Tensor
    pm_acc_before: Tensor
    pm_comp_before: Tensor
    pm_acc_after: Tensor
    pm_comp_after: Tensor
    p3d_cameras: PerspectiveCameras
    scene_id: str
    snippet_id: str
    backbone_out: EvlBackboneOutput | None = None
    """Optional cached EVL backbone outputs (used to skip backbone inference)."""


class VinOracleIterableDataset(IterableDataset[VinOracleBatch]):
    """Iterable dataset yielding :class:`VinOracleBatch` with *online* oracle RRI labels."""

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

            yield _vin_oracle_batch_from_label(
                label_batch,
                efm_keep_keys=self._efm_keep_keys,
            )


def _vin_oracle_batch_from_label(
    label_batch: OracleRriLabelBatch,
    *,
    efm_keep_keys: set[str] | None,
) -> VinOracleBatch:
    rri = label_batch.rri
    sample = label_batch.sample
    if efm_keep_keys is not None:
        sample = sample.prune_efm(efm_keep_keys)

    return VinOracleBatch(
        efm_snippet_view=sample,
        candidate_poses_world_cam=label_batch.depths.poses,
        reference_pose_world_rig=label_batch.depths.reference_pose,
        rri=rri.rri,
        pm_dist_before=rri.pm_dist_before,
        pm_dist_after=rri.pm_dist_after,
        pm_acc_before=rri.pm_acc_before,
        pm_comp_before=rri.pm_comp_before,
        pm_acc_after=rri.pm_acc_after,
        pm_comp_after=rri.pm_comp_after,
        p3d_cameras=label_batch.depths.p3d_cameras,
        scene_id=sample.scene_id,
        snippet_id=sample.snippet_id,
        backbone_out=None,
    )


class VinOracleCacheAppendIterableDataset(IterableDataset[VinOracleBatch]):
    """Iterable dataset that yields cached batches and appends new online samples."""

    def __init__(
        self,
        *,
        cache: OracleRriCacheDataset,
        base: IterableDataset[EfmSnippetView],
        labeler: OracleRriLabeler,
        appender: OracleRriCacheAppender,
        backbone: EvlBackbone | None,
        max_new_samples: int,
        max_attempts_per_batch: int,
        verbosity: Verbosity,
        efm_keep_keys: set[str] | None,
    ) -> None:
        super().__init__()
        self._cache = cache
        self._base = base
        self._labeler = labeler
        self._appender = appender
        self._backbone = backbone
        self._max_new_samples = int(max_new_samples)
        self._max_attempts = int(max_attempts_per_batch)
        self._console = Console.with_prefix(self.__class__.__name__).set_verbosity(
            verbosity,
        )
        self._efm_keep_keys = efm_keep_keys

    def __iter__(self) -> Iterator[VinOracleBatch]:
        for idx in range(len(self._cache)):
            yield self._cache[idx]

        if self._max_new_samples <= 0:
            return

        base_iter = iter(self._base)
        attempts = 0
        appended = 0
        while appended < self._max_new_samples:
            try:
                sample = next(base_iter)
            except StopIteration:
                return

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

            backbone_out = None
            if self._backbone is not None:
                backbone_out = self._backbone.forward(sample.efm)

            entry = self._appender.append(label_batch, backbone_out=backbone_out)
            self._cache.append_entry(entry)

            batch = _vin_oracle_batch_from_label(
                label_batch,
                efm_keep_keys=self._efm_keep_keys,
            )
            batch.backbone_out = backbone_out
            yield batch
            appended += 1


def _default_train_ds() -> AseEfmDatasetConfig:
    return AseEfmDatasetConfig(
        load_meshes=True,
        require_mesh=True,
        batch_size=1,
        verbosity=Verbosity.QUIET,
        is_debug=False,
        wds_shuffle=True,
    )


def _default_val_ds() -> AseEfmDatasetConfig:
    return AseEfmDatasetConfig(
        load_meshes=True,
        require_mesh=True,
        batch_size=1,
        verbosity=Verbosity.QUIET,
        is_debug=False,
    )


class VinDataModuleConfig(BaseConfig["VinDataModule"]):
    """Configuration for :class:`VinDataModule`."""

    target: type[VinDataModule] = Field(
        default_factory=lambda: VinDataModule,
        exclude=True,
    )

    paths: PathConfig = Field(default_factory=PathConfig)
    """Project path resolver (propagated to cache configs)."""

    train_dataset: AseEfmDatasetConfig = Field(default_factory=_default_train_ds)
    """Training dataset configuration (must provide meshes for oracle labels)."""

    val_dataset: AseEfmDatasetConfig = Field(default_factory=_default_val_ds)
    """Validation dataset configuration (must provide meshes for oracle labels)."""

    labeler: OracleRriLabelerConfig = Field(default_factory=OracleRriLabelerConfig)
    """Oracle labeler configuration (candidates, rendering, RRI)."""

    train_cache: OracleRriCacheDatasetConfig | None = Field(
        default_factory=lambda: OracleRriCacheDatasetConfig(),
    )
    """Optional offline cache for training (skips online label generation)."""

    val_cache: OracleRriCacheDatasetConfig | None = None
    """Optional offline cache for validation/testing."""

    shuffle: bool = True
    """Whether to shuffle the train dataset at each epoch (only applies to offline caches)."""

    cache_backbone: EvlBackboneConfig | None = Field(default_factory=EvlBackboneConfig)
    """EVL backbone config used when appending new cache samples."""

    train_cache_new_samples_per_epoch: int = 0
    """Number of new online samples to append to the train cache per epoch."""

    val_cache_new_samples_per_epoch: int = 0
    """Number of new online samples to append to the val cache per epoch."""

    cache_append_allow_mismatch: bool = False
    """Allow appending samples even when cache metadata signatures mismatch."""

    cache_append_create_if_missing: bool = False
    """Create cache metadata/index if missing when appending samples."""

    efm_keep_keys: list[str] | None = Field(
        default_factory=lambda: [
            ARIA_POSE_T_WORLD_RIG,
            ARIA_POSE_TIME_NS,
            "pose/gravity_in_world",
            ARIA_POINTS_WORLD,
            ARIA_POINTS_DIST_STD,
            ARIA_POINTS_INV_DIST_STD,
            ARIA_POINTS_TIME_NS,
            ARIA_POINTS_VOL_MIN,
            ARIA_POINTS_VOL_MAX,
            "points/lengths",
        ]
    )
    """Optional allowlist of EFM keys to keep in VIN batches."""

    prune_efm_snippet: bool = True
    """Whether to prune EFM snippets before returning VIN batches."""

    max_attempts_per_batch: int = 50
    """Maximum oracle attempts before raising (guards against overly strict sampling rules)."""

    num_workers: int = 16
    """Number of DataLoader worker processes (use >0 for offline caches; keep 0 for online labeler)."""

    batch_size: int | None = None

    persistent_workers: bool = False
    """Whether to keep DataLoader workers alive between epochs (ignored when num_workers=0)."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Verbosity level for dataset/labeler diagnostics."""

    is_debug: bool = False
    """Enable debug defaults (forces num_workers=0, lowers verbosity)."""

    use_train_as_val: bool = False

    @property
    def needs_labeler(self) -> bool:
        """Whether this config requires an OracleRriLabeler to generate labels."""
        return (
            self.train_cache is None
            or self.train_cache_new_samples_per_epoch > 0
            or self.val_cache_new_samples_per_epoch > 0
        )

    @model_validator(mode="after")
    def _check_compatibility(self) -> VinDataModuleConfig:
        cache_split_enabled = bool(
            self.train_cache is not None and 0.0 < float(self.train_cache.train_val_split) < 1.0,
        )
        if self.use_train_as_val and not cache_split_enabled:
            Console.with_prefix(self.__class__.__name__, "_check_compatibility").warn(
                "use_train_as_val is enabled; validation will use the training dataset."
            )
        if self.needs_labeler and self.num_workers > 0:
            raise ValueError(
                "OracleRriLabeler is not multiprocess-safe; set num_workers=0 when online labeling is needed.",
            )
        if self.needs_labeler and self.batch_size is not None:
            raise ValueError(
                "OracleRriLabeler only supports batch_size=None; do not set batch_size when online labeling is needed.",
            )
        if self.batch_size is not None:
            raise NotImplementedError(
                "VinDataModule only supports batch_size=None; do not set batch_size. Need a custom collate_fn to batch variable-length candidate sets.",
            )
        return self


class VinDataModule(pl.LightningDataModule):
    """LightningDataModule that yields online or cached oracle-labelled VIN batches."""

    _train_base: IterableDataset[EfmSnippetView] | None
    """Optional online dataset used to generate oracle labels for training."""

    _val_base: IterableDataset[EfmSnippetView] | None
    """Optional online dataset used to generate oracle labels for validation/testing."""

    _labeler: OracleRriLabeler | None
    """Oracle labeler used for online label generation or cache appending."""

    _train_cache: OracleRriCacheDataset | None
    """Optional offline cache backing the training dataloader."""

    _val_cache: OracleRriCacheDataset | None
    """Optional offline cache backing the validation/test dataloader."""

    _cache_backbone: EvlBackbone | None
    """Optional EVL backbone - wrapper around frozen EVL for computing and caching scene features."""

    _train_cache_appender: OracleRriCacheAppender | None
    """Cache appender for adding new training samples to the offline cache."""

    _val_cache_appender: OracleRriCacheAppender | None
    """Cache appender for adding new validation samples to the offline cache."""

    def __init__(self, config: VinDataModuleConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump_jsonable())

        self._train_base: IterableDataset[EfmSnippetView] | None = None
        self._val_base: IterableDataset[EfmSnippetView] | None = None
        self._labeler: OracleRriLabeler | None = None
        self._train_cache: OracleRriCacheDataset | None = None
        self._val_cache: OracleRriCacheDataset | None = None
        self._cache_backbone: EvlBackbone | None = None
        self._train_cache_appender: OracleRriCacheAppender | None = None
        self._val_cache_appender: OracleRriCacheAppender | None = None
        self._efm_keep_keys: set[str] | None = None

    def _apply_cache_splits(self, console: Console) -> None:
        train_cache = self.config.train_cache
        if train_cache is None:
            return

        split_fraction = float(train_cache.train_val_split)
        if split_fraction <= 0.0 or split_fraction >= 1.0:
            return

        val_cache = self.config.val_cache
        if val_cache is None:
            val_cache = train_cache.model_copy(deep=True)
            val_cache.split = "val"
            val_cache.train_val_split = split_fraction
            self.config.val_cache = val_cache
            console.log(
                f"Derived val_cache from train_cache (train_val_split={split_fraction:.3f}).",
            )

        if val_cache is None or val_cache.cache.cache_dir != train_cache.cache.cache_dir:
            return

        if train_cache.split == "all":
            train_cache.split = "train"
        if val_cache.split == "all":
            val_cache.split = "val"

        if val_cache.train_val_split != train_cache.train_val_split:
            console.warn(
                "val_cache.train_val_split differs from train_cache; syncing to training split.",
            )
            val_cache.train_val_split = train_cache.train_val_split

        if self.config.use_train_as_val:
            self.config.use_train_as_val = False
            console.warn(
                "Disabling use_train_as_val because train/val cache split is enabled.",
            )

    def _cache_with_simplification(
        self,
        cache_cfg: OracleRriCacheDatasetConfig,
        simplification: float,
    ) -> OracleRriCacheDataset:
        cfg = cache_cfg.model_copy(deep=True)
        cfg.simplification = simplification
        cfg.return_format = "vin_batch"
        if self._efm_keep_keys is not None and cfg.efm_keep_keys is None:
            cfg.efm_keep_keys = sorted(self._efm_keep_keys)
        return cfg.setup_target()

    # --------------------------------------------------------------------- setup
    def setup(self, stage: Stage | str | None = None) -> None:
        console = Console.with_prefix(self.__class__.__name__, "setup")
        self._apply_cache_splits(console)
        keep_keys_list = None
        if self.config.prune_efm_snippet and self.config.efm_keep_keys:
            keep_keys_list = [key for key in self.config.efm_keep_keys if key]
        self._efm_keep_keys = set(keep_keys_list) if keep_keys_list else None
        if self.config.train_cache is not None:
            self.config.train_cache.return_format = "vin_batch"
            if self._efm_keep_keys is not None and self.config.train_cache.efm_keep_keys is None:
                self.config.train_cache.efm_keep_keys = list(keep_keys_list or [])
        if self.config.val_cache is not None:
            self.config.val_cache.return_format = "vin_batch"
            if self._efm_keep_keys is not None and self.config.val_cache.efm_keep_keys is None:
                self.config.val_cache.efm_keep_keys = list(keep_keys_list or [])

        requested = Stage.from_str(stage) if stage is not None else None
        train_append = self.config.train_cache_new_samples_per_epoch > 0
        val_append = self.config.val_cache_new_samples_per_epoch > 0

        if train_append and self.config.train_cache is None:
            raise ValueError(
                "train_cache_new_samples_per_epoch requires train_cache to be set.",
            )
        if val_append and self.config.val_cache is None:
            raise ValueError(
                "val_cache_new_samples_per_epoch requires val_cache to be set.",
            )
        if (train_append or val_append) and self.config.num_workers > 0:
            raise ValueError(
                "Appending to cache requires num_workers=0 (single-process writes).",
            )

        # needs the OracleRriLabeler to generate new labels?
        if self.config.needs_labeler and self._labeler is None:
            self._labeler = self.config.labeler.setup_target()
            console.log("Initialized OracleRriLabeler for online label generation.")

        if (train_append or val_append) and self._cache_backbone is None:
            if self.config.cache_backbone is None:
                raise ValueError(
                    "cache_backbone must be set when appending new cache samples.",
                )
            self._cache_backbone = self.config.cache_backbone.setup_target()
            console.log("Initialized EVL backbone for cache sample appending.")

        if requested is None or requested is Stage.TRAIN:
            if self.config.train_cache is not None:
                if self._train_cache is None:
                    self._train_cache = self.config.train_cache.setup_target()
                if train_append and self._train_base is None:
                    self._train_base = self.config.train_dataset.setup_target()
                if train_append and self._train_cache_appender is None:
                    appender_cfg = OracleRriCacheAppenderConfig(
                        paths=self.config.paths,
                        cache=self.config.train_cache.cache,
                        labeler=self.config.labeler,
                        dataset=self.config.train_dataset,
                        backbone=self.config.cache_backbone,
                        include_backbone=True,
                        include_depths=True,
                        include_pointclouds=True,
                        allow_mismatch=self.config.cache_append_allow_mismatch,
                        create_if_missing=self.config.cache_append_create_if_missing,
                        verbosity=self.config.verbosity,
                    )
                    self._train_cache_appender = appender_cfg.setup_target()
            elif self._train_base is None:
                self._train_base = self.config.train_dataset.setup_target()

        if requested is None or requested in (Stage.VAL, Stage.TEST):
            if self.config.val_cache is not None:
                if not self.config.val_cache.load_backbone:
                    raise ValueError(
                        "val_cache.load_backbone must be True for VIN validation/testing.",
                    )
                if self._val_cache is None:
                    self._val_cache = self.config.val_cache.setup_target()
                if val_append and self._val_base is None:
                    self._val_base = self.config.val_dataset.setup_target()
                if val_append and self._val_cache_appender is None:
                    appender_cfg = OracleRriCacheAppenderConfig(
                        paths=self.config.paths,
                        cache=self.config.val_cache.cache,
                        labeler=self.config.labeler,
                        dataset=self.config.val_dataset,
                        backbone=self.config.cache_backbone,
                        include_backbone=True,
                        include_depths=True,
                        include_pointclouds=True,
                        allow_mismatch=self.config.cache_append_allow_mismatch,
                        create_if_missing=self.config.cache_append_create_if_missing,
                        verbosity=self.config.verbosity,
                    )
                    self._val_cache_appender = appender_cfg.setup_target()
            elif self._val_base is None:
                self._val_base = self.config.val_dataset.setup_target()

    # ------------------------------------------------------------------ loaders
    def train_dataloader(self) -> DataLoader:
        console = Console.with_prefix(self.__class__.__name__, "train-dataloader")
        self.setup(stage=Stage.TRAIN)
        _offline_ds = False
        if self._train_cache is not None:
            ds: Dataset[VinOracleBatch] | IterableDataset[VinOracleBatch]
            if self.config.train_cache_new_samples_per_epoch > 0:
                assert self._train_base is not None
                assert self._train_cache_appender is not None
                ds = VinOracleCacheAppendIterableDataset(
                    cache=self._train_cache,
                    base=self._train_base,
                    labeler=self._require_labeler(),
                    appender=self._train_cache_appender,
                    backbone=self._cache_backbone,
                    max_new_samples=self.config.train_cache_new_samples_per_epoch,
                    max_attempts_per_batch=self.config.max_attempts_per_batch,
                    verbosity=self.config.verbosity,
                    efm_keep_keys=self._efm_keep_keys,
                )
                console.log(
                    f"Training with cache append: new_samples_per_epoch={self.config.train_cache_new_samples_per_epoch}.",
                )
            else:
                _offline_ds = True
                ds = self._train_cache
                console.log("Training with offline samples only.")
        else:
            assert self._train_base is not None
            ds = VinOracleIterableDataset(
                base=self._train_base,
                labeler=self._require_labeler(),
                max_attempts_per_batch=self.config.max_attempts_per_batch,
                verbosity=self.config.verbosity,
                efm_keep_keys=self._efm_keep_keys,
            )
            console.log(
                "Training with online oracle label generation and computation of backbone features.",
            )
        return DataLoader(
            ds,
            batch_size=None,
            shuffle=self.config.shuffle if _offline_ds else False,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        self.setup(stage=Stage.VAL)
        if self._val_cache is not None and not self.config.use_train_as_val:
            ds: Dataset[VinOracleBatch] | IterableDataset[VinOracleBatch]
            if self.config.val_cache_new_samples_per_epoch > 0:
                assert self._val_base is not None
                assert self._val_cache_appender is not None
                ds = VinOracleCacheAppendIterableDataset(
                    cache=self._val_cache,
                    base=self._val_base,
                    labeler=self._require_labeler(),
                    appender=self._val_cache_appender,
                    backbone=self._cache_backbone,
                    max_new_samples=self.config.val_cache_new_samples_per_epoch,
                    max_attempts_per_batch=self.config.max_attempts_per_batch,
                    verbosity=self.config.verbosity,
                    efm_keep_keys=self._efm_keep_keys,
                )
            else:
                ds = self._val_cache
        elif self.config.use_train_as_val:
            assert self._train_cache is not None
            ds = self._cache_with_simplification(self.config.train_cache, simplification=0.01)
        else:
            assert self._val_base is not None
            ds = VinOracleIterableDataset(
                base=self._val_base,
                labeler=self._require_labeler(),
                max_attempts_per_batch=self.config.max_attempts_per_batch,
                verbosity=self.config.verbosity,
                efm_keep_keys=self._efm_keep_keys,
            )
        return DataLoader(
            ds,
            batch_size=None,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    # ------------------------------------------------------------------ helpers
    def iter_oracle_batches(self, *, stage: Stage) -> Iterator[VinOracleBatch]:
        """Iterate oracle batches without going through a DataLoader."""
        self.setup(stage=stage)
        if stage is Stage.TRAIN and self._train_cache is not None:
            if self.config.train_cache_new_samples_per_epoch > 0:
                assert self._train_base is not None
                assert self._train_cache_appender is not None
                ds = VinOracleCacheAppendIterableDataset(
                    cache=self._train_cache,
                    base=self._train_base,
                    labeler=self._require_labeler(),
                    appender=self._train_cache_appender,
                    backbone=self._cache_backbone,
                    max_new_samples=self.config.train_cache_new_samples_per_epoch,
                    max_attempts_per_batch=self.config.max_attempts_per_batch,
                    verbosity=self.config.verbosity,
                    efm_keep_keys=self._efm_keep_keys,
                )
                return iter(ds)
            return iter(self._train_cache)
        if stage in (Stage.VAL, Stage.TEST) and self._val_cache is not None:
            if self.config.val_cache_new_samples_per_epoch > 0:
                assert self._val_base is not None
                assert self._val_cache_appender is not None
                ds = VinOracleCacheAppendIterableDataset(
                    cache=self._val_cache,
                    base=self._val_base,
                    labeler=self._require_labeler(),
                    appender=self._val_cache_appender,
                    backbone=self._cache_backbone,
                    max_new_samples=self.config.val_cache_new_samples_per_epoch,
                    max_attempts_per_batch=self.config.max_attempts_per_batch,
                    verbosity=self.config.verbosity,
                    efm_keep_keys=self._efm_keep_keys,
                )
                return iter(ds)
            return iter(self._val_cache)

        base = self._train_base if stage is Stage.TRAIN else self._val_base
        if base is None:
            raise RuntimeError(
                f"Missing base dataset for stage={stage}. Did setup() run?",
            )
        ds = VinOracleIterableDataset(
            base=base,
            labeler=self._require_labeler(),
            max_attempts_per_batch=self.config.max_attempts_per_batch,
            verbosity=self.config.verbosity,
            efm_keep_keys=self._efm_keep_keys,
        )
        return iter(ds)

    def _require_labeler(self) -> OracleRriLabeler:
        if self._labeler is None:
            raise RuntimeError("Oracle labeler not initialized. Call setup() first.")
        return self._labeler


__all__ = ["VinDataModule", "VinDataModuleConfig", "VinOracleBatch"]
