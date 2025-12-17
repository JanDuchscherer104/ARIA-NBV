"""LightningDataModule for VIN training with online oracle labels.

The training data-flow mirrors `oracle_rri/scripts/train_vin.py`:

EFM snippet → candidate generation → depth rendering → backprojection → oracle RRI → VIN (CORAL).

This module keeps the expensive oracle labeler in the data pipeline to enable
end-to-end smoke tests and small-scale training without precomputed caches.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
import torch
from efm3d.aria.pose import PoseTW
from pydantic import Field
from torch.utils.data import DataLoader, IterableDataset

from ..data import AseEfmDatasetConfig, EfmSnippetView
from ..pipelines.oracle_rri_labeler import OracleRriLabelBatch, OracleRriLabeler, OracleRriLabelerConfig
from ..utils import BaseConfig, Console, Stage, Verbosity

Tensor = torch.Tensor


@dataclass(slots=True)
class VinOracleBatch:
    """Single-snippet VIN training batch produced from an oracle label run.

    Attributes:
        efm: Raw EFM snippet dict (zero-copy view over the underlying WebDataset sample).
        candidate_poses_world_cam: ``PoseTW["N 12"]`` candidate poses as world←camera for the rendered subset.
        reference_pose_world_rig: ``PoseTW["12"]`` reference pose as world←rig_reference for the snippet.
        candidate_poses_camera_rig: ``PoseTW["N 12"]`` candidate poses as camera←rig_reference frame (preferred for training).
        rri: ``Tensor["N", float32]`` oracle RRI per candidate (same ordering as candidates).
        pm_dist_before: ``Tensor["N", float32]`` Chamfer-style point↔mesh distance before (broadcasted).
        pm_dist_after: ``Tensor["N", float32]`` Chamfer-style point↔mesh distance after (per-candidate).
        pm_acc_before: ``Tensor["N", float32]`` point→mesh accuracy distance before (broadcasted).
        pm_comp_before: ``Tensor["N", float32]`` mesh→point completeness distance before (broadcasted).
        pm_acc_after: ``Tensor["N", float32]`` point→mesh accuracy distance after (per-candidate).
        pm_comp_after: ``Tensor["N", float32]`` mesh→point completeness distance after (per-candidate).
        scene_id: ASE scene id for diagnostics.
        snippet_id: Snippet id (tar key/url stem) for diagnostics.
    """

    efm: Mapping[str, Any]
    candidate_poses_world_cam: PoseTW
    reference_pose_world_rig: PoseTW
    candidate_poses_camera_rig: PoseTW
    rri: Tensor
    pm_dist_before: Tensor
    pm_dist_after: Tensor
    pm_acc_before: Tensor
    pm_comp_before: Tensor
    pm_acc_after: Tensor
    pm_comp_after: Tensor
    scene_id: str
    snippet_id: str


class VinOracleIterableDataset(IterableDataset[VinOracleBatch]):
    """Iterable dataset yielding :class:`VinOracleBatch` with online oracle RRI labels."""

    def __init__(
        self,
        *,
        base: IterableDataset[EfmSnippetView],
        labeler: OracleRriLabeler,
        max_attempts_per_batch: int,
        verbosity: Verbosity,
    ) -> None:
        super().__init__()
        self._base = base
        self._labeler = labeler
        self._max_attempts = int(max_attempts_per_batch)
        self._console = Console.with_prefix(self.__class__.__name__).set_verbosity(verbosity)

    def __iter__(self) -> Iterator[VinOracleBatch]:
        base_iter = iter(self._base)
        attempts = 0
        while True:
            sample = next(base_iter)
            try:
                label_batch = self._labeler.run(sample)
            except ValueError as exc:
                attempts += 1
                self._console.warn(f"skip: scene={sample.scene_id} snip={sample.snippet_id} err={exc}")
                if attempts >= self._max_attempts:
                    raise RuntimeError(
                        f"Exceeded max_attempts_per_batch={self._max_attempts} without a valid oracle label batch."
                    ) from exc
                continue

            attempts = 0
            oracle_rri = label_batch.rri.rri.detach()
            if oracle_rri.numel() == 0 or not torch.isfinite(oracle_rri).any():
                self._console.warn(f"skip: empty/non-finite rri scene={sample.scene_id} snip={sample.snippet_id}")
                continue

            yield _vin_oracle_batch_from_label(label_batch)


def _vin_oracle_batch_from_label(label_batch: OracleRriLabelBatch) -> VinOracleBatch:
    camera_rig = label_batch.depths.camera.T_camera_rig
    if not isinstance(camera_rig, PoseTW):
        raise TypeError(f"Expected PoseTW for camera.T_camera_rig, got {type(camera_rig)}")

    rri = label_batch.rri

    return VinOracleBatch(
        efm=label_batch.sample.efm,
        candidate_poses_world_cam=label_batch.depths.poses,
        reference_pose_world_rig=label_batch.depths.reference_pose,
        candidate_poses_camera_rig=camera_rig,
        rri=rri.rri,
        pm_dist_before=rri.pm_dist_before,
        pm_dist_after=rri.pm_dist_after,
        pm_acc_before=rri.pm_acc_before,
        pm_comp_before=rri.pm_comp_before,
        pm_acc_after=rri.pm_acc_after,
        pm_comp_after=rri.pm_comp_after,
        scene_id=label_batch.sample.scene_id,
        snippet_id=label_batch.sample.snippet_id,
    )


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

    target: type["VinDataModule"] = Field(default_factory=lambda: VinDataModule, exclude=True)

    train_dataset: AseEfmDatasetConfig = Field(default_factory=_default_train_ds)
    """Training dataset configuration (must provide meshes for oracle labels)."""

    val_dataset: AseEfmDatasetConfig = Field(default_factory=_default_val_ds)
    """Validation dataset configuration (must provide meshes for oracle labels)."""

    labeler: OracleRriLabelerConfig = Field(default_factory=OracleRriLabelerConfig)
    """Oracle labeler configuration (candidates, rendering, RRI)."""

    max_attempts_per_batch: int = 50
    """Maximum oracle attempts before raising (guards against overly strict sampling rules)."""

    num_workers: int = 0
    """Number of DataLoader worker processes (keep at 0; labeler is not multiprocessing-friendly)."""

    persistent_workers: bool = False
    """Whether to keep DataLoader workers alive between epochs (ignored when num_workers=0)."""

    verbosity: Verbosity = Verbosity.NORMAL
    """Verbosity level for dataset/labeler diagnostics."""

    is_debug: bool = False
    """Enable debug defaults (forces num_workers=0, lowers verbosity)."""


class VinDataModule(pl.LightningDataModule):
    """LightningDataModule that yields online oracle-labelled VIN batches."""

    def __init__(self, config: VinDataModuleConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump())

        self._train_base: IterableDataset[EfmSnippetView] | None = None
        self._val_base: IterableDataset[EfmSnippetView] | None = None
        self._labeler: OracleRriLabeler | None = None

    # --------------------------------------------------------------------- setup
    def setup(self, stage: Stage | str | None = None) -> None:
        requested = Stage.from_str(stage) if stage is not None else None
        if self._labeler is None:
            self._labeler = self.config.labeler.setup_target()

        if requested is None or requested is Stage.TRAIN:
            if self._train_base is None:
                self._train_base = self.config.train_dataset.setup_target()

        if requested is None or requested in (Stage.VAL, Stage.TEST):
            if self._val_base is None:
                self._val_base = self.config.val_dataset.setup_target()

    # ------------------------------------------------------------------ loaders
    def train_dataloader(self) -> DataLoader:
        self.setup(stage=Stage.TRAIN)
        assert self._train_base is not None
        ds = VinOracleIterableDataset(
            base=self._train_base,
            labeler=self._require_labeler(),
            max_attempts_per_batch=self.config.max_attempts_per_batch,
            verbosity=self.config.verbosity,
        )
        return DataLoader(
            ds,
            batch_size=None,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        self.setup(stage=Stage.VAL)
        assert self._val_base is not None
        ds = VinOracleIterableDataset(
            base=self._val_base,
            labeler=self._require_labeler(),
            max_attempts_per_batch=self.config.max_attempts_per_batch,
            verbosity=self.config.verbosity,
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
        base = self._train_base if stage is Stage.TRAIN else self._val_base
        if base is None:
            raise RuntimeError(f"Missing base dataset for stage={stage}. Did setup() run?")
        ds = VinOracleIterableDataset(
            base=base,
            labeler=self._require_labeler(),
            max_attempts_per_batch=self.config.max_attempts_per_batch,
            verbosity=self.config.verbosity,
        )
        return iter(ds)

    def _require_labeler(self) -> OracleRriLabeler:
        if self._labeler is None:
            raise RuntimeError("Oracle labeler not initialized. Call setup() first.")
        return self._labeler


__all__ = ["VinDataModule", "VinDataModuleConfig", "VinOracleBatch"]
