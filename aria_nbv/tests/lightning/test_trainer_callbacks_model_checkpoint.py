"""Tests for `TrainerCallbacksConfig` checkpoint wiring."""

# ruff: noqa: S101, SLF001

from __future__ import annotations

from pathlib import Path

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from aria_nbv.lightning.lit_trainer_callbacks import TrainerCallbacksConfig


def _get_model_checkpoint(callbacks: list[object]) -> ModelCheckpoint:
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
    raise AssertionError("Expected a ModelCheckpoint callback.")


def test_model_checkpoint_accepts_every_n_train_steps(tmp_path: Path) -> None:
    cfg = TrainerCallbacksConfig(
        use_model_checkpoint=True,
        checkpoint_dir=tmp_path,
        checkpoint_monitor="train/loss",
        checkpoint_every_n_train_steps=2,
        checkpoint_save_last=True,
        checkpoint_save_on_train_epoch_end=False,
    )

    callbacks = cfg.setup_target(model_name="vin", has_logger=False)
    checkpoint = _get_model_checkpoint(callbacks)

    assert checkpoint.monitor == "train/loss"
    assert checkpoint.save_last is True
    assert checkpoint._every_n_train_steps == 2
    assert checkpoint._save_on_train_epoch_end is False


def test_model_checkpoint_rejects_multiple_schedules(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainerCallbacksConfig(
            use_model_checkpoint=True,
            checkpoint_dir=tmp_path,
            checkpoint_every_n_train_steps=2,
            checkpoint_train_time_interval={"seconds": 3},
        )
