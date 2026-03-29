"""Tests for interrupt checkpoint handling."""

# ruff: noqa: S101, SLF001

from __future__ import annotations

from pathlib import Path

from oracle_rri.configs import PathConfig
from oracle_rri.lightning.aria_nbv_experiment import AriaNBVExperimentConfig


class DummyTrainer:
    """Minimal trainer stub for checkpoint saving tests."""

    def __init__(self, epoch: int = 1, step: int = 5) -> None:
        """Init dummy trainer with a known epoch/step."""
        self.current_epoch = epoch
        self.global_step = step
        self.is_global_zero = True

    def save_checkpoint(self, path: str) -> None:
        """Write a minimal checkpoint file."""
        Path(path).write_text("checkpoint", encoding="utf-8")


def test_interrupt_checkpoint_saved_in_pathconfig(tmp_path: Path) -> None:
    """Ensure interrupt checkpoints land in PathConfig.checkpoints."""
    paths = PathConfig()
    original_checkpoints = paths.checkpoints
    try:
        paths.checkpoints = tmp_path / "checkpoints"
        cfg = AriaNBVExperimentConfig(run_name="test-run")
        trainer = DummyTrainer(epoch=3, step=42)

        checkpoint_path = cfg._save_interrupt_checkpoint(trainer)

        assert checkpoint_path is not None
        assert checkpoint_path.exists()
        assert checkpoint_path.parent == paths.checkpoints
        assert checkpoint_path.suffix == ".ckpt"
    finally:
        paths.checkpoints = original_checkpoints
