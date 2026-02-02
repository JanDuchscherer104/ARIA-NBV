from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Make vendored efm3d importable
sys.path.append(str(Path(__file__).resolve().parents[3] / "external" / "efm3d"))

from efm3d.aria import CameraTW, PoseTW

from oracle_rri.rendering.candidate_depth_renderer import CandidateDepthRenderer
from oracle_rri.utils import Console, Verbosity


class _FakeCameras:
    def __getitem__(self, _idx: object) -> "_FakeCameras":
        return self


def _make_renderer(*, max_candidates_final: int) -> CandidateDepthRenderer:
    renderer = CandidateDepthRenderer.__new__(CandidateDepthRenderer)
    renderer.config = SimpleNamespace(max_candidates_final=max_candidates_final)
    renderer.console = Console.with_prefix("test").set_verbosity(Verbosity.NORMAL).set_debug(False)
    return renderer


def test_filter_valid_candidates_logs_discarded_count_when_no_filtering() -> None:
    prev_verbosity = Console().verbosity
    prev_debug = Console().is_debug
    messages: list[str] = []

    Console.set_sink(messages.append)
    try:
        renderer = _make_renderer(max_candidates_final=4)
        num_total = 2
        depths = torch.zeros((num_total, 2, 2), dtype=torch.float32)
        depths_valid_mask = torch.ones((num_total, 2, 2), dtype=torch.bool)
        pose_batch = PoseTW(torch.zeros((num_total, 12), dtype=torch.float32))
        camera_calib = CameraTW(torch.zeros((num_total, 34), dtype=torch.float32))
        candidate_indices = torch.arange(num_total, dtype=torch.long)
        cameras = _FakeCameras()

        renderer._filter_valid_candidates(
            depths=depths,
            depths_valid_mask=depths_valid_mask,
            pose_batch=pose_batch,
            camera_calib=camera_calib,
            cameras=cameras,
            candidate_indices=candidate_indices,
        )

        assert any("Discarded 0 candidates (kept 2/2; invalid_zero_hit=0)." in line for line in messages)
    finally:
        Console.set_sink(None)
        Console().set_verbosity(prev_verbosity)
        Console().set_debug(prev_debug)


def test_filter_valid_candidates_logs_discarded_count_when_filtering() -> None:
    prev_verbosity = Console().verbosity
    prev_debug = Console().is_debug
    messages: list[str] = []

    Console.set_sink(messages.append)
    try:
        renderer = _make_renderer(max_candidates_final=2)
        num_total = 4
        depths = torch.zeros((num_total, 2, 2), dtype=torch.float32)
        depths_valid_mask = torch.tensor(
            [
                [[True, True], [True, True]],  # 4 hits
                [[True, True], [False, False]],  # 2 hits
                [[True, False], [False, False]],  # 1 hit
                [[True, True], [True, False]],  # 3 hits
            ],
            dtype=torch.bool,
        )
        pose_batch = PoseTW(torch.zeros((num_total, 12), dtype=torch.float32))
        camera_calib = CameraTW(torch.zeros((num_total, 34), dtype=torch.float32))
        candidate_indices = torch.arange(num_total, dtype=torch.long)
        cameras = _FakeCameras()

        renderer._filter_valid_candidates(
            depths=depths,
            depths_valid_mask=depths_valid_mask,
            pose_batch=pose_batch,
            camera_calib=camera_calib,
            cameras=cameras,
            candidate_indices=candidate_indices,
        )

        assert any(
            "Discarded 2 candidates (invalid_zero_hit=0, capped=2 due to max_candidates_final=2; kept 2/4)." in line
            for line in messages
        )
    finally:
        Console.set_sink(None)
        Console().set_verbosity(prev_verbosity)
        Console().set_debug(prev_debug)
