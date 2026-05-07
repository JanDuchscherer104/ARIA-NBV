"""Tests for deterministic Rerun inspector candidate color mapping."""

from __future__ import annotations

import numpy as np
import torch

from aria_nbv.rerun_inspector import (  # noqa: E402
    INVALID_RGBA,
    TARGET_OBB_RGBA,
    UNKNOWN_RGBA,
    VALID_RGBA,
    candidate_rgba,
    obb_semantic_rgba,
    oracle_rri_to_rgba,
    rank_to_rgba,
    step_to_rgba,
    validity_to_rgba,
)


def test_validity_to_rgba_is_uint8_and_exact() -> None:
    colors = validity_to_rgba([True, False, True])

    assert colors.dtype == np.uint8
    assert colors.shape == (3, 4)
    np.testing.assert_array_equal(colors[0], VALID_RGBA)
    np.testing.assert_array_equal(colors[1], INVALID_RGBA)
    np.testing.assert_array_equal(colors[2], VALID_RGBA)


def test_rank_to_rgba_is_stable_and_respects_validity() -> None:
    colors = rank_to_rgba(torch.tensor([0, 1, 8]), validity=[True, False, True])
    repeated = rank_to_rgba([0, 1, 8], validity=np.array([True, False, True]))

    assert colors.dtype == np.uint8
    np.testing.assert_array_equal(colors, repeated)
    np.testing.assert_array_equal(colors[0], np.array([37, 99, 235, 255], dtype=np.uint8))
    np.testing.assert_array_equal(colors[1], INVALID_RGBA)
    np.testing.assert_array_equal(colors[2], np.array([37, 99, 235, 255], dtype=np.uint8))


def test_oracle_rri_to_rgba_is_deterministic_with_unknown_and_invalid() -> None:
    values = np.array([0.0, 0.5, 1.0, np.nan], dtype=np.float32)
    validity = [True, True, False, True]

    colors = oracle_rri_to_rgba(values, validity=validity, vmin=0.0, vmax=1.0)
    repeated = oracle_rri_to_rgba(torch.from_numpy(values), validity=validity, vmin=0.0, vmax=1.0)

    assert colors.dtype == np.uint8
    assert colors.shape == (4, 4)
    np.testing.assert_array_equal(colors, repeated)
    np.testing.assert_array_equal(colors[0], np.array([68, 1, 84, 255], dtype=np.uint8))
    np.testing.assert_array_equal(colors[1], np.array([33, 145, 140, 255], dtype=np.uint8))
    np.testing.assert_array_equal(colors[2], INVALID_RGBA)
    np.testing.assert_array_equal(colors[3], UNKNOWN_RGBA)


def test_candidate_rgba_dispatches_modes() -> None:
    np.testing.assert_array_equal(candidate_rgba("validity", validity=[True, False]), validity_to_rgba([True, False]))
    np.testing.assert_array_equal(candidate_rgba("rank", ranks=[0]), rank_to_rgba([0]))
    np.testing.assert_array_equal(candidate_rgba("oracle_rri", oracle_rri=[1.0]), oracle_rri_to_rgba([1.0]))


def test_step_to_rgba_and_obb_semantic_palettes_are_distinct() -> None:
    step_colors = step_to_rgba([0, 1, 8], alpha=200)

    assert step_colors.dtype == np.uint8
    np.testing.assert_array_equal(step_colors[0], step_colors[2])
    assert step_colors[0, 3] == 200
    assert not np.array_equal(step_colors[0], step_colors[1])

    gt = obb_semantic_rgba([3, 4], family="gt", target_mask=[False, True])
    detected = obb_semantic_rgba([3, 4], family="detected")

    assert not np.array_equal(gt[0, :3], detected[0, :3])
    np.testing.assert_array_equal(gt[1], TARGET_OBB_RGBA)
