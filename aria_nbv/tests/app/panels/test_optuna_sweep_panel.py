"""Tests for Optuna sweep panel helpers."""

# ruff: noqa: S101, D103, SLF001, PLR2004

from pathlib import Path

import numpy as np
import pandas as pd
from oracle_rri.app.panels import optuna_sweep


def test_normalize_param_value_handles_paths_and_arrays() -> None:
    assert optuna_sweep._normalize_param_value(Path("foo")) == "foo"
    assert optuna_sweep._normalize_param_value(np.int64(3)) == 3
    assert optuna_sweep._normalize_param_value([1, 2, 3]) == "[1, 2, 3]"


def test_infer_param_kind_detects_numeric() -> None:
    series = pd.Series(list(range(20)))
    assert optuna_sweep._infer_param_kind(series) == "numeric"


def test_infer_param_kind_detects_categorical() -> None:
    series = pd.Series(["a", "b", "a", "c"])
    assert optuna_sweep._infer_param_kind(series) == "categorical"


def test_select_param_columns_filters_prefix() -> None:
    df = pd.DataFrame(
        {
            "trial": [1, 2],
            "param.foo": [True, False],
            "param.bar": [0.1, 0.2],
        },
    )
    cols = optuna_sweep._select_param_columns(df)
    assert cols == ["param.bar", "param.foo"]


def test_bin_numeric_series_handles_degenerate() -> None:
    series = pd.Series([1.0, 1.0, 1.0])
    binned = optuna_sweep._bin_numeric_series(series, bins=4)
    assert len(binned.unique()) == 1


def test_interaction_matrix_pivot() -> None:
    df = pd.DataFrame(
        {
            "value": [0.5, 0.6, 0.7, 0.9],
            "param.a": [1, 1, 2, 2],
            "param.b": ["x", "y", "x", "y"],
        },
    )
    pivot = optuna_sweep._interaction_matrix(
        df,
        param_x="param.a",
        param_y="param.b",
        bins=3,
        agg=np.nanmean,
    )
    assert not pivot.empty
