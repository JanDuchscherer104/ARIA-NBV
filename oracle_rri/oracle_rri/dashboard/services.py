"""Cached loaders and executors for Streamlit dashboard."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import streamlit as st

from ..data import AseEfmDatasetConfig


@st.cache_resource(
    show_spinner=False,
    hash_funcs={AseEfmDatasetConfig: lambda c: json.dumps(c.model_dump(mode="json", round_trip=True), sort_keys=True)},
)
def load_dataset(cfg: AseEfmDatasetConfig):
    return cfg.setup_target()


def load_sample(cfg: AseEfmDatasetConfig, sample_idx: int = 0):
    dataset = load_dataset(cfg)
    it = iter(dataset)
    for _ in range(sample_idx + 1):
        sample = next(it)
    return sample


@lru_cache(maxsize=1)
def get_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)
