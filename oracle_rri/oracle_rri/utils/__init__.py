"""Lightning-adjacent utilities for Document Classifier."""

from .base_config import BaseConfig, NoTarget, SingletonConfig
from .console import Console
from .performance import (
    ENV_VAR as PERFORMANCE_ENV_VAR,
)
from .performance import (
    PerformanceMode,
    get_performance_mode,
    pick_fast_depth_renderer,
    prefer_cpu,
    prefer_gpu,
    select_device,
    set_performance_mode,
)
from .rich_summary import build_nested, rich_summary
from .summary import summarize

__all__ = [
    "BaseConfig",
    "Console",
    "NoTarget",
    "SingletonConfig",
    "PerformanceMode",
    "PERFORMANCE_ENV_VAR",
    "get_performance_mode",
    "set_performance_mode",
    "prefer_gpu",
    "prefer_cpu",
    "select_device",
    "pick_fast_depth_renderer",
    "rich_summary",
    "build_nested",
    "summarize",
]
