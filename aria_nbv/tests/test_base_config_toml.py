"""Regression tests for :class:`oracle_rri.utils.base_config.BaseConfig`."""

import tomllib

import torch

from oracle_rri.utils import BaseConfig


class _DeviceConfig(BaseConfig[object]):
    """Minimal config holding a torch device."""

    device: torch.device = torch.device("cpu")
    """Torch device used by the component."""


def test_to_toml_converts_torch_device_to_str() -> None:
    cfg = _DeviceConfig(device=torch.device("cpu"))
    rendered = cfg.to_toml(include_comments=False, include_type_hints=False)

    parsed = tomllib.loads(rendered)
    assert parsed["device"] == "cpu"
