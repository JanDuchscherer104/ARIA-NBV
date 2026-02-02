from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from oracle_rri.app.panels.offline_cache_utils import _estimate_nbytes, _p3d_cameras_nbytes


def test_estimate_nbytes_tensor() -> None:
    tensor = torch.zeros((10,), dtype=torch.float32)
    assert _estimate_nbytes(tensor) == 10 * 4


def test_estimate_nbytes_numpy() -> None:
    arr = np.zeros((3, 4), dtype=np.float64)
    assert _estimate_nbytes(arr) == 3 * 4 * 8


def test_estimate_nbytes_tensor_method() -> None:
    class _HasTensor:
        def __init__(self) -> None:
            self._t = torch.ones((5,), dtype=torch.int16)

        def tensor(self) -> torch.Tensor:
            return self._t

    obj = _HasTensor()
    assert _estimate_nbytes(obj) == 5 * 2


def test_estimate_nbytes_nested_containers() -> None:
    class _HasTensor:
        def __init__(self, value: torch.Tensor) -> None:
            self._t = value

        def tensor(self) -> torch.Tensor:
            return self._t

    @dataclass(slots=True)
    class _Bundle:
        a: torch.Tensor
        b: list[torch.Tensor]
        c: dict[str, torch.Tensor]
        d: _HasTensor
        e: int

    a = torch.zeros((2, 3), dtype=torch.float32)  # 24 bytes
    b0 = torch.zeros((4,), dtype=torch.int64)  # 32 bytes
    c0 = torch.zeros((1,), dtype=torch.uint8)  # 1 byte
    d0 = _HasTensor(torch.zeros((7,), dtype=torch.float16))  # 14 bytes
    bundle = _Bundle(a=a, b=[b0], c={"x": c0}, d=d0, e=1)

    assert _estimate_nbytes(bundle) == 24 + 32 + 1 + 14


def test_estimate_nbytes_cycle_safe() -> None:
    items: list[object] = []
    items.append(items)
    assert _estimate_nbytes(items) == 0


def test_p3d_cameras_nbytes() -> None:
    class _DummyCameras:
        def __init__(self) -> None:
            self.R = torch.zeros((2, 3, 3), dtype=torch.float32)
            self.T = torch.zeros((2, 3), dtype=torch.float32)
            self.focal_length = torch.zeros((2, 2), dtype=torch.float32)
            self.principal_point = torch.zeros((2, 2), dtype=torch.float32)
            self.image_size = torch.zeros((2, 2), dtype=torch.int64)

    cams = _DummyCameras()
    expected = (
        cams.R.numel() * cams.R.element_size()
        + cams.T.numel() * cams.T.element_size()
        + cams.focal_length.numel() * cams.focal_length.element_size()
        + cams.principal_point.numel() * cams.principal_point.element_size()
        + cams.image_size.numel() * cams.image_size.element_size()
    )
    assert _p3d_cameras_nbytes(cams) == expected
