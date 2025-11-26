import torch

from oracle_rri.rendering.plotting import depth_histogram, hit_ratio_bar


def test_depth_histogram_returns_traces():
    depths = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, 0.5], [0.5, 0.5]],
        ]
    )
    fig = depth_histogram(depths, bins=5, zfar=5.0)
    assert len(fig.data) == depths.shape[0]


def test_hit_ratio_bar_uses_zfar():
    depths = torch.tensor(
        [
            [[1.0, 2.0], [6.0, 7.0]],
            [[9.0, 9.0], [9.0, 9.0]],
        ]
    )
    fig = hit_ratio_bar(depths, zfar=5.0)
    bars = fig.data[0]
    assert len(bars.x) == depths.shape[0]
    # first candidate has 2/4 hits, second 0
    assert bars.y[0] == 0.5
    assert bars.y[1] == 0.0
