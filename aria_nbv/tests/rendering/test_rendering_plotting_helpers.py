import torch

from oracle_rri.rendering.plotting import depth_histogram


def test_depth_histogram_returns_traces():
    depths = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, 0.5], [0.5, 0.5]],
        ]
    )
    fig = depth_histogram(depths, bins=5, zfar=5.0)
    assert len(fig.data) == depths.shape[0]
