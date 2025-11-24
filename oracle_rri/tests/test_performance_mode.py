import torch

from oracle_rri.pose_generation import CandidateViewGeneratorConfig
from oracle_rri.rendering import Efm3dDepthRendererConfig, Pytorch3DDepthRendererConfig
from oracle_rri.rendering.candidate_depth_renderer import CandidateDepthRendererConfig
from oracle_rri.utils import performance as perf


def test_pick_fast_renderer_upgrades_in_gpu_mode(monkeypatch):
    prev_mode = perf.get_performance_mode()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    perf.set_performance_mode(perf.PerformanceMode.GPU)

    cfg_cpu = Efm3dDepthRendererConfig(device="cpu", zfar=15.0)
    upgraded = perf.pick_fast_depth_renderer(cfg_cpu)

    assert isinstance(upgraded, Pytorch3DDepthRendererConfig)
    assert upgraded.zfar == cfg_cpu.zfar
    assert str(upgraded.device) == "cuda"

    perf.set_performance_mode(prev_mode)


def test_pick_fast_renderer_respects_cpu_mode(monkeypatch):
    prev_mode = perf.get_performance_mode()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    perf.set_performance_mode(perf.PerformanceMode.CPU)

    cfg_cpu = Efm3dDepthRendererConfig(device="cpu")
    upgraded = perf.pick_fast_depth_renderer(cfg_cpu)

    assert isinstance(upgraded, Efm3dDepthRendererConfig)
    assert str(upgraded.device) == "cpu"

    perf.set_performance_mode(prev_mode)


def test_candidate_depth_renderer_config_swapped_in_gpu_mode(monkeypatch):
    prev_mode = perf.get_performance_mode()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    perf.set_performance_mode(perf.PerformanceMode.GPU)

    cfg = CandidateDepthRendererConfig(renderer=Efm3dDepthRendererConfig(device="cpu"))
    assert isinstance(cfg.renderer, Pytorch3DDepthRendererConfig)
    assert str(cfg.renderer.device) == "cuda"

    perf.set_performance_mode(prev_mode)


def test_candidate_generator_device_follows_global_mode(monkeypatch):
    prev_mode = perf.get_performance_mode()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    perf.set_performance_mode(perf.PerformanceMode.GPU)
    cfg_gpu = CandidateViewGeneratorConfig()
    assert cfg_gpu.device.type == "cuda"

    perf.set_performance_mode(perf.PerformanceMode.CPU)
    cfg_cpu = CandidateViewGeneratorConfig()
    assert cfg_cpu.device.type == "cpu"

    perf.set_performance_mode(prev_mode)
