# Seminar: Next Best View Estimation (NBV)

See the [GitHub Pages](https://janduchscherer104.github.io/NBV/) for more information.

## Global GPU / Performance Mode

The oracle stack now understands a global performance flag so you can switch
all configurable devices/backends to CUDA with a single knob.

- **Environment variable**: set `NBV_PERFORMANCE_MODE=gpu` to force CUDA where
  available. Use `cpu` to pin everything to CPU, leave unset/`auto` for the
  previous behaviour.
- **Programmatic**: `from oracle_rri.utils import PerformanceMode, set_performance_mode`
  then call `set_performance_mode(PerformanceMode.GPU)` before building configs.

When GPU mode is active the library will

- resolve `select_device("auto")` to CUDA wherever possible,
- upgrade `CandidateDepthRendererConfig` to the PyTorch3D backend and run it on
  CUDA,
- propagate the CUDA choice into candidate sampling and point-cloud rendering.

If CUDA is requested but unavailable the components fall back to CPU and log a
warning; no behaviour changes occur when the flag is left at `auto`.
