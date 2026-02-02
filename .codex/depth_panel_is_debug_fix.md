# Depth panel is_debug fix

Date: 2026-01-26

## Issue
- `Pytorch3DDepthRendererConfig` rejected `is_debug` from the Streamlit UI, raising a
  Pydantic extra-field error when opening the depth panel.

## Changes
- Added `is_debug: bool` to `Pytorch3DDepthRendererConfig` and wired it into the
  renderer console (`Console.set_debug`).
- Added a unit test to ensure the config accepts `is_debug` and propagates it to
  the renderer instance.

## Tests
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/rendering/test_pytorch3d_renderer.py::test_pytorch3d_config_accepts_debug`
