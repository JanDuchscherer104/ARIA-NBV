# Rendering Ownership

`aria_nbv.rendering` owns candidate depth rendering, EFM/PyTorch3D renderer
integration, unprojection, candidate point-cloud construction, and rendering
diagnostics.

Rendering contracts should keep shapes, frames, units, and invalid-value
semantics explicit because downstream RRI and VIN code depends on them.
