## Pose generation overhaul (Nov 28, 2025)

- Rebuilt `pose_generation` pipeline: sampling now uses modular direction samplers (uniform/power-spherical), world-frame filtering with `delta_azimuth_deg`, and roll-free orientation via new `view_axes_from_poses` helper.
- Introduced dataclass-based `CandidateContext` / `CandidateSamplingResult`, optional per-rule masks + debug extras, and PowerSpherical wrappers in `reference_power_spherical_distributions.py`.
- Refactored pruning rules to operate on the new context; MinDistance rule can emit distances when diagnostics are enabled.
- Updated downstream consumers (`candidate_depth_renderer`, dashboard panels, tests) to accept the dataclass outputs; kept `azimuth_full_circle` for UI compatibility but sampling relies on `delta_azimuth_deg`.
- Tests: `tests/pose_generation` skipped because `efm3d` import is missing in the environment; `tests/rendering/test_pytorch3d_renderer` fails to import `efm3d` for the same reason (no module on PYTHONPATH). No functional regressions observed in available checks.

Follow-ups:
- Consider adding an env hook or install step to expose `external/efm3d` for tests, or add skip guards similar to other suites.
- UI now exposes `delta_azimuth_deg` directly; `azimuth_full_circle` is inferred when delta=360°.
