# Architecture section update (VINv3)

- Updated `docs/typst/paper/sections/06-architecture.typ` to reflect VINv3 implementation details.
- Added candidate-specific semidense projection stats, validity mask, reliability weighting, and grid-CNN encoding.
- Added voxel-projection FiLM modulation, optional trajectory context, and updated feature fusion formula.
- Marked the VIN v2 diagram as legacy reference.

Compile:
- `typst compile --root docs docs/typst/paper/main.typ`
- Refined math notation in the architecture section (projection, validity mask, reliability weights, grid CNN features).
- Added appendix tables defining all lit_module logged losses/metrics in `docs/typst/paper/sections/12b-appendix-extra.typ`.
