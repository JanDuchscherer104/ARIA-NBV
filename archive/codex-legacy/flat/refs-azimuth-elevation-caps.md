# References for azimuth/elevation cap rescaling

## Scope

This note documents sources used to justify the “azimuth/elevation caps without rejection” mapping used in:

- `oracle_rri/oracle_rri/pose_generation/positional_sampling.py` (`PositionSampler._scale_into_caps`)
- `docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ`

## What the code does (high level)

Given a raw unit direction in rig frame `(x, y, z)` (LUF convention: `x` left, `y` up, `z` forward), the sampler:

1. Computes angles
   - azimuth `psi = atan2(x, z)` (rotation around +Y, 0 aligned to +Z)
   - elevation `theta = atan2(y, sqrt(x^2 + z^2))` (angle above the XZ plane)
2. Caps azimuth by linearly scaling `psi` into a smaller band of width `Δ_az` and reconstructing the horizontal direction via `sin/cos`.
3. Caps elevation by mapping `y = sin(theta)` from `[-1, 1]` into `[sin(theta_min), sin(theta_max)]`, then rescales the horizontal components to preserve unit norm:
   `sqrt(1 - y'^2) / sqrt(x^2 + z^2)`.
4. Renormalizes for numerical stability.

## Sources and why they apply

### Papula (2024) — spherical coordinates + coordinate transforms / Jacobian

`literature/978-3-658-45806-5.pdf` (Papula, *Mathematische Formelsammlung*, 13th ed., 2024) contains:

- Spherical coordinate definitions and the Cartesian↔spherical relations (`x = r sin J cos j`, `y = r sin J sin j`, `z = r cos J`) and the inverse relations including the quadrant-aware angle recovery. (Sections “9.2.4 Kugelkoordinaten” and “9.2.5 Zusammenhang zwischen den kartesischen und den Kugelkoordinaten”.)
- The volume element / Jacobian in spherical coordinates (`dV = r^2 sin J dr dJ dj`), which underlies the common “sample azimuth uniformly and sample `cos(J)` (equivalently `sin(elevation)`) uniformly” construction for uniform surface-area sampling. (Section “3.2.4 Berechnung eines Dreifachintegrals in Kugelkoordinaten”.)

These are used as a compact, textbook reference for the trigonometric relations and the measure term that motivates using `sin(theta)` as the linear coordinate for elevation bands.

### Marsaglia (1972) — uniform sphere sampling parameterization

Marsaglia’s short note “Choosing a Point from the Surface of a Sphere” provides a classic parameterization of uniform sampling on the unit sphere using:

- a uniform azimuth angle, and
- a uniform auxiliary variable `u ∈ [-1, 1]` for the “vertical” coordinate,

with horizontal components scaled by `sqrt(1 - u^2)`.

This directly matches the structure of the cap mapping used here:
we map a component like `u` (in our coordinate choice: `y = sin(theta)`) into a restricted range and rescale the horizontal plane by `sqrt(1 - y'^2)` while keeping azimuth uniform.

## Patch summary

- Added citations in `docs/typst/paper/sections/12c-appendix-oracle-rri-labeler.typ`.
- Added BibTeX entries to `docs/references.bib`:
  - `Formelsammlung-papula2024`
  - `SpherePointPicking-marsaglia1972`

## Optional future references (not added)

If we want additional, more “directional statistics” oriented backing, consider adding a dedicated reference (e.g., a directional statistics textbook) explaining the surface-area element and uniformity in `(azimuth, cos(polar))` coordinates.

