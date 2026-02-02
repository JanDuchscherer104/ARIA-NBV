# Oracle RRI Typst Alignment (2026-01-30)

## Goal

Resolve the *user-authored* TODOs in `docs/typst/paper/sections/05-oracle-rri.typ` and align the narrative + notation with `docs/typst/slides/slides_4.typ` (esp. candidate generation + view jitter, depth rendering diagnostics, and backprojection).

Constraints:
- Leave agent-authored `TODO(paper-cleanup)` comments untouched.
- Prefer shared notation via `docs/typst/shared/macros.typ` (`#symb`, `#eqs`), and follow slide notation for candidate generation.

## What changed

### Candidate generation (paper section 05)

- Added a single “Key parameters (candidate generation)” table using slide symbols:
  `N_q, r_min/r_max, theta_min/theta_max, psi_span, psi_delta, theta_delta, phi_delta`,
  plus `align_to_gravity` and `min_dist_to_mesh`.
- Rewrote **position sampling** to match slide equations:
  - `bold(s)_q ~ U(S^2)`, `r_q ~ U(r_min, r_max)`
  - `(psi, theta) = angles(bold(s)_q)`, then linear map into caps, then `bold(s)'_q` and `c_q = T^w_r (r_q bold(s)'_q)`.
  - Moved `pos_ref.png` into this subsection.
- Reworked **view directions + jitter** to match slide narrative:
  - Clarified that `(psi, theta)` caps apply to the *jitter delta* (not the base view).
  - Jitter composition uses `bold(R)_("delta") = R_z(psi) R_y(theta) R_x(phi)` and a right-multiplicative pose update.
  - Moved `view_dirs_ref.png` into the view-directions subsection.

### Depth rendering diagnostics

- Split the stacked diagnostics figure into two standalone figures:
  - `cand_renders_1x3.png` and `depth_histograms_3x3.png`,
  each with its own caption/label and a short explanatory paragraph.
- Updated references in `docs/typst/paper/sections/09-diagnostics.typ` and
  `docs/typst/paper/sections/12-appendix-gallery.typ` to the new figure labels.

### Backprojection

- Updated the NDC mapping to match slides (including `@PyTorch3D-Cameras-2025`).
- Replaced the placeholder `Pi^(-1)(...)` notation with:
  `bold(p)_("world") = "unproject"(x_ndc, y_ndc, d_q)` (with `d_q` sampled from `D_q`).

### Shared macros

- Added a `#gh(path)` helper in `docs/typst/shared/macros.typ` that links to
  `https://github.com/JanDuchscherer104/NBV/blob/main/<path>` while displaying only the filename.

## Validation

- `typst compile --root docs docs/typst/paper/main.typ` succeeded.
- `typst compile --root docs docs/typst/slides/slides_4.typ` succeeded.

## Follow-up update (same day)

- Resolved remaining inline TODOs in `05-oracle-rri.typ`:
  - Clarified that $bold(s)_q$ is a unit direction in spherical coordinates (azimuth/elevation).
  - Switched remaining unqualified file-path references to `#gh(...)`.
  - Switched backprojection equation to $Pi^{-1}$ notation (instead of `"unproject"`).
  - Split the orientation-jitter grid into two figures with concise interpretation text.
- Ensured the paper compiles by importing `macros.typ` in sections that use `#code-inline`:
  - `docs/typst/paper/sections/07b-training-config.typ`
  - `docs/typst/paper/sections/11-conclusion.typ`

## Follow-ups / optional cleanup

- If desired, extend `#symb` with explicit scalars (`psi_delta`, `theta_delta`, …) for even stricter notation control (currently these are plain math symbols).
- Consider whether the detailed “Radial look-away” construction should remain in the main section or be moved to an appendix if space becomes tight.
