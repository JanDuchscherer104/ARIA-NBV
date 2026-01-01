# VIN slides: torchsummary + input-feature plots

## Goal

Add **evidence-backed** VIN documentation to the slide deck `docs/typst/slides/slides_3.typ`:

- show **real input modalities** from ASE/EFM snippets,
- document **exact tensor shapes** seen in practice (EFM inputs → EVL neck features → VIN concat),
- include a **torchsummary**-style layer/shape/parameter summary of the **trainable VIN modules**,
- add **plots/diagnostics** for candidate pose descriptors and EVL neck feature volumes,
- keep all math conventions: **vectors/matrices/tensors are written as `bold(...)`**,
- ensure the slide deck compiles.

## What was added

### 1) Scripts (real-data, reproducible)

- `oracle_rri/scripts/summarize_vin.py`
  - Prints shapes for key EFM snippet tensors, EVL feature-contract tensors, and VIN feature concat.
  - Uses `torchsummary` to summarize the trainable modules:
    - `ShellShPoseEncoder` (SH for `bold(u), bold(f)` + 1D Fourier features for radius)
    - `VinScorerHead` (MLP + CORAL head)

- `oracle_rri/scripts/plot_vin_inputs.py`
  - Saves plots to `docs/figures/impl/vin/`:
    - `vin_pose_descriptor.png` (histograms + az/el scatter for `bold(u)` and `bold(f)`)
    - `vin_evl_features.png` (occ/obb feature magnitude mid-slice + histogram)
  - Note: the slides no longer include raw RGB/SLAM example images; they focus on conceptual + feature plots.

- `oracle_rri/scripts/plot_vin_encodings.py`
  - Produces conceptual/encoding visualizations:
    - `vin_shell_descriptor_concept.png` (3D sketch for `bold(t), r, bold(u), bold(f)`)
    - `vin_sh_components.png` (SH basis components over az/el)
    - `vin_radius_fourier_features.png` (linear `r` vs `log(r+epsilon)` Fourier feature curves)

- `oracle_rri/scripts/plot_vin_binning.py`
  - Runs `OracleRriLabeler` for a few snippets and plots:
    - raw RRI distribution,
    - clipped z distribution + quantile edges,
    - resulting label histogram (`K` bins).

### 2) Slide deck integration

- `docs/typst/slides/slides_3.typ`
  - Added VIN slides that show:
    - descriptor interpretation (`bold(t), r, bold(u), bold(f), s`),
    - encoding visualizations (SH + radius Fourier),
    - EVL feature diagnostics,
    - binning visualization (`vin_rri_binning.png`),
    - `torchsummary` split across two slides for readability.

### 3) Small model hygiene tweak

- `oracle_rri/oracle_rri/vin/model.py`
  - Freezes the **inactive** pose encoder (LFF vs SH) based on `pose_encoding_mode`, so trainable-parameter
    reporting and optimizers don’t carry unused parameters.

## How to reproduce

### Generate plots

```bash
cd oracle_rri
uv run python scripts/plot_vin_inputs.py \
  --scene-id 81283 --atek-variant efm --device auto \
  --num-samples 256 --mesh-simplify-ratio 0.02 \
  --out-dir ../docs/figures/impl/vin
```

```bash
cd oracle_rri
uv run python scripts/plot_vin_encodings.py \
  --scene-id 81283 --atek-variant efm --device cpu \
  --num-samples 256 --mesh-simplify-ratio 0.02 \
  --out-dir ../docs/figures/impl/vin
```

```bash
cd oracle_rri
uv run python scripts/plot_vin_binning.py \
  --scene-id 81283 --atek-variant efm --device auto \
  --num-snippets 4 --max-candidates 16 --backprojection-stride 16 \
  --out-dir ../docs/figures/impl/vin
```

### Print shapes + torchsummary

```bash
cd oracle_rri
uv run python scripts/summarize_vin.py \
  --scene-id 81283 --atek-variant efm --device auto --num-candidates 4
```

### Compile slides

```bash
typst compile --root docs \
  docs/typst/slides/slides_3.typ \
  docs/typst/slides/slides_3.pdf
```

## Notes / Open questions

- The candidate descriptor plots currently reflect the **candidate orientation policy** used by `CandidateViewGenerator` (e.g. “look-at reference” vs “look-outward”). If we change orientation rules, the `s = <bold(f), -bold(u)>` distribution will change accordingly.
- EVL feature magnitude plots are **diagnostics** only; for model features we still want to discuss:
  - channel compression (1×1×1 conv) before pooling/sampling,
  - frustum/ray sampling vs center sampling for candidate-conditioned queries.
