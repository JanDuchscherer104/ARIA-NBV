# Aria-VIN-NBV Paper Progress

## Scope
- Drafted a 17-page Typst paper describing the current Aria-VIN-NBV system.
- Added a local IEEE template override to remove missing-font warnings.
- Expanded the architecture section with rigorous, implementation-accurate details.
- Added a WandB run analysis section covering Jan 3, 2026 runs (>500 steps).
- Removed duplicate Fig./Table prefixes in cross-references and clarified the scene-field projection description.
- Replaced placeholder citations in `docs/contents/impl/aria_nbv_package.qmd` with Wikipedia links and updated `docs/references.bib`.
- Compiled and inspected all pages via `pdftotext` previews.

## Key Findings / Notes
- Local template override: `docs/typst/paper/charged_ieee_local.typ` uses DejaVu Serif/Mono to avoid missing TeX Gyre font warnings.
- `pdfinfo` still reports a benign "Suspects object is wrong type (boolean)" warning, but the PDF renders and page counts are correct.
- Cross-references now render cleanly (no "Fig. Fig." / "Table Table" duplicates).
- Integration test `test_vin_v2_real_data.py` passed; saw torch AMP deprecation warnings from PointNeXt layers.
- WandB run analysis (Jan 3, 2026):
  - `nn0jcqwr` logged NaNs immediately; `global_step=0` despite 1341 log steps.
  - `m06auwmr` reached `global_step=547` but train/val losses were NaN; val spearman ~0.004.
  - `jejo31ut` reached `global_step=1506` with finite losses; val spearman ~0.127, top3 ~0.222.

## Suggestions / Next Steps
- Decide whether to keep the paper at 17 pages or compress back to 15 pages.
- Investigate NaN sources in early runs (nn0jcqwr, m06auwmr) with targeted debug logging.
- Replace planned ablations with quantitative results once additional runs complete.
- Tighten table layouts (WandB + ablations) to avoid excessive line breaks; consider narrower columns or splitting tables.
- Revisit float placement/order in the architecture section so subsection lettering (A/B/C) reads in sequence.

## Commands Used
- `typst compile --root docs docs/typst/paper/main.typ docs/typst/paper/main.pdf`
- `pdfinfo docs/typst/paper/main.pdf`
- `pdftotext -f N -l N docs/typst/paper/main.pdf -`
- `quarto render docs/contents/impl/aria_nbv_package.qmd --to html`
- `quarto check`
- `oracle_rri/.venv/bin/python -m pytest oracle_rri/tests/integration/test_vin_v2_real_data.py`
