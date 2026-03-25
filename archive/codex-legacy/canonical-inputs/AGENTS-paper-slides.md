# AGENTS – Paper + Slides (Typst)

This file is a focused workflow guide for writing the **final paper** in `docs/typst/paper/` and the **slide decks** in `docs/typst/slides/`.
It complements `.codex/AGENTS.md` and captures Typst + template conventions for this repo.

---

## 1) Quickstart (Where to Work)

- **Paper**: `docs/typst/paper/main.typ` + `docs/typst/paper/sections/`
- **Slides**: `docs/typst/slides/slides_{1,2,3}.typ` (optional theme overrides in `docs/typst/slides/custom-template.typ`)
- **Shared notation/macros**: `docs/typst/shared/macros.typ` (prefer adding new symbols here, not ad-hoc per deck)

---

## 2) Render / Compile (Required Before Finalizing)

Typst files in this repo are typically compiled with `--root docs` so absolute project paths (e.g. `/references.bib`) resolve to `docs/...`.

**Bootstrapping (paper only)**
- `docs/typst/paper/main.typ` expects `docs/_shared/references.bib` (via `#let shared_path = "/_shared/"`).
  - Create once: `mkdir -p docs/_shared && cp -f docs/references.bib docs/_shared/references.bib`
  - Keep in sync when `docs/references.bib` changes.

**Write outputs into the repo**
- In some sandboxed environments Typst cannot write output files outside the repo (e.g. `/tmp/...`). Prefer `.codex/_render/...` for generated PDFs/PNGs.
- Create the output dir once: `mkdir -p .codex/_render`

**Compile commands (examples)**
- Slides (compile once): `typst compile --root docs docs/typst/slides/slides_2.typ .codex/_render/slides_2.pdf --diagnostic-format short`
- Slides (watch): `typst watch --root docs docs/typst/slides/slides_2.typ .codex/_render/slides_2.pdf`
- Paper (compile once): `typst compile --root docs docs/typst/paper/main.typ .codex/_render/paper.pdf --diagnostic-format short`

Notes:
- The slide decks include `#bibliography("/references.bib", ...)` → expects `docs/references.bib` when compiling with `--root docs`.
- `docs/typst/paper/main.typ` currently uses `#let shared_path = "/_shared/"` → expects `docs/_shared/references.bib` when compiling with `--root docs`. Create `docs/_shared/` (or adjust `shared_path`) before relying on paper compilation.
- If the IEEE template warns about missing fonts, check discovered fonts with `typst fonts` and add fonts via `--font-path` if needed.

---

## 3) Autonomous “Compile → Inspect → Critique → Fix” Loop

When working autonomously on a bigger task, **treat compilation + visual inspection as part of the inner loop** (not a final step).

**Loop (repeat until clean)**
1) Compile to PDF (no errors; warnings reviewed).
2) Render the relevant pages to PNG.
3) Inspect the PNGs (agent: use `view_image`; human: open the PDF).
4) Critique against the checklist below and fix issues.
5) Re-compile and re-inspect.

**Render pages to PNG (Typst-native)**
- Slides (pages 1–3): `typst compile --root docs -f png --ppi 200 --pages 1-3 docs/typst/slides/slides_2.typ .codex/_render/slides_2-{0p}.png`
- Paper (pages 1–2): `typst compile --root docs -f png --ppi 200 --pages 1-2 docs/typst/paper/main.typ .codex/_render/paper-{0p}.png`

**Fallback: render from PDF (if you already compiled)**
- `pdftoppm -f 1 -l 2 -png .codex/_render/paper.pdf .codex/_render/paper_ppm`
- `mutool draw -o .codex/_render/paper_mutool-%d.png -F png -r 200 .codex/_render/paper.pdf 1-2`

**Critique checklist (fast)**
- Text: no overflow/cutoff, consistent font sizes, readable line lengths.
- Layout: alignments clean, grids balanced, no “floating” orphan elements.
- Figures: correct scale/cropping, legible labels, consistent caption style, no raster artifacts.
- Equations: rendered correctly, consistent notation, referenced in the surrounding text.
- References: citations resolved (no raw keys), cross-refs point to the right objects.

---

## 4) Templates + Packages (Keep Consistent)

**Paper template: `@preview/charged-ieee:0.1.4`**

- Configured via `#show: ieee.with(...)` in `docs/typst/paper/main.typ`
- Keep paper content in `docs/typst/paper/sections/` and `#include` from `main.typ`

**Slides template: `@preview/definitely-not-isec-slides:1.0.1`**

- Used via `#show: definitely-not-isec-theme.with(...)`
- Use `#title-slide()`, `#section-slide(...)`, and `#slide(title: [...])[ ... ]`
- Optional: `docs/typst/slides/custom-template.typ` overrides header/footer and `color-block` styling (rounded blocks, logo-only header)

**PDF embedding: `@preview/muchpdf:0.1.1`**
- Used in slides as `muchpdf(read(path, encoding: none))` for paper figures/teasers stored as PDFs.
- Can also be used in the paper if you need to embed a specific PDF page as a figure.

---

## 5) Style + Writing (Paper + Slides)

- Audience: graduate-level ML/CV; prefer clear claims + evidence over long prose.
- Slides: short bullets, minimal paragraphs; use 2-column grids for density (`#grid(columns: (1fr, 1fr), ...)`) and `#color-block(...)` where appropriate.
- Notation:
  - Use `$bold(..)$` for vectors/matrices/tensors (math mode).
  - Prefer `docs/typst/shared/macros.typ` for acronyms/symbols (e.g. `#RRI`, `#EFM3D`, `#SLAM`) and keep notation consistent across paper + slides.
- Abstract hygiene (IEEE-style): 1 paragraph, self-contained (≈150–250 words), no refs/abbrevs, cover **topic → purpose → method → results → conclusion**.
- Claims discipline: each claim must be backed by a figure/table/metric; avoid causal claims without ablations.
- Integrity + AI tools: follow IEEE ethics; AI systems are not authors; disclose AI assistance if the venue requires it.

For deeper guidance + sources, see `.codex/academic-writing-guidelines.md`.

---

## 6) Assets, Cross-Refs, and Bibliography

- Store slide assets under `docs/figures/...` and keep a single `#let fig_path = ...` per deck.
- Wrap images/tables in `#figure(...)` with captions and labels, then reference with `@label`.
- Citations use the same `@key` syntax as cross-references (after adding a bibliography with `#bibliography(...)`).
- Bibliography source of truth: `docs/references.bib` (Quarto) and `/references.bib` (Typst when compiling with `--root docs`).
- Reused figures/equations: store locally, cite the primary source (`@key`), and respect licenses/permissions.

---

## 7) Typst Essentials (Minimal)

- `#import`, `#include`, `#let` for modularization and variables.
- `#set` and `#show` for styling and show rules.
- Labels: attach `<label>` to a referenceable element; reference with `@label`.
- Citations: `@bib_key` (or `#cite(...)` if needed) after `#bibliography(...)`.
- Layout: `#grid(...)`, `#stack(...)`, `#h(...)`, `#v(...)`.

When generating Typst code, consult Typst docs via Context7: `get-library-docs /websites/typst_app`.

---

## 8) Core Reference Papers (Local Sources + Bib Keys)

Use this table when adding figures/equations or aligning terminology. “Bib key” refers to entries in `docs/references.bib` (Typst citation syntax: `@key`).

| Topic | Bib key | LaTeX source | Literature review (QMD) |
| --- | --- | --- | --- |
| EFM3D / EVL | `EFM3D-straub2024` | `literature/tex-src/arXiv-EFM3D/main.tex` | `docs/contents/literature/efm3d.qmd` |
| VIN-NBV / RRI policy learning | `VIN-NBV-frahm2025` | `literature/tex-src/arXiv-VIN-NBV/main.tex` | `docs/contents/literature/vin_nbv.qmd` |
| GenNBV (continuous-action NBV) | `GenNBV-chen2024` | `literature/tex-src/arXiv-GenNBV/main.tex` | `docs/contents/literature/gen_nbv.qmd` |
| SceneScript (ASE + structured scene language) | `SceneScript-avetisyan2024` | `literature/tex-src/arXiv-scene-script/main.tex` | `docs/contents/literature/scene_script.qmd` |
| Project Aria (platform/tooling overview) | `projectaria-engel2023` | `literature/tex-src/arXiv-project-aria/main.tex` | `docs/contents/ase_dataset.qmd` |

---

## 9) When to Ask for Clarification

- If the target venue, required template, or section structure is unclear.
- If a figure or equation source is missing or ambiguous.
- If requested content conflicts with existing slide/paper conventions.
