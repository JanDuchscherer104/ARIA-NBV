## Task
Assess `docs/contents/theory/candidate_generation/candidate_generation_revision.qmd` for math delimiter cleanup ($...$, $$...$$) and fix formatting.

## Findings (latest)
- `make context*` still fails because `Makefile` line 8 uses `YELL3[0;33m` without `:=` (missing separator), so context helpers remain broken.
- `docs/contents/theory/candidate_generation/candidate_generation_revision.qmd` now contains the GPT5-PRO revision text. Two math fixes applied:
  - Converted the spherical-coordinate block to a display equation using `$$ ... $$` with `aligned`.
  - Wrapped the expectation term `(\mathbb{E}[d] \approx c \cdot (0, 1, 0))` in inline math delimiters.

## Blockers / Next Steps
- If desired, patch the `Makefile` color variable to re-enable `make context` and `make context-dir-tree`.
- No further formatting blockers in `candidate_generation_revision.qmd` identified after math delimiter fixes.
