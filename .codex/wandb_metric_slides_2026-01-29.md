# W&B metric figure slides (2026-01-29)

## Summary
- Added three slides pairing logged metrics/losses with W&B figures:
  - Train CORAL relative loss (epoch + step).
  - Validation CORAL relative loss + top-3 accuracy.
  - Aux regression loss + aux weight schedule.
- Compiled slides to `/tmp/slides_4.pdf`.

## Files touched
- `docs/typst/slides/slides_4.typ`

## Follow-ups
- Consider adding a single “trainer safeguards” slide covering skip/drop logic and binner setup if you want more details for the professor.
