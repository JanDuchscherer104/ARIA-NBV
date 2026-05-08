# Thesis Prose Fixture

Use this as a mental regression test when polishing ARIA-NBV writing.

## Bad

This work leverages semantic relevance to unlock a holistic and powerful
next-best-view pipeline for AR reconstruction. The proposed method highlights
the crucial role of entity awareness in the rapidly evolving landscape of
interactive 3D reconstruction.

## Better

ARIA-NBV scores candidate views by expected reconstruction improvement for the
selected entity, not only by scene-level surface coverage. This changes the NBV
objective from "what improves the global model most?" to "what improves the
user-relevant target under the current interaction context?"

## Why Better

- It names the mechanism: candidate scoring by entity-specific expected
  improvement.
- It states the contrast: entity-level versus scene-level objective.
- It avoids unsupported importance claims.
- It gives the reader a testable interpretation of the method.
