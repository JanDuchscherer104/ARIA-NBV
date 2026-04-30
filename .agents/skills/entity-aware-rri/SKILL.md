---
name: entity-aware-rri
description: Use when ARIA-NBV work touches target/entity selection, GT OBB crops, target-specific RRI labels, target-conditioned VIN fields, or entity-aware diagnostics.
---

# Entity-Aware RRI

## When To Use

Use this skill for:

- target/entity eligibility and selection policies
- GT OBB cropped RRI, target mesh/point support, and invalid target reasons
- target-conditioned VIN batch fields, labels, and diagnostics
- scene-level versus target-level RRI comparisons

Do not use it for generic scene-level RRI unless an entity/target contract is
involved.

## Read First

1. `docs/contents/roadmap.qmd` sections M3 and M4
2. `docs/contents/questions.qmd` sections RQ1, RQ2, and RQ4
3. `aria_nbv/aria_nbv/rri_metrics/AGENTS.md`
4. `aria_nbv/aria_nbv/data_handling/AGENTS.md`
5. `.agents/memory/state/GOTCHAS.md`

## Rules

- Start with GT OBBs and a small trusted subset before predicted-OBB realism
  ablations.
- Keep unsupported targets explicit with invalid reasons; do not encode them as
  low-RRI valid samples.
- Log scene and target RRI separately.
- Preserve `PoseTW` / `CameraTW` boundaries and document crop frame, margin,
  units, and empty-crop semantics.

## Verification

- `cd aria_nbv && uv run pytest tests/rri_metrics`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py`
- `cd docs && quarto render contents/roadmap.qmd contents/questions.qmd` when target-aware claims change
- `make check-agent-memory` for memory or skill edits
