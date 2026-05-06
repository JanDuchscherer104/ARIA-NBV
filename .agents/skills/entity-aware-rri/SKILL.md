---
name: entity-aware-rri
description: Use when ARIA-NBV work touches target/entity selection, GT OBB crops, target-specific RRI labels, target-conditioned VIN fields, or entity-aware diagnostics.
metadata:
  applies_to:
    - "aria_nbv/aria_nbv/rri_metrics/**"
    - "aria_nbv/aria_nbv/data_handling/**"
    - "aria_nbv/aria_nbv/vin/**"
    - "docs/contents/thesis/**"
  triggers:
    - "target RRI"
    - "entity-aware"
    - "GT OBB"
    - "V0 target"
    - "V1 observed target"
    - "OBS-SEL"
    - "PRED-Q"
    - "GT-EVAL"
  must_read:
    - "docs/contents/thesis/roadmap.qmd#roadmap-m3"
    - "docs/contents/thesis/questions.qmd#rq2-target-encoding"
    - "aria_nbv/aria_nbv/rri_metrics/AGENTS.md"
    - "aria_nbv/aria_nbv/data_handling/AGENTS.md"
  verification:
    - "cd aria_nbv && uv run pytest tests/rri_metrics"
    - "cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py"
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

1. `docs/contents/thesis/roadmap.qmd` sections M3 and M4
2. `docs/contents/thesis/questions.qmd` sections RQ1, RQ2, and RQ4
3. `aria_nbv/aria_nbv/rri_metrics/AGENTS.md`
4. `aria_nbv/aria_nbv/data_handling/AGENTS.md`
5. `.agents/memory/state/GOTCHAS.md`

## Rules

- V0 sanity/upper-bound: actor input may use GT OBB; label and evaluation crop
  use GT OBB; claim only oracle upper-bound or target-RRI correctness.
- V1 main protocol: selector and scorer/Q_H input use observed or predicted OBB
  descriptors, confidence, geometry, projected area, and support signals; label
  and evaluation use matched GT OBB crops under OBS-SEL / PRED-Q / GT-EVAL.
- Do not claim main target-conditioned performance from V0-only GT-input runs.
- Start correctness on a small trusted subset before predicted-OBB realism
  ablations or scale-up.
- Keep unsupported targets explicit with invalid reasons; do not encode them as
  low-RRI valid samples.
- Log scene and target RRI separately.
- Preserve `PoseTW` / `CameraTW` boundaries and document crop frame, margin,
  units, and empty-crop semantics.
- Target-specific RRI for target `e` should follow the same improvement form as
  scene RRI over the target support:
  `RRI_e(q) = (d(P_t, M_e) - d(P_t union P_q, M_e)) / (d(P_t, M_e) + eps)`.
- Empty or unsupported target crops are invalid samples. They are not
  scene-level fallback labels and not lowest-bin valid targets.
- Acceptance checks should show crop support, target OBB, current points,
  candidate points, target RRI, scene RRI, and invalid reasons side by side.

## Verification

- `cd aria_nbv && uv run pytest tests/rri_metrics`
- `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py`
- `cd docs && quarto render contents/thesis/roadmap.qmd contents/thesis/questions.qmd` when target-aware claims change
- `make check-agent-memory` for memory or skill edits
