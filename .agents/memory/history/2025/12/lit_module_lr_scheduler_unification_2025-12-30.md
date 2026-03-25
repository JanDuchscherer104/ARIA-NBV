---
id: 2025-12-30_lit_module_lr_scheduler_unification_2025-12-30
date: 2025-12-30
title: "Lit Module Lr Scheduler Unification 2025 12 30"
status: legacy-imported
topics: [lit, module, lr, scheduler, unification]
source_legacy_path: ".codex/lit_module_lr_scheduler_unification_2025-12-30.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Lit module lr scheduler unification

## Summary
- Unified VIN lightning scheduler configuration into a single `lr_scheduler` field (union of OneCycle/Plateau or None).
- Moved OneCycle total-steps + max-LR resolution and Lightning scheduler dict construction into scheduler configs.
- Removed explicit float/int casting for pydantic-typed scheduler/optimizer fields.

## Tests
- `./oracle_rri/.venv/bin/pytest oracle_rri/oracle_rri/lightning/lit_module.py`
- `./oracle_rri/.venv/bin/pytest tests/lightning/test_interrupt_checkpoint.py`
- `./oracle_rri/.venv/bin/pytest tests/vin/test_vin_diagnostics.py -m integration`

## Potential issues
- This is a breaking config change: `scheduler`, `scheduler_plateau`, and `scheduler_onecycle` fields were removed. Existing TOML/CLI configs must be updated to use `lr_scheduler`.
- `OneCycleSchedulerConfig` now requires `trainer` or `total_steps`; calling `configure_optimizers()` without a trainer will raise a clear error.
- Max LR fallback uses the first optimizer param group; multi-group schedulers may want explicit per-group max LR support.

## Suggestions
- Consider a backward-compat model validator to map legacy scheduler fields into the new `lr_scheduler` union.
- If multi-group optimizers become common, allow `max_lr` lists in config or derive from each param group.
