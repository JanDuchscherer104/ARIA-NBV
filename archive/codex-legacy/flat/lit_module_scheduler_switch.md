# LitModule Scheduler Switch

## Summary
- Added configurable LR scheduler selection with ReduceLROnPlateau/OneCycle/none.

## Changes
- `oracle_rri/oracle_rri/lightning/lit_module.py`: added scheduler config classes, wiring in `VinLightningModuleConfig`, and scheduler selection in `configure_optimizers`.

## Notes / Suggestions
- If you want per-parameter-group OneCycle LRs, extend `OneCycleSchedulerConfig` to accept a list and pass it through.
