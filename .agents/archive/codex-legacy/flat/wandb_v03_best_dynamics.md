# W&B v03-best dynamics (rtjvfyyp)

Summary
- Run id: `rtjvfyyp` (name: `v03-best`), project `traenslenzor/aria-nbv`.
- History shows `epoch=25` and no early-stopping keys in config/summary; stop is consistent with `max_epochs=25`.
- Train/val `coral_loss_rel_random` plateau: early 0.742/0.725 → mid 0.673/0.676 → late 0.636/0.665 with tiny late slopes.
- `lr-AdamW` decays monotonically `8.0e-4` → `4.34e-4` (no warmup peak). Step-level loss correlates positively with LR (rho ≈ 0.43).
- Step-level loss CV ≈ 0.11 in the late segment (min ≈ 0.45, max ≈ 0.91), suggesting moderate noise.

Implications
- Dynamics suggest a plateau by ~epoch 20; further gains likely require LR schedule changes or increased signal-to-noise (larger batch / grad accumulation).
- Evidence supports revisiting one-cycle hyperparameters (peak timing and magnitude) rather than dropping the schedule outright.

Paper update
- Added a “Latest run dynamics (v03-best)” subsection to `docs/typst/paper/sections/12g-appendix-vin-v3-streamline.typ`.
