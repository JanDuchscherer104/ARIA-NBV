---
id: 2026-01-02_attr_inference_tensor_fix_2026-01-02
date: 2026-01-02
title: "Attr Inference Tensor Fix 2026 01 02"
status: legacy-imported
topics: [attr, inference, tensor, 2026, 01]
source_legacy_path: ".codex/attr_inference_tensor_fix_2026-01-02.md"
confidence: low
---

> Imported from legacy Codex note during the 2026-03-24 scaffolding migration.

# Attribution inference-tensor fix

## Summary
- Switched VIN debug forward pass in Testing & Attribution panel from `torch.inference_mode()` to `torch.no_grad()`.
- Cloned/detached feature tensors (`feats`, `pose_vec`, `global_feat`) before passing them into Captum to avoid inference-tensor autograd errors.

## Files changed
- `oracle_rri/oracle_rri/app/panels/testing_attribution.py`

## Tests
- `ruff check oracle_rri/oracle_rri/app/panels/testing_attribution.py`

## Notes
- Captum requires normal tensors for backward; inference-mode outputs are incompatible.
