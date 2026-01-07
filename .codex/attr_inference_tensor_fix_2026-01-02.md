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
