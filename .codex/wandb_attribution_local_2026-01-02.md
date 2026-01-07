# W&B attribution without doc_classifier (2026-01-02)

## Changes
- Added `oracle_rri/oracle_rri/interpretability/attribution.py` and package init to host the Captum attribution engine locally.
- Removed `oracle_rri/oracle_rri/app/panels/doc_classifier_utils.py` to eliminate imports from external/traenslenzor doc_classifier.
- Refactored `oracle_rri/oracle_rri/app/panels/wandb.py` attribution explorer to use the local attribution engine and a torchvision-based model path.

## Notes / Findings
- Attribution explorer now builds a torchvision classifier (alexnet/resnet50/vit_b_16) and optionally loads a state_dict; class label display is configurable via a comma-separated list.
- Captum remains an optional dependency; if missing, the panel surfaces a runtime error when attribution is requested.

## Tests
- `oracle_rri/.venv/bin/python -m pytest tests` fails during collection with
  `ImportError: cannot import name 'backproject_depth' from oracle_rri.rendering.unproject` in
  `tests/rendering/test_unproject.py` (pre-existing, unrelated).

## Follow-up (doc-classifier cleanup)
- Removed all remaining doc-classifier-related helpers and attribution explorer UI from `oracle_rri/oracle_rri/app/panels/wandb.py`.
- Attribution section now displays a disabled notice only.
