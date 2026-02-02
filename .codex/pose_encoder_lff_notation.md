# Pose encoder diagram: LFF notation cleanup

## Task

Improve the symbolic representation of the **R6D + Learnable Fourier Features (LFF)** pose encoding in
`docs/diagrams/vin_nbv/mermaid/pose_encoder.mmd`, specifically replacing the code-like
`nn.Sequential([...])` depiction of the MLP.

## What changed

- Updated the LFF block label to a **math-first** description that matches the implementation in
  `oracle_rri/oracle_rri/vin/pose_encoding.py::LearnableFourierFeatures`:
  - Fourier features: `z = [cos(W_r x), sin(W_r x)] / sqrt(F_ff)`
  - 2-layer MLP with GELU: `Linear(F_ff→F_pe) → GELU → Linear(F_pe→F_q)` (equiv. `y = W2 * GELU(W1 z + b1) + b2`)
- Made the flatten/reshape edges explicit about `x` and `y` tensor shapes.
- Added `F_ff` to the diagram symbol list.

## Notes / alignment with code

- In code, the LFF pipeline is:
  1) `xwr = x @ Wr.T`
  2) `fourier = cat([cos(xwr), sin(xwr)]) / sqrt(fourier_dim)`
  3) `enc = MLP(fourier)` where `MLP = Linear(fourier_dim→hidden_dim) → GELU → Linear(hidden_dim→output_dim)`
- `F_ff` corresponds to `fourier_dim` and `F_pe` to `hidden_dim`.
- `F_q` in the diagram refers to `LearnableFourierFeatures.out_dim` (which becomes `output_dim` when
  `include_input=False`, the current default in `LearnableFourierFeaturesConfig`).

## Validation

- Rendered the updated Mermaid diagram successfully with:
  - `npx -y @mermaid-js/mermaid-cli -i docs/diagrams/vin_nbv/mermaid/pose_encoder.mmd -o /tmp/pose_encoder.svg`
