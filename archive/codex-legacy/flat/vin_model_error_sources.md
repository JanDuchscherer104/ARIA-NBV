# Potential error sources in `vin/model.py` (training issues)

## Summary of likely contributors

- **Occupancy scale mismatch**: `occ_pr_is_logits=False` assumes EVL outputs probabilities; if the model emits logits, the scene field is distorted.
- **Frustum sampling window**: the grid is clamped to a square around the principal point (0.95 × min(H,W)), which may under-sample the true FOV on wide/rectangular images.
- **Fixed shallow depths**: default `frustum_depths_m=[0.5,1,2,3]` can miss relevant surfaces in large rooms, producing weak/invalid local tokens.
- **Voxel transform interpretation**: if `voxel/T_world_voxel` is not truly world←voxel, the world→voxel mapping is wrong, leading to low token validity (consistent with ~0.7 valid fraction).
- **Global mean pooling only**: the only global summarization is a voxel-wide mean; for sparse scenes this becomes nearly constant, pushing the head to rely on pose encodings.
- **Scale mismatch between tokens**: pose encodings are unnormalized, while field features are GN-normalized; imbalance can bias the MLP.
- **Candidate validity thresholding**: `candidate_min_valid_frac` hard-masks candidates, reducing supervision and introducing discontinuities.

## Suggested checks

- Verify EVL `occ_pr` semantics (probability vs logits) and flip `occ_pr_is_logits` if needed.
- Compare frustum-sampled points to depth-rendered rays on a single snippet (sanity visualization).
- Inspect `token_valid` distribution vs candidate depth range; increase depth samples if many out-of-grid.
- Visualize `voxel/T_world_voxel` frame alignment by projecting known world points to voxel space.

