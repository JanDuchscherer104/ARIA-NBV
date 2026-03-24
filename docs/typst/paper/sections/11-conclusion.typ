= Conclusion

#import "../../shared/macros.typ": *

We presented an oracle supervision and diagnostics pipeline for quality-driven
NBV research in egocentric indoor scenes. The approach computes per-candidate
oracle RRI labels from ASE ground-truth meshes and semi-dense reconstructions,
and provides tooling to inspect candidate generation, depth rendering, and
surface-error behavior. We also outlined an ordinal label representation
(CORAL) and trained a preliminary VIN v3 candidate scorer on frozen EVL
features. Learning a full next-best-view policy on top of these labels remains
future work, including entity-aware objectives and learning more expressive view-conditioned representations.
Furthermore, we have implemented a rich ML experiment management and development suite with offline caching, W&B + Optuna integration, various helpful cli functionalites and a comprehensive Streamlit app for data inspection and debugging.

#let cache = json("/typst/slides/data/offline_cache_stats.json").offline_cache
#let wb = json("/typst/slides/data/wandb_rtjvfyyp_summary.json").wandb
#let m = wb.metrics
// </rm>

Quantitatively, our current offline cache covers #cache.unique_scenes ASE-EFM GT
scenes with #cache.index_entries oracle-labelled snippets (train/val
#cache.train_entries/#cache.val_entries) drawn from #cache.total_snippets
snippet windows. The best VIN v3 run (#code-inline[v03-best], W&B id
#code-inline[#wb.run_id]) achieves non-trivial ranking performance (val Spearman
≈ #calc.round(m.at("val-aux/spearman"), digits: 3), val top-3 accuracy
≈ #calc.round(m.at("val-aux/top3_accuracy"), digits: 3)), indicating that the
oracle labels and feature set can support learning while leaving room for
further improvements and controlled ablations.
// </rm>
