#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style

= Offline cache and batching

#let cache = json("/typst/slides/data/offline_cache_stats.json").offline_cache
#let s = cache.sample_sizes_mb

This appendix summarizes why we cache oracle outputs, what is stored, and the
approximate storage footprint per snippet and for the full ASE mesh subset.

== Motivation

The oracle pipeline combines candidate sampling, depth rendering, backprojection,
and point-to-mesh scoring. Each step is GPU-heavy and hard to parallelize inside
PyTorch. In practice, this results in per-snippet runtimes on the order of tens
of seconds, while the EVL backbone alone can consume multiple GB of GPU memory
per forward pass. Caching makes training a standard supervised learning problem
and enables larger batch sizes for noisy ordinal supervision.

== Storage footprint

Let $S_"cur"$ denote the size of the cached subset and $f_"cover"$ the
scene coverage fraction. We estimate the full-coverage size as:

$ S_"full" approx S_"cur" / f_"cover" $.

For the current cache we observe:

- $S_"cur" = #cache.samples_size_gb$ GB for `samples/*.pt`.
- $S_"vin" = #cache.vin_snippet_cache_gb$ GB for `vin_snippet_cache`.
- $S_"full" approx #cache.full_coverage_total_gb$ GB for 100 mesh scenes.

#figure(
  kind: "table",
  supplement: [Table],
  caption: [Representative per-snippet tensor footprint (CPU, float32).],
  text(size: 8.5pt)[
    #let rows = (
      ([backbone_out], [#s.backbone MB]),
      ([candidate_pcs], [#s.candidate_pcs MB]),
      ([depths], [#s.depths MB]),
      ([candidates], [#s.candidates MB]),
      ([rri], [#s.rri MB]),
      ([vin_snippet], [#s.vin_snippet MB]),
      ([total + backbone], [#s.total_with_backbone MB]),
      ([min train payload], [#s.total_min_train MB]),
    )
    #table(
      columns: (16em, auto),
      align: (left, left),
      toprule(),
      table.header([Field], [MB]),
      midrule(),
      ..rows.flatten(),
      bottomrule(),
    )
  ],
)

== Minimal training payload

The bare minimum for VIN training (with cached candidates) is:

- Candidate poses and PyTorch3D camera parameters.
- Oracle targets (RRI + point-to-mesh components).
- `VinSnippetView` (semi-dense points, lengths, trajectory).

This minimal payload is approximately #s.total_min_train MB per snippet in the
current configuration. Caching the full EVL backbone output adds ~#s.backbone MB
per snippet, dominating storage when enabled.
