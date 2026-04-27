---
id: 2026-03-30_python_mojo_benchmark_plotting
date: 2026-03-30
title: "Python vs Mojo benchmark plotting utilities"
status: done
topics: [benchmarking, plotting, mojo, python, cli]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/aria_nbv/utils/benchmark_plotting.py
  - aria_nbv/scripts/plot_python_vs_mojo_benchmarks.py
  - aria_nbv/tests/utils/test_benchmark_plotting.py
artifacts:
  - polished latency, speedup, scaling, and throughput comparison plots from CSV benchmark trials
  - CLI entrypoint for rendering benchmark report bundles
assumptions:
  - real Python vs Mojo benchmark measurements will be supplied as CSV input because no Mojo toolchain or measured Mojo outputs were present in the workspace
---

Task

Provide polished benchmarking plots comparing Python against Mojo, verify them by rendering examples locally, and improve the first-pass visuals based on direct inspection.

Method

Added a benchmark plotting utility that loads raw CSV trials, summarizes them per benchmark / implementation / size, and renders grouped latency, speedup, scaling, and optional throughput figures. Added a small CLI wrapper for report generation and extended tests to cover the new plotting surfaces. Rendered a temporary synthetic demo benchmark bundle locally and iterated on chart width, legend ordering, axis labeling, speedup ordering, and subplot spacing after inspecting the exported PNGs.

Findings

The main visual issues only showed up after rendering: the original grouped plots were too cramped for multi-line benchmark labels, the speedup chart sorted problem sizes lexicographically, and repeated scaling x-axis titles visually collided between panels. Widening the figures, enforcing Python-first implementation ordering, adding a throughput chart, fixing numeric size ordering, and limiting the x-axis title to the bottom subplot produced materially cleaner output.

Verification

- `cd aria_nbv && ruff format aria_nbv/utils/benchmark_plotting.py scripts/plot_python_vs_mojo_benchmarks.py tests/utils/test_benchmark_plotting.py`
- `cd aria_nbv && ruff check aria_nbv/utils/benchmark_plotting.py scripts/plot_python_vs_mojo_benchmarks.py tests/utils/test_benchmark_plotting.py`
- `cd aria_nbv && uv run pytest -s tests/utils/test_benchmark_plotting.py`
- `cd aria_nbv && uv run --with 'kaleido==0.2.1' python scripts/plot_python_vs_mojo_benchmarks.py --input /tmp/nbv_python_vs_mojo_demo.csv --out-dir /tmp/nbv_python_mojo_plots --title-prefix 'Python vs Mojo Demo' --write-png`

Canonical state impact

No canonical state doc changed. This adds a benchmarking and diagnostics utility surface but does not change the project's architectural claims.
