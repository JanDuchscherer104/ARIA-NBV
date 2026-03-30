---
id: 2026-03-30_mojo_benchmark_report
date: 2026-03-30
title: "Mojo Benchmark Report Script"
status: done
topics: [mojo, benchmarking, plotting, memory]
confidence: high
canonical_updates_needed: []
files_touched:
  - aria_nbv/scripts/benchmark_mojo_candidate_generation.py
  - aria_nbv/scripts/report_mojo_candidate_generation.py
artifacts:
  - /tmp/nbv-mojo-report/benchmark_summary.json
  - /tmp/nbv-mojo-report/benchmark_timings.csv
  - /tmp/nbv-mojo-report/benchmark_runtime.png
  - /tmp/nbv-mojo-report/mojo_impl_overview.png
---

Task: Provide visual and numeric representations of the new Mojo candidate-generation path, benchmark it against the current Python/Trimesh backend with Matplotlib, and summarize the repo test inventory.

Method: Refactored the benchmark helper into an importable `run_benchmark_case(...)`, added `report_mojo_candidate_generation.py` to generate JSON, CSV, and PNG artifacts, then ran the report into `/tmp/nbv-mojo-report`. Test inventory counts were collected with a static AST scan over `aria_nbv/tests`.

Findings: The report now emits a runtime comparison figure plus an implementation-topology figure. On the current run (`repeats=7`, `num_samples=256`, `mesh_faces=1280`), the clearance-only case measured about `11.66x` speedup (`104.40 ms` vs `8.95 ms`) and the full pipeline measured about `1.56x` speedup (`197.40 ms` vs `126.54 ms`) while preserving candidate equivalence. The repo currently contains `91` test files and `299` statically discovered test functions.

Verification:
- `ruff check aria_nbv/scripts/benchmark_mojo_candidate_generation.py aria_nbv/scripts/report_mojo_candidate_generation.py`
- `ARIA_NBV_MOJO_SITE_PACKAGES=/home/jandu/repos/NBV/.mojo-venv/lib/python3.12/site-packages /tmp/nbv-mojo-skill-pr/aria_nbv/.venv/bin/python /tmp/nbv-mojo-skill-pr/aria_nbv/scripts/report_mojo_candidate_generation.py --out-dir /tmp/nbv-mojo-report`

Canonical state impact: none beyond the earlier backend debrief and state update.
