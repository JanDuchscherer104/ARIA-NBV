# arXiv Source Trees

This directory stores extracted arXiv LaTeX source bundles for papers that directly shape the benchmark design in this repository.

## Included papers

- [EFM3D: A Benchmark for Measuring Progress Towards 3D Egocentric Foundation Models](https://arxiv.org/abs/2406.10224): 2024-06-14. Local source tree: `arXiv-EFM3D/`.
- [GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction](https://arxiv.org/abs/2402.16174): 2024-02-25. Local source tree: `arXiv-GenNBV/`.
- [Project Aria: A New Tool for Egocentric Multi-Modal AI Research](https://arxiv.org/abs/2308.13561): 2023-08-24. Local source tree: `arXiv-project-aria/`.
- [SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model](https://arxiv.org/abs/2403.13064): 2024-03-19. Local source tree: `arXiv-scene-script/`.
- [VIN-NBV: A View Introspection Network for Next-Best-View Selection](https://arxiv.org/abs/2505.06219): 2025-05-09. Local source tree: `arXiv-VIN-NBV/`.
- [Informative Object-centric Next Best View for Object-aware 3D Gaussian Splatting in Cluttered Scenes](https://arxiv.org/abs/2602.08266): 2026-02-09. Local source tree: `arXiv-Instance-NBV/`.
- [PB-NBV: Efficient Projection-Based Next-Best-View Planning Framework for Reconstruction of Unknown Objects](https://arxiv.org/abs/2501.10663): 2025-01-18. Local source tree: `arXiv-PB-NBV/`.
- [Next Best View Selections for Semantic and Dynamic 3D Gaussian Splatting](https://arxiv.org/abs/2512.22771): 2025-12-28. Local source tree: `arXiv-Dynamic-3DGS/`.
- [Hestia: Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction](https://arxiv.org/abs/2508.01014): 2025-08-01. Local source tree: `arXiv-Hestia/`.

## Manifest and refresh workflow

The checked-in manifest for this directory lives at:

- `sources.jsonl`

Each JSONL row describes one paper and the local output names needed to fetch its arXiv e-print source tree and, optionally, its PDF. The current schema is:

- required: `arxiv_id`, `tex_dir`
- optional: `title`, `short_title`, `source_url`, `pdf_url`, `pdf_file`

Run the downloader from the repo root:

```bash
uv run python scripts/download_arxiv_tex_src.py docs/literature/sources.jsonl
uv run python scripts/download_arxiv_tex_src.py docs/literature/sources.jsonl --download-pdfs
```

By default the script writes extracted source trees into `docs/literature/tex-src/` and PDFs into `docs/literature/pdf/`. Use `--overwrite` to replace an existing extracted tree or PDF target.

## Why these sources are kept locally

- to inspect the exact paper contribution statements and method structure
- to recover figure, notation, and pipeline terminology directly from the source
- to anchor our wrapper and stage design to the paper, not only to the GitHub README
- to make future report writing and figure reuse easier

## Most useful files


### EFM3D

- `arXiv-EFM3D/main.tex`
- `arXiv-EFM3D/method.tex`
- `arXiv-EFM3D/dataset.tex`

### GenNBV

- `arXiv-GenNBV/main.tex`
- `arXiv-GenNBV/3-Method.tex`

### Project Aria

- `arXiv-project-aria/main.tex`
- `arXiv-project-aria/device.tex`
- `arXiv-project-aria/mps.tex`

### SceneScript

- `arXiv-scene-script/main.tex`
- `arXiv-scene-script/sections/structured_scene_language.tex`

### VIN-NBV

- `arXiv-VIN-NBV/main.tex`
- `arXiv-VIN-NBV/sec/3_methods.tex`

### Instance-NBV

- `arXiv-Instance-NBV/main.tex`

### PB-NBV

- `arXiv-PB-NBV/main.tex`

### Dynamic 3DGS NBV

- `arXiv-Dynamic-3DGS/main.tex`

### Hestia

- `arXiv-Hestia/main.tex`

The repo-level lookup material for these papers lives in:

- `.agents/references/agent_reference.md`
