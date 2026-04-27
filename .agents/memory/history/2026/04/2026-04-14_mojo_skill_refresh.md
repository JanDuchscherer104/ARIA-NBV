---
id: 2026-04-14_mojo_skill_refresh
date: 2026-04-14
title: "Mojo acceleration skill refresh for Apple-Silicon kernel work"
status: done
topics: [mojo, apple-silicon, skills, context7, acceleration]
confidence: high
canonical_updates_needed: []
files_touched:
  - .agents/skills/mojo-nbv-acceleration/SKILL.md
  - .agents/skills/mojo-nbv-acceleration/references/mojo-context7-summary.md
  - .agents/skills/mojo-nbv-acceleration/references/public-acceleration-examples.md
  - .agents/references/context7_library_ids.md
assumptions:
  - The ViSTA-SLAM `curope` donor files should be treated as read-only external references because their file headers are CC BY-NC-SA 4.0, even though the repo root license is broader.
---

Task
- Reworked the Mojo acceleration skill so it can guide actual CUDA-to-Mojo and Apple-Silicon work in this repo instead of only giving high-level evaluation advice.

Method
- Read the repo hot-path state, existing Apple-Silicon Mojo docs, current Mojo backend modules, local `.mojo` kernels, parity tests, and benchmark scripts.
- Pulled official Modular Mojo docs through Context7 for Python interop, pointers, SIMD, packaging, GPU, Metal, and Apple-GPU behavior.
- Inspected public donor examples:
  - `modular/mojo-gpu-puzzles`
  - `zhangganlin/vista-slam` CUDA RoPE extension
  - `furnace-dev/sonic-mojo`
- Rewrote the skill around the repo’s actual backend pattern and added compact agent-facing reference notes.

Findings
- The old skill under-described the repo’s real Mojo integration path, which already uses `mojo.importer`, `PythonModuleBuilder`, and raw CPU buffer staging.
- The shared Context7 library-id reference did not actually list `/websites/modular_mojo`, even though the skill depended on it.
- ViSTA-SLAM is a strong donor example for a narrow CUDA extension behind a stable Python API, but its `curope` files are marked CC BY-NC-SA 4.0 and should not be vendored casually.
- Official Modular material is strong on Python interop, pointers, GPU kernels, and Apple GPU capability, but the best implementation guidance for this repo still comes from combining those docs with the repo-local `.mojo` kernels and benchmark gates.

Verification
- `make check-agent-memory`

Canonical State Impact
- None. The work changed skill and reference surfaces only.
