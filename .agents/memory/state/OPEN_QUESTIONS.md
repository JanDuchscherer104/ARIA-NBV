---
id: open_questions
updated: 2026-03-24
scope: repo
owner: jan
status: active
tags: [research, nbv, vin, training]
---

# Open Questions

## Research Questions
- What self-supervised objective is suitable for NBV pre-training before oracle imitation learning?
- How should invalid candidate views be handled: hard pruning, explicit penalties, or a mixed strategy?
- Which candidate-generation settings produce the best tradeoff between realism and exploration?
- How should ordinal RRI binning adapt when candidate-generation settings shift the label distribution?
- Which view-conditioned features add the most signal beyond the current EVL voxel context?

## System Questions
- Where should gravity-aligned sampling convenience end and physical rig-frame supervision begin?
- Which diagnostics and acceptance checks are strong enough to catch pose-frame mismatches early?
- What is the minimum stable context bundle Codex needs for code, paper, and experiment tasks without reintroducing large static dumps?
