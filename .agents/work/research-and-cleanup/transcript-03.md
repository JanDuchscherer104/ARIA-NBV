[User-Turn01]
Please do a cross review of generalizable / transferable best practices, skills, agents scaffold, ...

https://github.com/JanDuchscherer104/ARIA-NBV vs https://github.com/JanDuchscherer104/prml-vslam. project ARIA-NBV's scaffolding isn't as good, it contains plenty of slop in the docs/ folder!

we need to develop better custom skills and also mabye employ oh my codex or something alike.

[Agent-Turn02]
## Executive diagnosis

Your read is right: **PRML-VSLAM is currently the better scaffold**, while **ARIA-NBV is the better research system**. ARIA-NBV has richer domain machinery—ASE/EFM/VIN/RRI, oracle caches, W&B/Optuna, Streamlit diagnostics, Typst/Quarto reporting—but the surrounding repo scaffold is noisy and partially copied. PRML-VSLAM has cleaner repo-level ownership, clearer docs boundaries, better CI/dev ergonomics, and a more disciplined agent-memory/tooling layer.

The key move is not to copy PRML wholesale. It is to **port PRML’s scaffold discipline into ARIA while keeping ARIA’s research-specific context-routing power**.

The methodological spine should remain ARIA’s: quality-driven NBV using RRI rather than coverage-only objectives, grounded in VIN-NBV, GenNBV, SceneScript/ASE, and the current Aria-VIN-NBV work.

---

## Cross-review: what transfers from PRML-VSLAM

| Area                        | PRML-VSLAM pattern                                                                                                                                          | ARIA-NBV current state                                                                                                                                                     | Transfer recommendation                                                                                                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Repo front door**         | README states mission, documentation map, status, implemented/not-yet-implemented items, and quick entry commands. ([GitHub][1])                            | README is mostly command snippets; GitHub “About” has no description/topic metadata. ([GitHub][2])                                                                         | Rewrite ARIA `README.md` around mission, current research claim, status, reproducibility, and quick-start. Keep command catalog secondary.                                               |
| **Docs boundary**           | Docs site is minimal: report, update slides, build contract. ([Janduchscherer104][3])                                                                       | Public site mixes dataset resources, literature reviews, implementation guides, external implementation indices, and agent memory/scaffold pages. ([Janduchscherer104][4]) | Split public docs from internal agent/generated context. Public site should serve paper, report, experiments, API, and maybe curated implementation guides—not agent operational sprawl. |
| **Agent root guidance**     | PRML root `AGENTS.md` is short and points to sources of truth, local nested guides, and `make ci`. ([GitHub][5])                                            | ARIA root `AGENTS.md` is much denser and includes broad routing, state, gotchas, commands, debrief rules, and many references. ([GitHub][6])                               | Convert ARIA root `AGENTS.md` to a thin dispatcher. Move long guidance into skills/references.                                                                                           |
| **Nested package guidance** | PRML package guide is strong: typed contracts, config-as-factory, docstring contracts, frame semantics, wrapper boundaries, and verification. ([GitHub][7]) | ARIA package guide is already good and domain-specific: `PoseTW`/`CameraTW`, EFM/ATEK conventions, config-as-factory, Plotly diagnostics, frame semantics. ([GitHub][8])   | Keep ARIA nested guides. They are one of the scaffold’s strengths.                                                                                                                       |
| **Tooling baseline**        | PRML has root `pyproject.toml`, pre-commit, `make ci`, lint/test/docs build targets, and GitHub CI. ([GitHub][9])                                           | ARIA has a package `pyproject.toml` under `aria_nbv/`, no root pre-commit file visible, and only Quarto publish workflow visible. ([GitHub][10])                           | Add root-level dev contract: pre-commit, CI, `make ci`, `make lint`, `make test`, `make docs-check`, and `make agent-check`.                                                             |
| **Agent memory/backlog**    | PRML uses compact `.agents/AGENTS_INTERNAL_DB.md`, TOML issues/todos/resolved, and `make agents-db`. ([GitHub][11])                                         | ARIA has a richer `.agents/memory/state/` structure and good project-state docs, but also copied/misaligned skill surfaces. ([GitHub][12])                                 | Keep ARIA’s state docs, but add PRML-style ranked TOML backlog for actionable scaffold/code/docs debt.                                                                                   |
| **Codex/plugin layer**      | PRML has `.codex`, `.codex-plugin`, MemPalace MCP, startup hook, and skills configured. ([GitHub][13])                                                      | ARIA has `.agents/skills` and context tooling, but no equivalent visible Codex runtime/plugin scaffold. ([GitHub][14])                                                     | Add optional Codex/OMX layer only after pruning docs and skills. Use templates, not host-specific config.                                                                                |

---

## Concrete ARIA-NBV scaffold problems

The most important problem is **copied scaffold drift**. ARIA’s `agents-db` skill still says it is for “PRML VSLAM’s internal agent memory,” which is a direct wrong-repo leak. ([GitHub][15]) The ARIA `code-review` skill is worse: its metadata says `code-review-litkg-rs`, and the content talks about `litkg-rs`, `CODEOWNER.md`, `docs/architecture.md`, Rust benchmark promotion, and autoresearch winner branches—none of which match ARIA-NBV. ([GitHub][16])

That is exactly the class of “slop” to eliminate first, because agents will faithfully follow wrong context. This is not just style. A 2026 arXiv study on repository-level `AGENTS.md` files found that unnecessary context can reduce coding-agent task success and increase inference cost by more than 20%, and concluded that human-written context files should contain only minimal requirements. ([arXiv][17])

ARIA also has **public-docs / internal-docs mixing**. The project site exposes dataset resources, implementation guides, external implementation indices, and agent memory/scaffold pages in the same navigation area. ([Janduchscherer104][4]) That makes sense during exploration, but it is bad for a thesis/project surface: advisor-facing/public docs should not mirror every operational aid.

---

## Recommended target scaffold for ARIA-NBV

Use this layout as the target:

```text
ARIA-NBV/
  README.md                     # mission, status, quick start, reproducibility
  SETUP.md                      # environment, CUDA, dataset access, caches
  CONTRIBUTING.md               # lightweight dev workflow
  AGENTS.md                     # thin root dispatcher only
  pyproject.toml                # optional: root workspace/dev if feasible
  Makefile                      # ci, lint, test, docs, context, agent checks
  .pre-commit-config.yaml
  .github/workflows/
    ci.yml
    quarto-publish.yml

  aria_nbv/
    pyproject.toml              # keep if inner workspace remains intentional
    AGENTS.md
    aria_nbv/
      data_handling/AGENTS.md
      rri_metrics/AGENTS.md
      vin/AGENTS.md
      planning/AGENTS.md        # add once non-myopic planning stabilizes

  .agents/
    skills/
      repo-context-router/
      oracle-rri-labeling/
      vin-training-eval/
      nbv-geometry-contracts/
      docs-paper-sync/
      simplification-and-backlog/
      code-review/
      dataset-cache-ops/
    references/
      python_conventions.md
      coordinate_frames.md
      external_library_ids.md
      operator_quick_reference.md
    memory/
      state/
        PROJECT_STATE.md
        DECISIONS.md
        OPEN_QUESTIONS.md
        GOTCHAS.md
      history/
    backlog/
      issues.toml
      todos.toml
      refactors.toml
      resolved.toml

  docs/
    index.qmd
    report/ or typst/paper/
    slides/
    contents/
      theory/
      implementation/
      experiments/
      literature/
    _generated/                 # ignored unless explicitly needed
```

The central design rule: **public docs explain the project; `.agents` helps agents work on the project.** Generated context should not live in the public docs tree unless it is intentionally published.

---

## Custom skills ARIA-NBV should have

### 1. `repo-context-router`

Purpose: replace the current broad `aria-nbv-context` skill with a slimmer router.

Keep the good part: ARIA’s current context skill has a useful progressive-disclosure ladder: paper → canonical state → source index → references → outlines → targeted search → heavy context only last. ([GitHub][18])

Change the bad part: make it shorter and less prescriptive. Its job is only to localize work, then hand off.

**Trigger:** broad or cross-surface tasks.
**Do not trigger:** exact file edits, simple formatting, one-module fixes.
**Acceptance criterion:** after using it, the agent should name the exact target files/modules and stop routing.

### 2. `nbv-geometry-contracts`

Purpose: protect the highest-risk technical layer: poses, cameras, coordinate frames, candidate view projection, PyTorch3D conventions, LUF/NDC, and CW90 gotchas.

ARIA already has good module rules around preserving candidate-vs-rig frame semantics and preventing display-only rotations from leaking into model/cache inputs. ([GitHub][19]) This deserves its own skill because frame bugs are thesis-grade failure modes.

**Owns:** `PoseTW`, `CameraTW`, `T_target_source`, LUF frame semantics, world/rig/candidate/voxel transforms, NDC backprojection, candidate frusta.
**Verification:** geometry smoke tests, rendering/backprojection tests, and projection diagnostics.

### 3. `oracle-rri-labeling`

Purpose: own candidate generation, mesh rendering, candidate depth, backprojection, point↔mesh distance, RRI, binning, and label cache semantics.

ARIA’s RRI module guide already treats oracle-label semantics, binning definitions, and metric names as contract changes. ([GitHub][20]) Promote that into a skill so agents do not treat RRI edits as local refactors.

**Owns:** `oracle_rri.py`, `metrics.py`, `types.py`, `rri_binning.py`, `coral.py`, oracle cache schemas, label distributions.
**Verification:** RRI unit tests, binning tests, candidate render tests, and label-distribution plots.

### 4. `vin-training-eval`

Purpose: own VIN scorer experiments, offline cache training, Lightning, W&B/Optuna, CORAL decoding, validation metrics, confusion matrices, and stage-shift diagnostics.

The ARIA paper/report states that VIN is still a preliminary candidate scorer, not a complete end-to-end NBV policy.  The skill should prevent overclaiming and enforce controlled ablations.

**Owns:** VIN v3/v4, Lightning datamodule/module, offline cache, training diagnostics, top-k accuracy, Spearman, calibration, ablation logs.
**Verification:** targeted VIN tests, Lightning batch/collate tests, one training dry-run/smoke when feasible.

### 5. `docs-paper-sync`

Purpose: enforce one narrative source of truth across Typst paper, Quarto docs, slides, README, and project state.

ARIA docs guidance already says Typst paper is the highest-level ground truth, Quarto should align to the paper, bibliography should be single-source, and generated artifacts should not be stored in tracked docs paths unless required. ([GitHub][21]) This skill should own docs cleanup.

**Owns:** Typst paper, slides, Quarto pages, bibliography, public site nav, docs pruning.
**Verification:** `quarto check`, Typst compile, bibliography check, broken-link check.

### 6. `simplification-and-backlog`

Purpose: merge the good parts of ARIA’s `simplification` and PRML’s `agents-db-and-simplification`.

ARIA’s simplification skill is actually strong: it has deletion-over-abstraction bias, ruthless mode, DTO/config/enum rules, and tool decision rules. ([GitHub][22]) Keep that. Fix the repo-specific leaks and connect it to a TOML backlog.

**Owns:** code pruning, docs pruning, dead skill deletion, copied scaffold cleanup, LOC-negative refactors.
**Verification:** `make loc`, focused tests, `make agents-db`, `make ci`.

### 7. `code-review`

Purpose: replace the current wrong `code-review-litkg-rs` skill with an ARIA-NBV review gate.

**Review targets:** geometry regressions, RRI semantic drift, cache schema drift, docs overclaims, coordinate-frame leaks, test gaps, notebook-only logic, generated-doc pollution.
**Output:** findings first, severity-ranked, with file/line references.

### 8. `dataset-cache-ops`

Purpose: own ASE download, ATEK shards, mesh availability, offline cache creation, VIN snippet cache, and storage pressure.

ARIA’s README already exposes ASE downloader and cache CLIs, including mesh-scene counts and cache commands. ([GitHub][2]) Turn that into a proper skill/runbook.

**Owns:** `nbv-downloader`, `nbv-cache-samples`, `nbv-cache-vin-snippets`, cache manifests, split indices, storage estimates.
**Verification:** dry-run list, small-scene download, cache manifest validation.

---

## Oh My Codex / OMX recommendation

Use **Oh My Codex only as an orchestration layer**, not as a substitute for fixing ARIA’s repo hygiene.

OMX advertises itself as a workflow layer for OpenAI Codex CLI, with canonical `$deep-interview`, `$ralplan`, `$team`, and `$ralph` workflows, persistent `.omx/` state, project guidance, and team execution. ([GitHub][23]) It is designed for macOS/Linux with Codex CLI; the README explicitly says native Windows/Codex App are not the tuned/default path. ([GitHub][23])

I would pilot it like this:

1. **Do not install/publish a full OMX setup into ARIA first.** First prune the existing wrong skills and docs sprawl.
2. Add **`.codex/config.example.toml`**, not a committed personal `.codex/config.toml`.
3. Add **`.codex/hooks.example.json`** with safe hooks only: context refresh, dirty-tree warning, maybe agent-memory validation.
4. Keep `.omx/` ignored.
5. Use OMX for only three workflows at first:

   * `$deep-interview` for vague thesis-scope changes.
   * `$ralplan` for large refactors or docs reorganization.
   * `$team` for parallel audits: docs slop audit, skill audit, test-gap audit.
6. Avoid using OMX team mode for geometry/RRI code changes until CI and targeted tests are reliable.

PRML’s MemPalace setup is a useful reference: it has a repo-local plugin, a MemPalace MCP server, and a session-start hook that refreshes context. ([GitHub][24]) But PRML’s checked-in `.codex/config.toml` also contains very host-specific runtime details, including a Windows Docker path and local MCP server assumptions. ([GitHub][25]) For ARIA, use **templates** and **operator-local config**, not committed personal config.

---

## Docs cleanup plan for ARIA-NBV

### Phase 1: classify everything in `docs/`

Create an inventory with four labels:

| Label       | Meaning                                 | Action                                           |
| ----------- | --------------------------------------- | ------------------------------------------------ |
| `public`    | should appear on GitHub Pages           | keep in nav                                      |
| `paper`     | thesis/report/slides source             | keep, but do not duplicate in Quarto             |
| `internal`  | agent/operator/context docs             | move to `.agents/` or hide from nav              |
| `generated` | API docs, context dumps, frozen outputs | ignore or regenerate; publish only intentionally |

Likely moves:

```text
docs/_generated/context/        -> .agents/generated/context/ or ignored cache
docs/contents/resources/agent_scaffold/ -> .agents/references/ or hidden generated mirror
docs/literature/tex-src/        -> literature/ or .agents/references/literature/
docs/contents/todos.qmd         -> .agents/backlog/ + short public roadmap only
docs/contents/questions.qmd     -> keep only if advisor-facing; otherwise state/open questions
```

### Phase 2: make `docs/index.qmd` a clean public landing page

It should contain:

```text
1. Project abstract / thesis claim
2. Current status
3. Reproducibility links
4. Paper/report/slides
5. Selected implementation guides
6. Selected literature notes
7. API reference
```

It should not contain agent memory, internal generated context, full scaffold mirrors, or raw task logs.

### Phase 3: introduce `docs/_quarto.yml` nav discipline

Add explicit nav groups:

```yaml
website:
  sidebar:
    contents:
      - index.qmd
      - section: "Research"
        contents:
          - contents/theory/rri_theory.qmd
          - contents/experiments/...
      - section: "Implementation"
        contents:
          - contents/impl/oracle_rri.qmd
          - contents/impl/vin_training.qmd
      - section: "References"
        contents:
          - contents/literature/vin_nbv.qmd
          - contents/literature/gennbv.qmd
```

Everything else should be unlisted, internal, or generated.

---

## What to copy directly from PRML-VSLAM

Copy/adapt these patterns:

1. **README structure**: documentation map, status, quick entry, implemented/not implemented. PRML’s README does this well. ([GitHub][1])
2. **`SETUP.md` split**: environment setup should not bloat README. PRML separates base setup, optional CUDA/ViSTA setup, MemPalace, validation, and Streamlit workbench. ([GitHub][26])
3. **Root `make ci` contract**: PRML’s Makefile makes CI, lint, tests, report/slides, agent DB, LOC, MemPalace, and graphify visible. ([GitHub][27])
4. **Pre-commit**: PRML uses pre-commit for merge conflict checks, TOML checks, whitespace, Ruff format, and Ruff check. ([GitHub][28])
5. **Docs `AGENTS.md` discipline**: PRML docs guidance explicitly avoids operational sprawl and command duplication, points detail to owning docs, and keeps report/slides scoped. ([GitHub][29])
6. **Package contract style**: typed interfaces, canonical config factory, explicit frame semantics, no silent fallback, thin wrappers. ([GitHub][7])

Do not copy PRML’s project-specific content, host-local `.codex` config, or generic external skills unless ARIA actually needs them.

---

## Immediate high-value fixes

1. **Delete or rewrite wrong copied skills.**

   * Rewrite `code-review` from scratch.
   * Fix `agents-db` references to PRML.
   * Audit all `SKILL.md` files for wrong repo names, wrong file paths, wrong commands, and stale workflow claims.

2. **Add root `README.md` structure.**

   * Mission.
   * Current claim.
   * What works.
   * What is experimental.
   * Quick start.
   * Reproducibility.
   * Docs map.
   * “Do not overclaim” status note.

3. **Add root CI and pre-commit.**

   * `ruff format --check`
   * `ruff check`
   * targeted pytest
   * docs smoke check
   * agent memory validation
   * no generated-doc pollution check

4. **Move agent/public docs apart.**

   * Agent scaffold belongs in `.agents`.
   * Public docs should not mirror operational memory by default.

5. **Create ARIA-specific skill suite.**

   * Start with `repo-context-router`, `oracle-rri-labeling`, `nbv-geometry-contracts`, `vin-training-eval`, `docs-paper-sync`, `simplification-and-backlog`, `code-review`, `dataset-cache-ops`.

6. **Pilot OMX after cleanup.**

   * Add template config only.
   * Keep `.omx/` ignored.
   * Use it for planning/audit/refactor orchestration, not as a substitute for CI and skill hygiene.

---

## Practical acceptance criteria

ARIA’s scaffold is “good enough” when:

```text
[ ] README explains the project without requiring the GitHub Pages site.
[ ] SETUP.md can get a fresh machine to a small smoke run.
[ ] make ci exists and is the default local verification gate.
[ ] pre-commit exists and matches CI.
[ ] Root AGENTS.md is thin and under control.
[ ] Every skill is ARIA-specific; no PRML/litkg/copied repo leakage.
[ ] docs/ contains public/research docs, not internal agent scratch/context dumps.
[ ] Generated context is ignored or clearly isolated.
[ ] .agents/memory/state contains current truth.
[ ] .agents/backlog or equivalent contains actionable scaffold debt.
[ ] Code-review skill checks geometry/RRI/cache/docs-overclaim risks.
[ ] OMX/Codex config is templated, not host-personal.
```

My strongest recommendation: **do the cleanup in this order: wrong skills → README/SETUP/CI → docs partition → new ARIA-specific skills → optional OMX pilot.** That sequence avoids building a fancier agent system on top of polluted context.

[1]: https://github.com/JanDuchscherer104/prml-vslam "GitHub - JanDuchscherer104/prml-vslam: Off-device monocular dense VSLAM benchmark for uncalibrated smartphone video streams, evaluating trajectory accuracy, dense 3D reconstruction quality, and efficiency on ADVIO and custom ARCore-logged datasets. · GitHub"
[2]: https://github.com/JanDuchscherer104/ARIA-NBV "GitHub - JanDuchscherer104/ARIA-NBV · GitHub"
[3]: https://janduchscherer104.github.io/prml-vslam/ "PRML VSLAM Publications – PRML VSLAM"
[4]: https://janduchscherer104.github.io/ARIA-NBV/ "Next-Best-View Planning with Foundation Models – Semantic NBV Planning"
[5]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/AGENTS.md "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/AGENTS.md "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/src/prml_vslam/AGENTS.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/AGENTS.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/pyproject.toml "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/pyproject.toml "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.agents/AGENTS_INTERNAL_DB.md "raw.githubusercontent.com"
[12]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.agents/memory/state "ARIA-NBV/.agents/memory/state at main · JanDuchscherer104/ARIA-NBV · GitHub"
[13]: https://github.com/JanDuchscherer104/prml-vslam/tree/main/.codex-plugin "prml-vslam/.codex-plugin at main · JanDuchscherer104/prml-vslam · GitHub"
[14]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.agents/skills "ARIA-NBV/.agents/skills at main · JanDuchscherer104/ARIA-NBV · GitHub"
[15]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.agents/skills/agents-db/SKILL.md "raw.githubusercontent.com"
[16]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.agents/skills/code-review/SKILL.md "raw.githubusercontent.com"
[17]: https://arxiv.org/abs/2602.11988 "[2602.11988] Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?"
[18]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.agents/skills/aria-nbv-context/SKILL.md "raw.githubusercontent.com"
[19]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/aria_nbv/vin/AGENTS.md "raw.githubusercontent.com"
[20]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/aria_nbv/rri_metrics/AGENTS.md "raw.githubusercontent.com"
[21]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/AGENTS.md "raw.githubusercontent.com"
[22]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.agents/skills/simplification/SKILL.md "raw.githubusercontent.com"
[23]: https://github.com/Yeachan-Heo/oh-my-codex "GitHub - Yeachan-Heo/oh-my-codex: OmX - Oh My codeX: Your codex is not alone. Add hooks, agent teams, HUDs, and so much more. · GitHub"
[24]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.codex-plugin/plugin.json "raw.githubusercontent.com"
[25]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.codex/config.toml "raw.githubusercontent.com"
[26]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/SETUP.md "raw.githubusercontent.com"
[27]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/Makefile "raw.githubusercontent.com"
[28]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.pre-commit-config.yaml "raw.githubusercontent.com"
[29]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/docs/AGENTS.md "raw.githubusercontent.com"


[User-Turn03]
Given the current state of our repo. please give me comprehensive GH issues for all actionable items in the confines of this project.
give roadmap (till end of september), milestones, action items. and for all (GH issues!)

[Agent-Turn04]
I prepared this as **GitHub-ready issue backlog + September roadmap**, not as actual created GitHub issues. The repo currently has **1 open issue**, **5 open draft PRs**, **0 milestones**, and only the default 9 labels, so the first issue should bootstrap the tracker itself before creating the rest. ([GitHub][1])

Artifacts:

* [Full roadmap + copy-paste GitHub issue bodies](sandbox:/mnt/data/aria_nbv_roadmap_gh_issues.md)
* [Machine-readable issues JSON](sandbox:/mnt/data/aria_nbv_roadmap_gh_issues.json)
* [Issue import pack: body files + `create_issues.sh`](sandbox:/mnt/data/aria_nbv_issue_pack.zip)

## Roadmap spine

The roadmap keeps the thesis centered on **quality-driven RRI for egocentric NBV**, with **oracle RRI + VIN scoring + non-myopic discrete planning** as the core. That matches the current project state: the repo already frames RRI as the primary objective, discrete candidate ranking as the canonical training/evaluation surface, and non-myopic planning as incremental work before any continuous-control claim. ([GitHub][2]) It also matches the paper’s current claim: VIN remains a preliminary candidate scorer rather than a complete end-to-end NBV policy.

The research basis remains: SceneScript/ASE for large-scale structured egocentric indoor data, GenNBV as the continuous free-space RL/coverage-policy contrast, and VIN-NBV as the direct reconstruction-quality/RRI inspiration.

| Milestone                                           |        Due | Outcome                                                                    |
| --------------------------------------------------- | ---------: | -------------------------------------------------------------------------- |
| **M0 — Governance & scope lock**                    | 2026-05-03 | Tracking system exists; thesis scope locked.                               |
| **M1 — Repo hygiene & reproducibility baseline**    | 2026-05-24 | README/SETUP/CI/docs/skills are trustworthy.                               |
| **M2 — Oracle/data correctness & scale gate**       | 2026-06-21 | Geometry, device, cache, and oracle bottlenecks are controlled.            |
| **M3 — VIN scorer baseline & controlled ablations** | 2026-07-19 | VIN becomes an evidence-backed baseline, not architecture-search noise.    |
| **M4 — Non-myopic planning baselines**              | 2026-08-16 | One-step, close-greedy, beam, and RL scaffolds are comparable.             |
| **M5 — Scale, entity-aware extension, and reports** | 2026-09-13 | Data scaling, strongest extension, and final experiment reports are ready. |
| **M6 — Thesis freeze & final delivery**             | 2026-09-30 | Paper, docs, slides, configs, release package, and demo are frozen.        |

## Immediate notes from current repo state

The current `README.md` is still effectively a short command dump rather than a project front door; it has a one-line “Seminar” heading and CLI snippets, while the GitHub repo metadata still says no description/topics are provided. ([GitHub][3]) The docs tree still exposes many legacy/internal surfaces, including `resources/agent_scaffold`, `todos.qmd`, `roadmap.qmd`, older VIN v2 proposal pages, external implementation indices, and implementation notebooks/pages all under the public docs area. ([GitHub][4])

Several open draft PRs already cover parts of the cleanup: PR #8 prunes stale docs and aligns Quarto/Typst, PR #9 refactors agent scaffold guidance and adds scaffold validation, and PR #6 deduplicates helper surfaces. These should be triaged before opening duplicate issues. ([GitHub][5])

## Proposed GitHub issues

Use **ARIA-GH-001** first to create labels/milestones, then create the rest. I recommend creating all P0 issues immediately, all P1 issues by milestone start, and P2 issues only once the milestone is stable.

### M0 — Governance & scope lock

**ARIA-GH-001 — Create roadmap milestones, labels, issue templates, and project board**
Priority: P0. Labels: `type:infra`, `area:project-management`, `priority:P0`.
Acceptance: M0–M6 milestones exist; area/type/priority labels exist; issue templates exist; project board groups by milestone and priority.

**ARIA-GH-002 — Lock thesis scope and September success criteria**
Priority: P0. Labels: `type:decision`, `area:thesis`, `area:planning`, `priority:P0`.
Acceptance: one-page scope memo distinguishes thesis core from stretch work; success metrics are defined; M1–M4 issues do not depend on unresolved stretch ideas.

**ARIA-GH-003 — Triage open draft PR backlog and convert residual work into issues**
Priority: P0. Labels: `type:infra`, `area:project-management`, `priority:P0`.
Acceptance: PR #3, #5, #6, #8, and #9 are merged, closed, or assigned to a milestone; useful residual work becomes issues; PR #9 Copilot comments are resolved.

### M1 — Repo hygiene & reproducibility baseline

**ARIA-GH-004 — Rewrite README, add SETUP.md and CONTRIBUTING.md**
Priority: P0. Labels: `type:docs`, `area:docs`, `area:reproducibility`, `priority:P0`.
Acceptance: README explains mission/status/claim/quick start; SETUP covers env/data/cache smoke flow; CONTRIBUTING defines branch, PR, verification, and debrief rules.

**ARIA-GH-005 — Add root `make ci`, package checks, docs checks, and pre-commit parity**
Priority: P0. Labels: `type:infra`, `area:ci`, `area:reproducibility`, `priority:P0`.
Acceptance: `make ci` exists; CI runs on PR/push; pre-commit and CI share core checks; heavy GPU/data tests are opt-in.

**ARIA-GH-006 — Separate public docs from internal agent/generated surfaces**
Priority: P0. Labels: `type:refactor`, `area:docs`, `area:agents`, `priority:P0`.
Acceptance: public nav is reader-facing; internal memory/scaffold content lives under `.agents/` or curated hidden/generated pages; Quarto render passes.

**ARIA-GH-007 — Prune stale Quarto pages and align docs with Typst paper/current code**
Priority: P0. Labels: `type:docs`, `area:docs`, `area:thesis`, `priority:P0`.
Acceptance: `todos.qmd`, `roadmap.qmd`, VIN v2 pages, implementation pages, README, and Typst paper agree on current state.

**ARIA-GH-008 — Replace copied/sloppy skills with ARIA-specific custom skills**
Priority: P0. Labels: `type:refactor`, `area:agents`, `area:scaffold`, `priority:P0`.
Acceptance: no `SKILL.md` references wrong repos/paths; skills exist for context routing, docs-paper sync, geometry contracts, oracle RRI, VIN training, dataset/cache ops, code review, and simplification.

**ARIA-GH-009 — Add scaffold validator and agent DB for issues/todos/resolved work**
Priority: P1. Labels: `type:infra`, `area:agents`, `area:scaffold`, `priority:P1`.
Acceptance: `make check-agent-scaffold` passes; backticked paths are validated; `.agents/issues.toml`, `.agents/todos.toml`, and `.agents/resolved.toml` are usable.

**ARIA-GH-010 — Pilot OMX/Codex templates without committing personal runtime config**
Priority: P2. Labels: `type:experiment`, `area:agents`, `area:scaffold`, `priority:P2`.
Acceptance: `.codex/config.example.toml` and hook templates exist; `.omx/` is ignored; no personal paths/API keys are committed. OMX is currently active and fast-moving, with install/setup workflows and native hooks/team runtime documented, so this should remain opt-in until repo hygiene is done. ([GitHub][6])

**ARIA-GH-011 — Create one-scene data/cache/training smoke tutorial**
Priority: P0. Labels: `type:docs`, `area:data`, `area:reproducibility`, `priority:P0`.
Acceptance: one small workflow verifies downloader, dataset, oracle cache reader, VIN batch construction, and model forward without requiring the full dataset.

### M2 — Oracle/data correctness & scale gate

**ARIA-GH-012 — Update existing #7: centralize device selection and make oracle pipeline MPS/CUDA/CPU-safe**
Priority: P0. Labels: `type:bug`, `area:oracle-rri`, `area:ci`, `priority:P0`.
Acceptance: issue #7 is enriched, no modules hand-roll inconsistent device parsing, CPU/MPS/CUDA behavior is explicit, and Streamlit/CLI/cache/renderer/Lightning use the same device path. Existing #7 is currently a one-line MPS/global-device issue, so update it rather than duplicating it. ([GitHub][7])

**ARIA-GH-013 — Add pose-frame/CW90 consistency guard for VIN, rendering, and cached batches**
Priority: P0. Labels: `type:bug`, `area:geometry`, `area:vin`, `area:oracle-rri`, `priority:P0`.
Acceptance: cached VIN batches pass pose/camera alignment checks; `apply_cw90_correction` cannot silently desynchronize poses and PyTorch3D cameras; regression tests protect the convention.

**ARIA-GH-014 — Fix candidate/RRI shuffling and known VIN batch shuffle failure**
Priority: P0. Labels: `type:bug`, `area:vin`, `area:data`, `priority:P0`.
Acceptance: known shuffle test passes; candidate pose, RRI, camera, metadata, and validity masks remain aligned after shuffle. PR #6 already notes a broader VIN-core shuffle failure outside its touched surface. ([GitHub][8])

**ARIA-GH-015 — Add offline cache config-hash metadata, manifest summary, and filtering**
Priority: P0. Labels: `type:enhancement`, `area:data`, `area:reproducibility`, `priority:P0`.
Acceptance: cache loader can report/filter config-hash groups; mixed-config samples remain usable as augmentation; manifest CLI exists.

**ARIA-GH-016 — Add online/extended offline sample-generation mode to training/datamodule**
Priority: P1. Labels: `type:enhancement`, `area:data`, `area:oracle-rri`, `priority:P1`.
Acceptance: datamodule supports offline-only and offline+online modes; generated samples can be persisted with config metadata; validation split semantics remain clear.

**ARIA-GH-017 — Make OracleRriLabeler safe with DataLoader `num_workers > 0`**
Priority: P0. Labels: `type:bug`, `area:data`, `area:oracle-rri`, `priority:P0`.
Acceptance: tiny oracle/cache dataloader works with `num_workers=0` and `num_workers=2`, or unsupported GPU multiprocessing fails with a clear reason.

**ARIA-GH-018 — Profile oracle throughput and define scaling budget**
Priority: P0. Labels: `type:experiment`, `area:oracle-rri`, `area:evaluation`, `priority:P0`.
Acceptance: benchmark script/table exists for candidate generation, mesh preprocessing, rendering, backprojection, Chamfer, and cache writing; roadmap has snippets/day and GB/snippet budget.

**ARIA-GH-019 — Protect fine-detail supervision: mesh/point-cloud downsampling policy and ablation**
Priority: P1. Labels: `type:experiment`, `area:oracle-rri`, `area:evaluation`, `priority:P1`.
Acceptance: resolution policy exists; simplification sensitivity table exists; banned settings cannot silently enter thesis cache generation.

**ARIA-GH-020 — Run candidate-generation realism/generalization experiments**
Priority: P1. Labels: `type:experiment`, `area:geometry`, `area:oracle-rri`, `priority:P1`.
Acceptance: report recommends default shell/candidate settings; candidate settings are versioned in configs/cache metadata; invalid-candidate strategy is decided.

**ARIA-GH-021 — Document RRI/Chamfer formulas in Typst slides and theory docs**
Priority: P1. Labels: `type:docs`, `area:oracle-rri`, `area:thesis`, `priority:P1`.
Acceptance: accuracy, completeness, point↔mesh Chamfer, and RRI formulas are in slides/paper/docs and match implementation.

### M3 — VIN scorer baseline & controlled ablations

**ARIA-GH-022 — Add torchmetrics and richer Lightning validation metrics**
Priority: P0. Labels: `type:enhancement`, `area:vin`, `area:evaluation`, `priority:P0`.
Acceptance: validation logs ranking, ordinal accuracy, calibration, confusion, label histogram, RRI bias/variance, and candidate validity metrics.

**ARIA-GH-023 — Fix predicted-RRI reporting semantics and calibration plots**
Priority: P0. Labels: `type:bug`, `area:vin`, `area:evaluation`, `priority:P0`.
Acceptance: no plot compares normalized ordinal expected value to metric RRI without conversion; calibration reports separate bin calibration from metric-RRI calibration.

**ARIA-GH-024 — Run controlled VIN ablation matrix with fixed seeds/splits/schedule**
Priority: P0. Labels: `type:experiment`, `area:vin`, `area:evaluation`, `priority:P0`.
Acceptance: controlled ablation table explains which VIN components to keep/drop; no claims rely on confounded cross-run comparisons.

**ARIA-GH-025 — Evaluate learnable CORAL bin centers/shifts and imbalance handling**
Priority: P1. Labels: `type:experiment`, `area:vin`, `priority:P1`.
Acceptance: fixed vs learnable bin representatives and plain/balanced/focal CORAL are compared; monotonicity and calibration are reported.

**ARIA-GH-026 — Test stage-aware features or stage-aware binning**
Priority: P1. Labels: `type:experiment`, `area:vin`, `area:data`, `priority:P1`.
Acceptance: per-stage Spearman/top-k/calibration report exists; chosen stage strategy is encoded in config/docs.

**ARIA-GH-027 — Prototype candidate-relative positional encoding and query-centric fusion**
Priority: P1. Labels: `type:experiment`, `area:vin`, `area:geometry`, `priority:P1`.
Acceptance: candidate-relative encoding is compared under the controlled baseline; frame conventions are documented and tested.

**ARIA-GH-028 — Add semidense reliability and candidate-visibility embeddings**
Priority: P1. Labels: `type:enhancement`, `area:vin`, `area:data`, `priority:P1`.
Acceptance: semidense reliability/visibility features are present in VIN batch contracts; ablation shows whether they improve ranking/calibration.

**ARIA-GH-029 — Prototype RGB/DINOv2 or EVL neck feature projection for candidate views**
Priority: P2. Labels: `type:experiment`, `area:vin`, `area:representation`, `priority:P2`.
Acceptance: prototype either demonstrates measurable value or is explicitly deferred; no thesis-core path depends on it unless it beats the core baseline.

**ARIA-GH-030 — Harden EVL checkpoint/config loading options**
Priority: P1. Labels: `type:enhancement`, `area:vin`, `area:reproducibility`, `priority:P1`.
Acceptance: known checkpoint variants are supported or fail clearly; resolved config captures backbone loading behavior.

### M4 — Non-myopic planning baselines

**ARIA-GH-031 — Define MDP contract for counterfactual non-myopic planning**
Priority: P0. Labels: `type:decision`, `area:planning`, `area:rl`, `priority:P0`.
Acceptance: state/action/transition/reward/termination/invalid-action contract exists; historical-only vs counterfactual vs privileged modalities are explicit.

**ARIA-GH-032 — Implement one-step greedy, close-greedy, and beam-search rollout evaluator**
Priority: P0. Labels: `type:enhancement`, `area:planning`, `area:evaluation`, `priority:P0`.
Acceptance: one script/config compares one-step, close-greedy, and beam search on the same snippets; outputs cumulative RRI, final Chamfer, path length, invalid counts, and compute cost.

**ARIA-GH-033 — Harden multi-step oracle RRI pipeline and cumulative-RRI plots**
Priority: P0. Labels: `type:enhancement`, `area:oracle-rri`, `area:planning`, `priority:P0`.
Acceptance: multi-step oracle evaluator is deterministic; synthetic tests validate cumulative metrics; Streamlit/report can inspect rollouts step-by-step.

**ARIA-GH-034 — Harden discrete-shell Gymnasium/SB3 RL baseline**
Priority: P1. Labels: `type:experiment`, `area:rl`, `area:planning`, `priority:P1`.
Acceptance: environment passes Gymnasium checks where applicable; baseline run is reproducible; report states whether RL is competitive against search.

**ARIA-GH-035 — Use VIN as fast surrogate reward or critic for planning**
Priority: P1. Labels: `type:experiment`, `area:vin`, `area:planning`, `area:rl`, `priority:P1`.
Acceptance: surrogate reward error and rollout degradation are quantified; decision exists whether VIN is usable for planning now or remains one-step-only.

**ARIA-GH-036 — Prototype continuous pose proposal with feasibility projection**
Priority: P2. Labels: `type:experiment`, `area:planning`, `area:geometry`, `status:stretch`, `priority:P2`.
Acceptance: prototype runs without changing core evaluator; feasibility projection is deterministic; continuous control is either promoted or deferred.

### M5 — Scale, entity-aware extension, and reports

**ARIA-GH-037 — Scale oracle cache coverage within the 100 GT-mesh ASE scenes**
Priority: P0. Labels: `type:experiment`, `area:data`, `area:oracle-rri`, `priority:P0`.
Acceptance: expanded cache manifest exists; training configs use it; paper/report states exact cache coverage, split, and candidate count.

**ARIA-GH-038 — Add multi-anchor and multi-candidate-set augmentation**
Priority: P1. Labels: `type:experiment`, `area:data`, `area:oracle-rri`, `priority:P1`.
Acceptance: augmented cache entries are identifiable/filterable; ablation compares baseline vs augmented cache.

**ARIA-GH-039 — Prototype entity-aware RRI using OBB-cropped mesh/points**
Priority: P1. Labels: `type:experiment`, `area:oracle-rri`, `area:vin`, `priority:P1`.
Acceptance: entity-aware labels are computed for at least one scene/entity subset; metric definition and limitations are documented.

**ARIA-GH-040 — Build W&B/Optuna report generator for final experiment tables**
Priority: P0. Labels: `type:infra`, `area:evaluation`, `area:thesis`, `priority:P0`.
Acceptance: one command regenerates final experiment tables; tables include run IDs, configs, seeds, data split/hash, and figure outputs.

**ARIA-GH-041 — Write sim-to-real, device, and human-in-the-loop feasibility memo**
Priority: P2. Labels: `type:docs`, `area:thesis`, `status:stretch`, `priority:P2`.
Acceptance: future-work section is grounded in actual primitives; no deployment claim is made without implementation evidence.

### M6 — Thesis freeze & final delivery

**ARIA-GH-042 — Freeze thesis narrative and synchronize README, Quarto, Typst paper, and slides**
Priority: P0. Labels: `type:docs`, `area:thesis`, `area:docs`, `priority:P0`.
Acceptance: public docs agree on contributions and limitations; paper compiles with final tables/figures; slides reflect evidence rather than aspirational scope.

**ARIA-GH-043 — Create reproducible final release package and smoke matrix**
Priority: P0. Labels: `type:infra`, `area:reproducibility`, `area:thesis`, `priority:P0`.
Acceptance: clean checkout can reproduce smoke results; final configs/run IDs are versioned; release notes list unsupported paths and known limitations.

**ARIA-GH-044 — Prepare final defense slides and demo storyboard**
Priority: P0. Labels: `type:docs`, `area:thesis`, `area:evaluation`, `priority:P0`.
Acceptance: final slides compile and fit allotted time; demo has live and static fallback; every major claim maps to a result.

## Creation order

1. Create **ARIA-GH-001 → ARIA-GH-003** manually first.
2. Update existing **issue #7** with **ARIA-GH-012** rather than opening a duplicate.
3. Create all remaining **P0** issues.
4. Create P1/P2 issues milestone-by-milestone.
5. Use the zip pack only after the milestones/labels exist, because the script assumes they are already present.

[1]: https://github.com/JanDuchscherer104/ARIA-NBV/issues "Issues · JanDuchscherer104/ARIA-NBV · GitHub"
[2]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.agents/memory/state/PROJECT_STATE.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/README.md "raw.githubusercontent.com"
[4]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/contents "ARIA-NBV/docs/contents at main · JanDuchscherer104/ARIA-NBV · GitHub"
[5]: https://github.com/JanDuchscherer104/ARIA-NBV/pull/8 "[codex] align docs to current state and prune stale pages by JanDuchscherer104 · Pull Request #8 · JanDuchscherer104/ARIA-NBV · GitHub"
[6]: https://github.com/Yeachan-Heo/oh-my-codex?utm_source=chatgpt.com "Yeachan-Heo/oh-my-codex: OmX"
[7]: https://github.com/JanDuchscherer104/ARIA-NBV/issues/7 "Make the (mutli-step) oracle_rri pipeline mps-compatible. · Issue #7 · JanDuchscherer104/ARIA-NBV · GitHub"
[8]: https://github.com/JanDuchscherer104/ARIA-NBV/pull/6 "[codex] Deduplicate aria_nbv helper surfaces by JanDuchscherer104 · Pull Request #6 · JanDuchscherer104/ARIA-NBV · GitHub"


[User-Turn05]
Please employ ruthless $simplification and identify all slop and targets for decluttering, removing noise in both the docs and the aria_nbv package!

[Agent-Turn06]
## Ruthless simplification stance

Treat ARIA-NBV as **three products**, not one big memory dump:

1. **Public thesis/project surface**: README, setup, paper, final docs, final experiments.
2. **Runtime/research package**: oracle RRI, candidate generation, rendering/backprojection, VIN scorer, training/evaluation.
3. **Internal research/agent archive**: todos, generated context, scaffold notes, old sweeps, external-code maps, experiment history.

Everything that does not clearly serve one of those three should be **deleted, moved out of public view, or merged into a smaller canonical file**.

Do **not** declutter away the scientific spine. The keep-core is: ASE/SceneScript as the egocentric synthetic-data substrate, GenNBV as the coverage/RL contrast, VIN-NBV as the RRI-quality objective, and Aria-NBV’s current contribution as oracle RRI supervision plus a preliminary VIN scorer, not a finished end-to-end NBV policy.

---

# P0 kill list

These are the highest-noise, highest-payoff cleanup targets.

| Target                                | Ruthless action                                                                                                                               | Why it is slop                                                                                                                                                                                                                                                                                                      | Verification                                                                                                                                     |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `.logs/`                              | **Delete from git. Add to `.gitignore`. Replace with artifact manifest.**                                                                     | The repo visibly tracks `.logs/checkpoints` and `.logs/ckpts`, including Lightning checkpoints and pretrained/model weights such as DINOv2, PointNeXt, SceneScript, and model-lite artifacts. Those are not source code. They bloat the repo and confuse reproducibility. ([GitHub][1]) ([GitHub][2]) ([GitHub][3]) | `git ls-files .logs` returns nothing. README/SETUP links to download/cache instructions instead.                                                 |
| Public docs nav                       | **Remove TODOs, Project State, Questions, Agent Scaffolding, generated scaffold pages, and external implementation indices from public nav.** | The live site exposes TODOs, Project State, Agent Scaffolding, implementation internals, external implementation maps, and many generated/internal pages as if they were public thesis content. That is the main docs slop. ([Janduchscherer104][4])                                                                | Public site has a small thesis-facing sidebar only. Internal pages live in `.agents/` or hidden docs.                                            |
| `docs/_quarto.yml`                    | **Rewrite and reformat from scratch.**                                                                                                        | The Quarto config is effectively one huge physical line, renders nearly everything under `contents`, exposes agent scaffold pages, and still contains old `NBV` repo links rather than `ARIA-NBV`. ([GitHub][5])                                                                                                    | YAML is readable, public nav is intentional, repo links point to `ARIA-NBV`, render passes.                                                      |
| Root `README.md`                      | **Rewrite as a real project front door.**                                                                                                     | Current README is a tiny command dump: seminar title, GitHub Pages link, and CLI/cache snippets. GitHub About also has no description/topics. ([GitHub][6]) ([GitHub][7])                                                                                                                                           | README explains mission, status, quick start, data/cache contract, known limits, and docs map.                                                   |
| `aria_nbv.data`                       | **Delete or finish migration.**                                                                                                               | The package itself says `aria_nbv.data` is deprecated and canonical work belongs in `aria_nbv.data_handling`, but `pyproject.toml` still exposes `nbv-downloader = aria_nbv.data.downloader:main`. ([GitHub][8]) ([GitHub][9])                                                                                      | No runtime CLI imports from deprecated package. Either no `aria_nbv.data`, or it is a tiny documented compatibility shim with deprecation tests. |
| `aria_nbv.data_handling` legacy layer | **Complete the cutover. Delete or quarantine legacy wrappers.**                                                                               | `data_handling` contains many `_legacy_*` modules, thin compatibility wrappers, old direct-import surfaces, and even hard-coded old `/home/jandu/repos/NBV/...` paths in docs. This is the largest package-level noise source. ([GitHub][10])                                                                       | One canonical cache API; no hard-coded old repo paths; no broad legacy imports in new code.                                                      |
| `pyproject.toml`                      | **Reformat and split dependencies into extras.**                                                                                              | It is effectively one giant line and mixes core package, training, W&B, Streamlit, Open3D, notebooks, external backbones, local path deps, and package scripts together. ([GitHub][9])                                                                                                                              | `core`, `train`, `viz`, `dev`, `notebooks`, `efm`, `all` extras exist; base import does not pull the whole research stack.                       |
| `vin/model_v3.py`                     | **Split and remove experiment-history sludge.**                                                                                               | The core VIN model carries long research-history docstrings, stale v2/sweep commentary, an inline TODO, deprecated config flags, and mixed responsibilities. ([GitHub][11])                                                                                                                                         | `model_v3.py` becomes a thin composition file; feature branches, heads, frame checks, and diagnostics are separated.                             |

---

# Docs slop audit

## Public docs should be reduced to this

Target public docs shape:

```text
docs/
  index.qmd                         # project landing page
  setup.qmd                         # reproducibility and data setup
  contents/
    theory/
      rri_theory.qmd                # RRI, Chamfer, ordinal labels
    implementation/
      oracle_rri_pipeline.qmd       # candidate generation, rendering, labels
      vin_scorer.qmd                # VIN architecture + training
      evaluation_protocol.qmd       # metrics, splits, cache, reporting
    literature/
      scene_script.qmd
      efm3d.qmd
      vin_nbv.qmd
      gen_nbv.qmd
  reference/                        # optional generated API docs
  typst/                            # paper/slides
```

Everything else is either internal, generated, historical, or stretch.

---

## Docs: delete / move / merge list

| Target                                                                                | Action                                                                                                                       | Rationale                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docs/contents/todos.qmd`                                                             | **Move to `.agents/memory/state/TODOS.md` or delete after GH issues exist.**                                                 | Public TODOs are not thesis content. They leak work-management noise into the project site. The current public site exposes TODOs directly. ([Janduchscherer104][4])                                          |
| `docs/contents/roadmap.qmd`                                                           | **Replace with a short public roadmap only if needed; detailed roadmap belongs in GH milestones/issues.**                    | Roadmap as docs becomes stale. Use GitHub issues/milestones for execution.                                                                                                                                    |
| `docs/contents/questions.qmd`                                                         | **Move to `.agents/memory/state/OPEN_QUESTIONS.md`.**                                                                        | Advisor/research questions can exist, but not as top-level public nav noise unless curated.                                                                                                                   |
| `docs/contents/ideas.qmd`                                                             | **Move to `.agents/memory/state/IDEAS.md` or `.agents/history/ideas/`.**                                                     | Raw idea lists are useful for agents, bad as public docs.                                                                                                                                                     |
| `docs/contents/resources/agent_scaffold/`                                             | **Remove from public docs nav. Keep generated mirror ignored or move to `.agents/generated/docs/`.**                         | The Quarto nav references a large Agent Scaffolding subtree, but the tracked folder currently only contains `.gitignore`, so the public docs depend on generated internal pages. ([GitHub][12]) ([GitHub][5]) |
| `docs/_generated/context/`                                                            | **Keep out of public docs. Move to `.agents/generated/context/` or leave ignored.**                                          | Root `AGENTS.md` points agents to generated context under docs, but generated context should not be part of the public documentation tree. ([GitHub][13])                                                     |
| `docs/_freeze/`                                                                       | **Delete unless a specific Quarto executable page needs frozen output. Otherwise ignore/regenerate.**                        | Frozen render artifacts are not core project documentation. The docs folder currently contains `_freeze/contents`. ([GitHub][14])                                                                             |
| `docs/contents/ext-impl/`                                                             | **Move to `.agents/references/external_implementations/`; optionally keep one curated public “external dependencies” page.** | ATEK/EFM3D/Project Aria implementation maps are useful internal references, not public thesis docs. ([GitHub][15])                                                                                            |
| `docs/literature/tex-src/`                                                            | **Move to `.agents/references/literature_raw/` or delete if reproducible from PDFs/BibTeX.**                                 | Raw TeX source inside docs is archival/reference material, not site content. `docs/literature/tex-src` is visible in the docs tree. ([GitHub][14])                                                            |
| `docs/contents/impl/data_pipeline_overview.quarto_ipynb`                              | **Delete or ignore generated notebook artifact.**                                                                            | `.quarto_ipynb` is generated execution debris unless intentionally maintained.                                                                                                                                |
| `docs/contents/impl/coral_intergarion.qmd`                                            | **Rename or merge.**                                                                                                         | Filename typo signals low-trust docs. Merge into `vin_scorer.qmd` unless it has unique content. ([GitHub][16])                                                                                                |
| `docs/contents/impl/vin_v2_feature_proposals.qmd`                                     | **Move to `.agents/history/vin_v2/` or delete.**                                                                             | VIN v2 proposals are historical, not current docs.                                                                                                                                                            |
| `docs/contents/impl/vin_v2_component_diagrams.qmd`                                    | **Move to history or delete.**                                                                                               | Same reason: stale v2 implementation material pollutes current architecture.                                                                                                                                  |
| `docs/contents/impl/optuna_vin_v2_searchspace_2026-01-07.qmd`                         | **Move to experiment archive.**                                                                                              | Dated sweep details belong in `.agents/history/experiments/` or W&B/Optuna reports, not public implementation docs.                                                                                           |
| `docs/contents/impl/vin_coverage_aware_training.qmd`                                  | **Merge into `vin_scorer.qmd` if still current; otherwise archive.**                                                         | One VIN training page should own current training objective and diagnostics.                                                                                                                                  |
| `docs/contents/impl/aria_nbv_overview.qmd`, `aria_nbv_package.qmd`, `overview.qmd`    | **Merge into one implementation overview.**                                                                                  | Three overview pages create duplicate explanations and drift. ([GitHub][16])                                                                                                                                  |
| `docs/contents/impl/oracle_rri_impl.qmd`, `rri_computation.qmd`, `oracle_rri.mermaid` | **Merge into one `oracle_rri_pipeline.qmd`.**                                                                                | RRI math, implementation, and diagram should be one canonical page.                                                                                                                                           |
| `docs/contents/repo_structure.qmd`                                                    | **Delete or move to `.agents/references/repo_structure.md`.**                                                                | Repo maps are agent/operator material, not public docs.                                                                                                                                                       |
| `docs/contents/diagrams.qmd`                                                          | **Keep only if diagrams are final. Otherwise merge diagrams into owning pages.**                                             | A generic diagrams page becomes a junk drawer.                                                                                                                                                                |
| `docs/contents/resources.qmd`                                                         | **Reduce to one curated resources page or delete.**                                                                          | “Resources” is too broad and currently routes into internal scaffold material.                                                                                                                                |
| `docs/contents/literature/hesita.qmd`                                                 | **Keep only if it appears in thesis narrative. Otherwise archive.**                                                          | Literature pages should be limited to papers directly supporting the argument. Current literature folder includes SceneScript, GenNBV, VIN-NBV, EFM3D, and HESITA. ([GitHub][17])                             |
| Quarto `code-tools` / source exposure                                                 | **Disable for public pages unless intentionally useful.**                                                                    | The live site exposes source/code-oriented affordances and internal-looking generated content, which makes the public surface feel like a dev dump. ([Janduchscherer104][4])                                  |

---

## Public docs should not expose this

Remove from navbar/sidebar:

```text
TODOs
Project State
Questions
Ideas
Agent Scaffolding
Generated context
External implementation symbol indices
Raw repo structure
Old VIN v2 proposals
Optuna dated sweep notes
```

Keep public:

```text
Home
Setup / Reproducibility
Theory: RRI + Chamfer + ordinal labels
Oracle RRI pipeline
VIN scorer
Evaluation protocol
Literature: SceneScript, EFM3D, VIN-NBV, GenNBV
API reference, if generated cleanly
Paper / slides
```

The public site currently advertises “TODOs,” “Project State,” “Resources → Agent Scaffolding,” and “External Implementations,” which is exactly the bloat to remove. ([Janduchscherer104][4])

---

# Package slop audit

## 1. `aria_nbv.data`: deprecated but still alive

Current state: `aria_nbv.data` says it is deprecated and that canonical ownership moved to `aria_nbv.data_handling`, but it still contains downloader/metadata utilities and is still used by a CLI entry point. ([GitHub][8]) ([GitHub][9])

Ruthless target:

```text
aria_nbv/data/                  DELETE or reduce to compatibility-only
  downloader.py                 MOVE to aria_nbv/data_handling/download.py
  metadata.py                   MOVE to aria_nbv/data_handling/metadata.py
  download_stats.py             MOVE or delete
```

Acceptance criteria:

```text
[ ] `nbv-downloader` imports from canonical package path.
[ ] No new code imports `aria_nbv.data`.
[ ] Deprecated package either gone or has one tiny warning-only shim.
[ ] Tests cover old CLI behavior if compatibility remains.
```

---

## 2. `data_handling`: migration never finished

This is the largest package-level cleanup target. The folder contains canonical internals like `_offline_dataset.py`, `_offline_store.py`, `_vin_runtime.py`, and `_vin_sources.py`, but also a large batch of `_legacy_*` files and thin compatibility wrappers. Its README explicitly describes old direct imports that still resolve, legacy modules, and a migration marker with hard-coded old repo paths. ([GitHub][10])

### Ruthless move

Collapse to this:

```text
aria_nbv/data_handling/
  __init__.py                    # tiny exports only
  snippets.py                    # EFM/ASE snippet loading
  cache_store.py                 # immutable offline cache store
  cache_manifest.py              # config hash, coverage, filtering
  vin_batch.py                   # VIN batch construction
  mesh_cache.py                  # mesh preprocessing/cache
  download.py                    # ASE download utilities
  metadata.py                    # scene metadata
  migration.py                   # temporary, deleted after cutover
  legacy/                        # optional temporary quarantine
```

Delete or quarantine:

```text
_legacy_cache_api.py
_legacy_dataset_mixins.py
_legacy_offline_cache_coverage.py
_legacy_offline_cache_serialization.py
_legacy_offline_cache_store.py
_legacy_oracle_cache.py
_legacy_vin_cache.py
_legacy_vin_provider.py
_legacy_vin_source.py
oracle_cache.py
vin_cache.py
vin_provider.py
offline_cache_coverage.py
offline_cache_serialization.py
offline_cache_store.py
```

The compatibility wrappers are only acceptable for one release window. After that, delete them.

Acceptance criteria:

```text
[ ] No `/home/jandu/repos/NBV` or old `NBV` path appears anywhere.
[ ] `rg "_legacy|NBV_LEGACY|legacy"` shows only quarantined files or nothing.
[ ] Public imports are documented in one place.
[ ] Old wrapper tests either removed or explicitly marked compatibility.
```

---

## 3. `data_handling/__init__.py`: package-root magic

The file uses a large lazy-export registry in a single physical line and advertises legacy cache/VIN surfaces. ([GitHub][18])

Ruthless target:

```python
"""Data loading, cache, and VIN batch contracts."""

from aria_nbv.data_handling.cache_store import OfflineCacheStore
from aria_nbv.data_handling.vin_batch import VinBatch
from aria_nbv.data_handling.snippets import EfmSnippetLoader

__all__ = [
    "OfflineCacheStore",
    "VinBatch",
    "EfmSnippetLoader",
]
```

No giant dynamic export maps. No “everything importable from root.” Force explicit imports.

---

## 4. `vin/model_v3.py`: architecture file mixed with history, TODOs, and stale config

The VIN v3 file currently combines model definition, historical comments about v2 sweeps, long architecture notes, deprecated config fields, inline TODOs, frame corrections, projection logic, and output decoding. ([GitHub][11])

Ruthless split:

```text
aria_nbv/vin/
  model_v3.py                    # thin composition only
  config.py                      # VinV3Config, no stale/deprecated flags
  frames.py                      # CW90 / pose consistency guards
  scene_field.py                 # EVL voxel field assembly
  pose_features.py               # pose encoders
  semidense_features.py          # semidense projection stats/grid CNN
  voxel_projection.py            # voxel projection FiLM
  heads.py                       # CORAL / scoring head
  diagnostics.py                 # summarize/plot helper hooks
```

Delete or move out of code:

```text
- Sweep narrative
- “VIN v2 showed...” history
- Inline research TODOs
- Deprecated flags like use_voxel_valid_frac_gate
- Paper-style architecture prose
```

Move that material to:

```text
docs/contents/implementation/vin_scorer.qmd
.agents/history/experiments/vin_v2.md
.agents/memory/state/OPEN_QUESTIONS.md
```

Acceptance criteria:

```text
[ ] Core model file is under ~250 lines.
[ ] No TODO in model code unless it is attached to a GitHub issue.
[ ] No deprecated config fields remain in active config.
[ ] CW90 consistency is tested in a dedicated frame test.
```

---

## 5. `vin/experimental`: promote or delete

The `vin` package contains an `experimental` directory. ([GitHub][19])

Ruthless rule:

```text
If an experiment is used by current training configs:
  promote it to a named module and test it.
Else:
  move it to .agents/history/vin_experiments/ or delete.
```

Package code should not carry half-alive experimental branches.

---

## 6. `rri_metrics/context_pytorch_3d_losses.md`: markdown inside runtime package

`rri_metrics` contains a markdown context file next to Python modules. ([GitHub][20])

Action:

```text
MOVE:
  aria_nbv/aria_nbv/rri_metrics/context_pytorch_3d_losses.md

TO ONE OF:
  docs/contents/implementation/pytorch3d_losses.qmd
  .agents/references/pytorch3d_losses.md
```

Runtime package directories should contain source code, not context dumps.

---

## 7. `rendering`: duplicated renderer/backprojection concepts

Current rendering files:

```text
candidate_depth_renderer.py
candidate_pointclouds.py
efm3d_depth_renderer.py
pytorch3d_depth_renderer.py
unproject.py
plotting.py
```

This suggests parallel abstractions that should be collapsed. ([GitHub][21])

Ruthless target:

```text
aria_nbv/rendering/
  depth_renderer.py              # interface / protocol
  pytorch3d_renderer.py          # one implementation
  backproject.py                 # one NDC-aligned backprojection implementation
  candidate_points.py            # candidate depth -> point cloud
  diagnostics.py                 # plotting only
```

Acceptance criteria:

```text
[ ] Only one backprojection path.
[ ] Only one PyTorch3D camera conversion path.
[ ] Tests verify render → backproject points land on mesh.
[ ] Candidate depth and point cloud code share camera convention utilities.
```

---

## 8. `utils`: junk drawer

Current `utils` contains:

```text
base_config.py
console.py
data_plotting.py
frames.py
grad_norms.py
optuna_optimizable.py
plotting.py
reporting.py
rich_summary.py
schemas.py
summary.py
typed_payloads.py
viz_utils.py
wandb_utils.py
```

This is a classic dumping ground. ([GitHub][22])

Ruthless merge plan:

```text
plotting.py + data_plotting.py + viz_utils.py
  -> diagnostics/plotting.py or app/plotting.py

summary.py + rich_summary.py + reporting.py
  -> reporting/summary.py

schemas.py + typed_payloads.py
  -> contracts.py or types.py

wandb_utils.py + optuna_optimizable.py
  -> experiments/
```

Keep only truly generic utilities in `utils`:

```text
console.py
frames.py
grad_norms.py
```

Everything else should move to a domain package.

---

## 9. `interpretability`: likely stretch-code noise

`interpretability` currently contains only `__init__.py` and `attribution.py`. ([GitHub][23])

Decision rule:

```text
If attribution appears in final paper/experiments:
  keep and add tests/docs.
Else:
  move to .agents/history/stretch/interpretability/ or delete.
```

Tiny unused packages are worse than absent packages because they suggest supported functionality.

---

## 10. `app` / Streamlit: keep, but quarantine UI logic

The app package contains controller/state/panels/UI code. ([GitHub][24])

Keep it, but enforce:

```text
app/ may visualize and orchestrate.
app/ may not own geometry, RRI math, cache contracts, or VIN feature logic.
```

If any oracle/RRI/VIN logic lives in Streamlit panels, move it back into the owning runtime modules.

---

## 11. `external/`: document or reduce

The repo contains external checkouts for ATEK, PointNeXt, EFM3D, openpoints shim, Project Aria tools, and SceneScript. ([GitHub][25])

Ruthless rule:

```text
For each external dependency:
  [ ] Is it used by current code?
  [ ] Is it pinned?
  [ ] Is it required for smoke tests?
  [ ] Is it optional?
```

Likely actions:

| External            | Action                                                                |
| ------------------- | --------------------------------------------------------------------- |
| `efm3d`             | Keep if VIN scorer uses EVL backbone. Pin and document.               |
| `ATEK`              | Keep if required for dataset format. Pin and document.                |
| `projectaria_tools` | Keep only if local checkout is needed; otherwise use package install. |
| `PointNeXt`         | Delete/quarantine unless active ablation uses it.                     |
| `scenescript`       | Delete/quarantine unless active comparison uses it.                   |
| `openpoints_shim`   | Keep only if PointNeXt remains.                                       |

---

## 12. `scripts/`: agent scripts mixed into project root

Current `scripts` are mostly context/docs/agent utilities: context index, literature index/search, QMD outline, agent-doc generation, API-doc generation, agent memory validation. ([GitHub][26])

Ruthless target:

```text
scripts/
  build_docs.sh
  generate_api_docs.sh
  validate_agent_memory.py

.agents/tools/
  nbv_context_index.sh
  nbv_get_context.sh
  nbv_literature_index.sh
  nbv_literature_search.sh
  nbv_qmd_outline.sh
  quarto_generate_agent_docs.py
```

Public repo scripts should be reproducibility/build scripts, not agent-internal tooling.

---

# Root scaffold slop

## `AGENTS.md`

Root `AGENTS.md` is too dense. It points to generated docs context, many references, commands, retrieve-on-demand docs, paper state, caveats, and debrief workflow all in one file. ([GitHub][13])

Ruthless target:

```text
Root AGENTS.md:
  - 10–20 lines max
  - project mission
  - canonical truth order
  - package/doc routing
  - `make ci`
  - where to find memory/state
```

Move the rest to:

```text
.agents/skills/repo-context-router/SKILL.md
.agents/references/coordinate_frames.md
.agents/references/operator_quick_reference.md
.agents/memory/state/PROJECT_STATE.md
```

The nested package `AGENTS.md` is much better and should mostly be kept because it contains concrete package rules, including the `data_handling` vs `data` boundary, config-as-factory convention, frame semantics, and anti-patterns. ([GitHub][27])

---

# Formatting slop

Several important files appear effectively collapsed into huge physical lines in raw GitHub views:

```text
README.md
docs/_quarto.yml
aria_nbv/pyproject.toml
aria_nbv/aria_nbv/data_handling/__init__.py
```

That makes diffs, reviews, and agent edits worse. It also hides accidental bloat. ([GitHub][6]) ([GitHub][5]) ([GitHub][9]) ([GitHub][18])

Ruthless action:

```text
[ ] Reformat Markdown into normal paragraphs/lists.
[ ] Reformat YAML/TOML into readable blocks.
[ ] Replace giant import/export registries with explicit modules.
[ ] Add pre-commit checks for formatting.
```

---

# Things to keep

Do **not** simplify these away; they are core or useful safety net.

| Keep                                                                                        | Why                                                                                                                                             |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `aria_nbv/aria_nbv/pipelines/oracle_rri_labeler.py`                                         | Core oracle supervision pipeline.                                                                                                               |
| `aria_nbv/aria_nbv/pose_generation/`                                                        | Candidate generation is core to oracle labels and planning. ([GitHub][28])                                                                      |
| `aria_nbv/aria_nbv/rri_metrics/{metrics.py,oracle_rri.py,rri_binning.py,coral.py,types.py}` | RRI, Chamfer, binning, and ordinal labels are thesis core. ([GitHub][20])                                                                       |
| `aria_nbv/aria_nbv/lightning/`                                                              | Training orchestration is useful if cleaned and kept separate from model code. ([GitHub][29])                                                   |
| `aria_nbv/aria_nbv/app/`                                                                    | Streamlit diagnostics are useful, but should stay UI-only.                                                                                      |
| `tests/`                                                                                    | The test tree is large and valuable for safe deletion/refactoring. Use it as the deletion guard. ([GitHub][30])                                 |
| `docs/typst/`                                                                               | Paper/slides should be canonical thesis artifacts. Docs guidance already says the Typst paper is the highest-level ground truth. ([GitHub][31]) |
| Literature pages for SceneScript, EFM3D, VIN-NBV, GenNBV                                    | These directly support the thesis narrative.                                                                                                    |

---

# Concrete GH issues to open for decluttering

## Issue 1 — Delete tracked binary/model artifacts from `.logs`

**Priority:** P0
**Area:** repo hygiene, reproducibility

**Scope**

Delete tracked training checkpoints and pretrained weights under `.logs/`. Add an artifact manifest and download/cache instructions.

**Acceptance**

```text
[ ] `git ls-files .logs` returns empty.
[ ] `.logs/` is ignored.
[ ] SETUP.md documents how to obtain required weights/checkpoints.
[ ] No source import assumes `.logs/ckpts` exists.
```

---

## Issue 2 — Rebuild public docs nav and remove internal/agent pages

**Priority:** P0
**Area:** docs

**Scope**

Rewrite `docs/_quarto.yml`; remove TODOs, project state, questions, generated context, agent scaffold, and external implementation maps from public nav.

**Acceptance**

```text
[ ] Public nav has only Home, Setup, Theory, Oracle RRI, VIN scorer, Evaluation, Literature, API/Paper.
[ ] Agent scaffold pages are not publicly linked.
[ ] `docs/_quarto.yml` is readable multi-line YAML.
[ ] All repo links point to ARIA-NBV, not old NBV.
[ ] `cd docs && quarto render` passes.
```

---

## Issue 3 — Collapse implementation docs into canonical pages

**Priority:** P0
**Area:** docs

**Scope**

Merge duplicate implementation pages into:

```text
oracle_rri_pipeline.qmd
vin_scorer.qmd
evaluation_protocol.qmd
implementation_overview.qmd
```

**Delete/archive**

```text
coral_intergarion.qmd
data_pipeline_overview.quarto_ipynb
vin_v2_feature_proposals.qmd
vin_v2_component_diagrams.qmd
optuna_vin_v2_searchspace_2026-01-07.qmd
```

**Acceptance**

```text
[ ] No stale VIN v2 pages in public docs.
[ ] No generated `.quarto_ipynb` tracked.
[ ] Current paper/docs/README agree on system scope.
```

---

## Issue 4 — Move public work-management docs into `.agents`

**Priority:** P0
**Area:** docs, agents

**Scope**

Move or delete:

```text
todos.qmd
roadmap.qmd
questions.qmd
ideas.qmd
repo_structure.qmd
```

**Acceptance**

```text
[ ] Public docs contain no raw task lists.
[ ] GH milestones/issues own execution roadmap.
[ ] `.agents/memory/state/` owns internal state.
```

---

## Issue 5 — Finish `aria_nbv.data` migration

**Priority:** P0
**Area:** package, data

**Scope**

Move downloader/metadata utilities into canonical `data_handling` or a new `datasets` module. Remove or shrink deprecated `aria_nbv.data`.

**Acceptance**

```text
[ ] `nbv-downloader` imports from canonical module.
[ ] `aria_nbv.data` is gone or compatibility-only.
[ ] No new code imports `aria_nbv.data`.
[ ] Deprecation behavior is tested if shim remains.
```

---

## Issue 6 — Complete `data_handling` legacy cutover

**Priority:** P0
**Area:** package, data

**Scope**

Remove `_legacy_*` modules and compatibility wrappers, or quarantine them under `data_handling/legacy/` with a deletion deadline.

**Acceptance**

```text
[ ] No hard-coded `/home/jandu/repos/NBV` paths.
[ ] No old `NBV` repo paths.
[ ] One canonical offline cache API.
[ ] `rg "_legacy|NBV_LEGACY"` is clean or only hits quarantined files.
```

---

## Issue 7 — Demagic `data_handling/__init__.py`

**Priority:** P1
**Area:** package

**Scope**

Replace large lazy-export map with small explicit exports.

**Acceptance**

```text
[ ] `__init__.py` is under ~30 lines.
[ ] Imports are explicit.
[ ] No hidden legacy exports.
[ ] Tests import canonical symbols directly from owning modules.
```

---

## Issue 8 — Split and clean `vin/model_v3.py`

**Priority:** P0
**Area:** VIN, package

**Scope**

Move frame checks, scene field assembly, semidense features, voxel projection, pose encoding, and CORAL head into separate modules.

**Acceptance**

```text
[ ] `model_v3.py` is a thin composition file.
[ ] Deprecated config fields removed.
[ ] No inline TODOs without issue links.
[ ] CW90/frame consistency covered by tests.
[ ] Historical sweep commentary moved to docs/history.
```

---

## Issue 9 — Consolidate rendering/backprojection

**Priority:** P0
**Area:** geometry, rendering

**Scope**

Unify renderer and backprojection APIs. Remove duplicate conventions.

**Acceptance**

```text
[ ] One PyTorch3D renderer implementation.
[ ] One NDC-aligned backprojection implementation.
[ ] Tests verify rendered depth backprojects onto mesh.
[ ] Candidate pointcloud code uses the same camera path as oracle labels.
```

---

## Issue 10 — Clean `utils` dumping ground

**Priority:** P1
**Area:** package

**Scope**

Move plotting/reporting/W&B/Optuna/schema utilities into domain packages.

**Acceptance**

```text
[ ] `utils` contains only truly generic helpers.
[ ] Plotting lives in diagnostics/app-specific modules.
[ ] Reporting lives in reporting/ or experiments/.
[ ] No duplicate plotting modules remain.
```

---

## Issue 11 — Move markdown/context files out of runtime package

**Priority:** P1
**Area:** package, docs

**Scope**

Move `rri_metrics/context_pytorch_3d_losses.md` out of `aria_nbv`.

**Acceptance**

```text
[ ] No `.md` context dump lives inside runtime source packages.
[ ] Reference material lives in docs or `.agents/references`.
```

---

## Issue 12 — Audit tiny/stretch packages

**Priority:** P2
**Area:** package

**Scope**

Evaluate whether `interpretability`, `vin/experimental`, PointNeXt-related code, and SceneScript-related local external code are current or stretch.

**Acceptance**

```text
[ ] Each stretch surface is promoted, archived, or deleted.
[ ] No unsupported experimental package remains importable as core.
```

---

# Recommended deletion order

Do this in order; each step reduces noise without blocking research.

```text
1. Delete `.logs` artifacts from git.
2. Rewrite README + repo metadata.
3. Rebuild Quarto nav and remove public agent/work-management pages.
4. Collapse duplicate implementation docs.
5. Fix `pyproject.toml` formatting and dependency extras.
6. Finish `aria_nbv.data` migration.
7. Complete `data_handling` legacy cutover.
8. Split `vin/model_v3.py`.
9. Consolidate rendering/backprojection.
10. Clean `utils`.
11. Archive/delete experimental/stretch packages.
```

The main simplification principle: **delete public ambiguity before deleting code**. First make the repo tell one clean story; then cut compatibility and package bloat behind that story.

[1]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.logs "ARIA-NBV/.logs at main · JanDuchscherer104/ARIA-NBV · GitHub"
[2]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.logs/checkpoints "ARIA-NBV/.logs/checkpoints at main · JanDuchscherer104/ARIA-NBV · GitHub"
[3]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.logs/ckpts "ARIA-NBV/.logs/ckpts at main · JanDuchscherer104/ARIA-NBV · GitHub"
[4]: https://janduchscherer104.github.io/ARIA-NBV/ "Next-Best-View Planning with Foundation Models – Semantic NBV Planning"
[5]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/_quarto.yml "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/README.md "raw.githubusercontent.com"
[7]: https://github.com/JanDuchscherer104/ARIA-NBV "GitHub - JanDuchscherer104/ARIA-NBV · GitHub"
[8]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/data "ARIA-NBV/aria_nbv/aria_nbv/data at main · JanDuchscherer104/ARIA-NBV · GitHub"
[9]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/pyproject.toml "raw.githubusercontent.com"
[10]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/data_handling "ARIA-NBV/aria_nbv/aria_nbv/data_handling at main · JanDuchscherer104/ARIA-NBV · GitHub"
[11]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/aria_nbv/vin/model_v3.py "raw.githubusercontent.com"
[12]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/contents/resources/agent_scaffold "ARIA-NBV/docs/contents/resources/agent_scaffold at main · JanDuchscherer104/ARIA-NBV · GitHub"
[13]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/AGENTS.md "raw.githubusercontent.com"
[14]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs "ARIA-NBV/docs at main · JanDuchscherer104/ARIA-NBV · GitHub"
[15]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/contents/ext-impl "ARIA-NBV/docs/contents/ext-impl at main · JanDuchscherer104/ARIA-NBV · GitHub"
[16]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/contents/impl "ARIA-NBV/docs/contents/impl at main · JanDuchscherer104/ARIA-NBV · GitHub"
[17]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/contents/literature "ARIA-NBV/docs/contents/literature at main · JanDuchscherer104/ARIA-NBV · GitHub"
[18]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/aria_nbv/data_handling/__init__.py "raw.githubusercontent.com"
[19]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/vin "ARIA-NBV/aria_nbv/aria_nbv/vin at main · JanDuchscherer104/ARIA-NBV · GitHub"
[20]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/rri_metrics "ARIA-NBV/aria_nbv/aria_nbv/rri_metrics at main · JanDuchscherer104/ARIA-NBV · GitHub"
[21]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/rendering "ARIA-NBV/aria_nbv/aria_nbv/rendering at main · JanDuchscherer104/ARIA-NBV · GitHub"
[22]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/utils "ARIA-NBV/aria_nbv/aria_nbv/utils at main · JanDuchscherer104/ARIA-NBV · GitHub"
[23]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/interpretability "ARIA-NBV/aria_nbv/aria_nbv/interpretability at main · JanDuchscherer104/ARIA-NBV · GitHub"
[24]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/app "ARIA-NBV/aria_nbv/aria_nbv/app at main · JanDuchscherer104/ARIA-NBV · GitHub"
[25]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/external "ARIA-NBV/external at main · JanDuchscherer104/ARIA-NBV · GitHub"
[26]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/scripts "ARIA-NBV/scripts at main · JanDuchscherer104/ARIA-NBV · GitHub"
[27]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/AGENTS.md "raw.githubusercontent.com"
[28]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/pose_generation "ARIA-NBV/aria_nbv/aria_nbv/pose_generation at main · JanDuchscherer104/ARIA-NBV · GitHub"
[29]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/lightning "ARIA-NBV/aria_nbv/aria_nbv/lightning at main · JanDuchscherer104/ARIA-NBV · GitHub"
[30]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/tests "ARIA-NBV/aria_nbv/tests at main · JanDuchscherer104/ARIA-NBV · GitHub"
[31]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/AGENTS.md "raw.githubusercontent.com"
