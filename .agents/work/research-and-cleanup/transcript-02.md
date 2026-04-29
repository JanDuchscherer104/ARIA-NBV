# Branched from .agents/work/research-and-cleanup/transcript-01.md. Same transcript until [User-Turn11].

[User-Turn11]
Please do a cross review of generalizable / transferable best practices, skills, agents scaffold, ...

https://github.com/JanDuchscherer104/ARIA-NBV vs https://github.com/JanDuchscherer104/prml-vslam. project ARIA-NBV's scaffolding isn't as good, it contains plenty of slop in the docs/ folder!

we need to develop better custom skills and also mabye employ oh my codex or something alike.

[Agent-Turn12]
## Executive diagnosis

`prml-vslam` has the better **operational scaffold**: clearer README, clearer setup boundary, stronger CI/pre-commit surface, better issue/backlog lifecycle, tool-specific agent shims, graph/memory infrastructure, and reviewer-oriented PR/issue skills. `ARIA-NBV` has the more complex research stack and already has a serious agent scaffold, but its docs are mixing too many roles: public thesis narrative, scratchpad notes, generated context, Quarto site material, Typst paper, literature dumps, implementation notes, agent scaffold mirrors, and roadmap/TODO material. That is exactly why the `docs/` folder feels sloppy.

The priority should be:

1. **Do not add more docs yet.**
2. **Classify and prune docs.**
3. **Move agent and planning surfaces out of public docs.**
4. **Port the good PRML scaffold pieces into ARIA carefully.**
5. **Develop ARIA-specific skills for oracle RRI, VIN, rollouts, entity-aware NBV, and documentation hygiene.**
6. **Use Oh My Codex / OMX only as an optional local orchestration layer, not as the canonical repo scaffold.**

`ARIA-NBV` already has a technically rich project: oracle RRI labeling, offline caches, frozen EVL features, VIN v3 candidate scoring, W&B/Optuna tooling, Streamlit diagnostics, and Typst/Quarto reporting are all present in the project description. That complexity justifies a stronger scaffold, but also makes slop more dangerous because agents can easily pick the wrong source of truth.

---

# 1. High-level comparison

| Area            | `prml-vslam`                                                                                                          | `ARIA-NBV`                                                                                                                        | Recommendation                                                      |
| --------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| README          | Strong front door: status, docs map, quick commands, deliverables, starting points.                                   | Much thinner; mostly points to GitHub Pages and lists commands.                                                                   | Port PRML-style README structure into ARIA.                         |
| Setup           | Dedicated `SETUP.md` owns environment/runbook details.                                                                | Setup is mostly scattered across README, docs, AGENTS, and package commands.                                                      | Add ARIA `SETUP.md`.                                                |
| Docs            | More disciplined split: docs/report, docs/slides, docs/architecture, docs/pipeline, docs/Questions.md.                | Too broad: contents, generated context, literature source, resources/agent_scaffold, ideas, todos, roadmap, implementation notes. | Triage docs into public docs, canonical memory, generated, archive. |
| Agent guidance  | Thin root `AGENTS.md`; nested deltas; explicit local sources of truth.                                                | More elaborate, but denser and partially inconsistent.                                                                            | Keep ARIA’s progressive disclosure, but simplify root.              |
| Skills          | Richer set: create PR, issue lifecycle, graphify, mempalace, pydantic, streamlit architecture, Rerun, simplification. | Good but narrower: context, code review, simplification, Python docstrings, Mojo acceleration, agents-db.                         | Port selected PRML skills, but adapt names and contracts.           |
| Backlog         | `.agents/issues.toml`, `todos.toml`, `refactors.toml`, `resolved.toml`, plus `make agents-db`.                        | Has an `agents-db` skill, but the visible `.agents` listing does not show the TOML DB files.                                      | Either add the ARIA DB or remove/fix the skill.                     |
| CI              | Has `.github/workflows/ci.yml` and `quarto-publish.yml`.                                                              | Visible workflow listing only shows `quarto-publish.yml`.                                                                         | Add ARIA CI workflow.                                               |
| Pre-commit      | Has `.pre-commit-config.yaml`.                                                                                        | I did not see a root pre-commit file in the public root listing.                                                                  | Add pre-commit for Ruff, TOML, merge conflicts.                     |
| Agent memory    | PRML uses local DB plus MemPalace.                                                                                    | ARIA uses `.agents/memory/state` with PROJECT_STATE, DECISIONS, OPEN_QUESTIONS, GOTCHAS.                                          | ARIA memory is good; combine with backlog DB.                       |
| Knowledge graph | PRML has `graphify-out/` and Graphify workflow.                                                                       | ARIA has its own context-generation scripts.                                                                                      | Keep ARIA context scripts; optionally add graphify later.           |

The strongest PRML advantage is that its root README explicitly defines a “Documentation Map,” status, quick entry commands, challenge context, evaluation, deliverables, and starting points. ([GitHub][1]) By contrast, ARIA’s README currently gives a thin project title, a link to GitHub Pages, and command snippets, while the repo sidebar says “No description, website, or topics provided.” ([GitHub][2])

---

# 2. What `prml-vslam` does better

## 2.1 It separates human docs from agent docs

PRML’s README tells humans where to look:

* `README.md` for project front door.
* `SETUP.md` for environment setup.
* package `README.md` and `REQUIREMENTS.md` for implementation contracts.
* `AGENTS.md` for agent workflow.
* `docs/Questions.md` for clarified scope and intent. ([GitHub][1])

That is exactly the pattern ARIA needs. The problem with ARIA is not lack of documentation; it is lack of **document-role discipline**.

For ARIA, every document should be assigned one of these roles:

```text
public thesis / project explanation
developer runbook
package contract
agent instruction
canonical project memory
backlog / issue tracker
generated context
historical archive
scratchpad
```

Anything that does not have one clear role should either be deleted, moved, or renamed.

## 2.2 It has a cleaner agent policy

PRML’s root `AGENTS.md` is compact. It says:

* root `AGENTS.md` is the only full repo-wide agent policy;
* nested `AGENTS.md` files add only deltas;
* README, Questions, references, and package docs are sources of truth;
* agents should read the nearest nested guide before editing;
* work should stay scoped;
* `make ci` should run before commits. ([GitHub][3])

ARIA’s root `AGENTS.md` is more advanced in some ways, but it is dense and tries to encode many things at once: context routing, memory, references, commands, gotchas, verification matrix, debrief rules, retrieved-on-demand docs, and multiple project-specific constraints. ([GitHub][4])

The better pattern is:

```text
AGENTS.md
  = short repo-wide invariant rules

nested AGENTS.md
  = local deltas only

.agents/skills/*
  = task workflows

.agents/memory/state/*
  = durable current truth

.agents/issues.toml etc.
  = active maintenance backlog

docs/*
  = human-facing docs only
```

## 2.3 It has PR and issue lifecycle skills

PRML has a `create-pr` skill that defines a reviewer-first PR body: summary paragraph, focused verification, work-package overview, detailed work-package sections, and explicit status language. ([GitHub][5])

It also has a `gh-issue-lifecycle` skill that requires issues to be de-duplicated, evidence-backed, and acceptance-criteria-driven. ([GitHub][6])

ARIA should port both. This is especially useful because ARIA currently has 5 open PRs visible on GitHub, but only one issue. ([GitHub][2]) A research repo with many moving parts needs explicit issue/PR hygiene, otherwise “docs slop” becomes “architecture slop.”

## 2.4 It has better local CI

PRML’s Makefile exposes a clear local verification suite:

```bash
make lint
make lint-check
make test
make ci
make report-pdf
make slides-pdf
make docs-build
make agents-db
make graphify
make mempalace-search
```

Its `make ci` runs pre-commit, tests, report build, and slides build. ([GitHub][7])

ARIA’s Makefile is powerful, but it is currently dominated by context-generation and documentation helper targets. It has excellent agent-context commands, but not the same clean “local CI as first-class contract” shape. ([GitHub][8])

ARIA should add a simple top-level contract:

```bash
make fmt
make lint
make test
make docs-check
make paper-pdf
make ci
```

Then keep specialized context targets as secondary.

## 2.5 It has repo-local memory and knowledge systems

PRML uses:

* `.agents/issues.toml`, `todos.toml`, `refactors.toml`, `resolved.toml`;
* `make agents-db`;
* Graphify for codebase graph navigation;
* MemPalace for repo-local mined docs and Codex chat histories. ([GitHub][9])

The MemPalace plugin is implemented as a repo-local Codex plugin that exposes a repo-local MCP server and pins the palace under `.artifacts/mempalace/palace`. ([GitHub][10]) It also has Codex hooks for `SessionStart`, `Stop`, and `PreCompact`. ([GitHub][11])

ARIA should not blindly copy this entire setup yet. But the pattern is useful:

```text
short root policy
+ task-specific skills
+ active backlog DB
+ current-state memory
+ optional graph/memory retrieval
```

---

# 3. What `ARIA-NBV` already does well

ARIA is not starting from zero. It already has several good scaffold ideas.

## 3.1 Progressive disclosure is already present

ARIA’s root `AGENTS.md` explicitly says repo skills live in `.agents/skills`, canonical memory in `.agents/memory`, references in `.agents/references`, and generated context in `docs/_generated/context`. It also directs agents to open the paper, canonical state docs, and source index first, then only open deeper guides when needed. ([GitHub][4])

That is good. Keep this idea.

## 3.2 ARIA has strong canonical memory

ARIA has `.agents/memory/state/PROJECT_STATE.md`, `DECISIONS.md`, `OPEN_QUESTIONS.md`, and `GOTCHAS.md`. ([GitHub][12])

The current `PROJECT_STATE.md` is especially valuable because it already captures the thesis spine: oracle RRI, discrete candidates, VIN-style scoring, non-myopic rollout work, geometry-first counterfactual state, and the need to keep continuous-control claims bounded. ([GitHub][13])

This should become the **main source of truth for agent orientation**, not random Quarto notes.

## 3.3 ARIA has better domain-specific technical guides than PRML

The ARIA nested package guides are more domain-aware than PRML’s. For example, the data-handling guide names cache contracts, raw snippet surfaces, VIN cache flows, and warns against hand-editing derived cache metadata. ([GitHub][14])

The VIN guide explicitly treats scorer inputs, prediction semantics, batch/container shapes, candidate-vs-rig frame semantics, and paper/docs updates as cross-surface contracts. ([GitHub][15])

That is exactly the type of local delta guidance agents need.

So the goal is not “replace ARIA scaffold with PRML scaffold.” The goal is:

```text
Keep ARIA’s domain-specific scaffolding.
Adopt PRML’s operational discipline.
Remove copied/stale/inconsistent surfaces.
```

---

# 4. Main ARIA problems

## 4.1 `docs/` mixes too many responsibilities

ARIA’s `docs/` contains:

* `_freeze/contents`
* `_generated/context`
* `contents`
* `figures`
* `literature/tex-src`
* `reference`
* `typst`
* Quarto config and index
* bibliography ([GitHub][16])

Inside `docs/contents`, ARIA has:

* `experiments`
* `ext-impl`
* `impl`
* `literature`
* `resources/agent_scaffold`
* `theory`
* `ase_dataset.qmd`
* `ideas.qmd`
* `questions.qmd`
* `repo_structure.qmd`
* `resources.qmd`
* `roadmap.qmd`
* `setup.qmd`
* `todos.qmd` ([GitHub][17])

That is too much for a public docs tree. It mixes:

```text
human docs
internal notes
agent scaffold mirrors
implementation scratchpads
generated context
public thesis narrative
roadmap
todos
literature index
```

This should be split.

## 4.2 ARIA’s `agents-db` skill appears stale/copied

ARIA’s `.agents/skills` contains `agents-db`, `aria-nbv-context`, `code-review`, `mojo-nbv-acceleration`, `python-docstrings`, and `simplification`. ([GitHub][18])

But ARIA’s `agents-db` skill still says “Use when working with PRML VSLAM’s internal agent memory” and references `.agents/issues.toml`, `.agents/todos.toml`, `.agents/refactors.toml`, and `.agents/resolved.toml`. ([GitHub][19]) The visible ARIA `.agents/` listing shows `archive`, `memory`, `references`, `skills`, and `workspace`, but not those TOML DB files. ([GitHub][20])

So either:

```text
Option A: port the PRML agents DB files/scripts properly into ARIA
```

or:

```text
Option B: delete/disable the agents-db skill from ARIA
```

Do not leave it half-present.

## 4.3 ARIA lacks PRML’s clean setup/runbook split

PRML has a dedicated `SETUP.md` that owns environment setup, Streamlit, optional ViSTA/CUDA setup, validation, and Codex/MemPalace utilities. ([GitHub][21])

ARIA currently puts many commands in README and agent guidance. It should get a dedicated `SETUP.md` with:

```text
Base setup
GPU/PyTorch/PyTorch3D setup
EFM3D / ATEK setup
external submodules
offline cache setup
W&B setup
Streamlit launch
paper/docs build
common failures
```

## 4.4 ARIA has only a publish workflow visible

The ARIA `.github/workflows` listing shows `quarto-publish.yml`. ([GitHub][22]) PRML has both `ci.yml` and `quarto-publish.yml`. ([GitHub][23])

ARIA should add CI, even if the full GPU stack cannot run on GitHub Actions. A useful CPU-only CI can still run:

```bash
uv lock --check
ruff format --check
ruff check
pytest -m "not slow and not integration"
python -m compileall
quarto check  # optional
typst compile docs/typst/paper/main.typ --root docs  # if feasible
```

## 4.5 Root agent guidance is too dense

Official Codex guidance says `AGENTS.md` should include repo layout, how to run/build/test, conventions, constraints, PR expectations, and what “done” means, but it should stay practical and short. ([OpenAI Developers][24])

ARIA’s root AGENTS is not wrong, but it is compressed into very dense lines and includes many responsibilities. The root should become more like PRML:

```text
sources of truth
repo map
core rules
default commands
verification
where to go next
```

Then local guides and skills should own the rest.

---

# 5. Target scaffold for ARIA

I would aim for this structure:

```text
AGENTS.md
CLAUDE.md
GEMINI.md
SETUP.md
README.md
Makefile
.pre-commit-config.yaml

.agents/
  AGENTS_INTERNAL_DB.md
  issues.toml
  todos.toml
  refactors.toml
  resolved.toml
  memory/
    state/
      PROJECT_STATE.md
      DECISIONS.md
      OPEN_QUESTIONS.md
      GOTCHAS.md
    history/
  references/
    agent_reference.md
    python_conventions.md
    nbv_math_reference.md
    external_library_ids.md
    pr_body_shape.md
    issue_structure.md
  skills/
    aria-nbv-context/
    oracle-rri-contracts/
    vin-training-diagnostics/
    counterfactual-rollout-planner/
    entity-aware-rri/
    docs-curator/
    scientific-writing/
    create-pr/
    gh-issue-lifecycle/
    agents-db/
    pydantic/
    simplification/
    streamlit-diagnostics/

docs/
  index.qmd
  contents/
    method/
    experiments/
    theory/
    literature/
    setup.qmd
    glossary.qmd
  typst/
    paper/
    slides/
  reference/
  figures/
  references.bib
  _generated/     # generated, preferably untracked
  _freeze/        # only if needed for Quarto execution state

aria_nbv/
  AGENTS.md
  README.md
  REQUIREMENTS.md
  aria_nbv/
    data_handling/
      AGENTS.md
      README.md
      REQUIREMENTS.md
    rri_metrics/
      AGENTS.md
      README.md
      REQUIREMENTS.md
    vin/
      AGENTS.md
      README.md
      REQUIREMENTS.md
    rollout/
      AGENTS.md
      README.md
      REQUIREMENTS.md
    entity/
      AGENTS.md
      README.md
      REQUIREMENTS.md
```

The important change is not the exact names. The important change is that every surface has one owner:

```text
README.md = human project front door
SETUP.md = environment/runbook
AGENTS.md = agent policy
.agents/memory/state = current truth
.agents/issues.toml = active maintenance debt
docs/ = public thesis / project docs
package README/REQUIREMENTS = implementation contracts
```

---

# 6. Skills to develop or port

OpenAI’s Codex skills use a `SKILL.md` file with required `name` and `description`, plus optional scripts, references, assets, and agents. Codex only loads the full skill when it decides the skill is relevant, which makes skills a good fit for progressive disclosure. ([OpenAI Developers][25])

## 6.1 Keep and improve: `aria-nbv-context`

ARIA’s existing `aria-nbv-context` skill is already a good discovery layer. It localizes tasks through paper, memory, generated context, references, outlines, AST summaries, and targeted search. ([GitHub][26])

Improve it by:

* shortening the hot path;
* explicitly forbidding `docs/contents/ideas.qmd` as source of truth unless the user asks for brainstorming;
* adding direct routing to new domain skills;
* removing stale references to generated mirrored agent docs if those are removed from public docs.

## 6.2 Add: `oracle-rri-contracts`

Use when touching:

```text
depth rendering
backprojection
NDC alignment
point↔mesh scoring
Chamfer accuracy/completeness
oracle labels
RRI binning
candidate validity
```

It should enforce:

```text
- Use PoseTW / CameraTW.
- State transform direction explicitly.
- Do not mix display-only frame corrections into model/cache/render paths.
- Do not change cache format without migration or rebuild instructions.
- Verify with targeted rendering/backprojection/RRI tests.
```

This skill should encode the equations from the ARIA paper:

[
CD(P,M)=A(P,M)+C(P,M)
]

[
RRI(q)=\frac{CD(P_t,M_{GT})-CD(P_t\cup P_q,M_{GT})}{CD(P_t,M_{GT})+\epsilon}
]

Those are central enough that agents should not have to rediscover them from the paper every time.

## 6.3 Add: `vin-training-diagnostics`

Use when touching:

```text
VIN v3/v4 model
CORAL
ordinal bins
Lightning module
W&B logging
Optuna sweeps
batch contracts
candidate ranking metrics
```

It should enforce:

```text
- Report Spearman, top-k bin accuracy, confusion matrix, label histograms.
- Distinguish cumulative probabilities from class marginals in CORAL.
- Check stage dependence and label collapse.
- Do not compare ablations if schedules/losses/batch sizes changed simultaneously.
- Always name the changed contract: input batch, prediction semantics, loss, metric, cache field.
```

## 6.4 Add: `counterfactual-rollout-planner`

Use when touching:

```text
multi-step rollouts
beam width
softmax sampling
Gumbel sampling
trajectory traces
cumulative RRI
Gymnasium / SB3 scaffold
```

It should enforce your key planning decision:

[
\text{cost}=O(B\cdot L\cdot N)
]

not:

[
O(N^L)
]

It should also define beam semantics:

```text
beam_width = number of sampled rollout chains
not full combinatorial branching
```

This avoids the old “search explosion” problem.

## 6.5 Add: `entity-aware-rri`

Use when touching:

```text
OBB-cropped RRI
object selection
target-conditioned candidate generation
entity-specific metrics
semantic/entity objectives
```

It should define:

[
RRI_e(q)
========

\frac{
CD(P_t^e,M_{GT}^e)-CD(P_t^e\cup P_q^e,M_{GT}^e)
}{
CD(P_t^e,M_{GT}^e)+\epsilon
}
]

and enforce:

```text
- invalid candidate is a constraint, not just lowest ordinal bin;
- OBB margins must be explicit;
- object crop must be applied consistently to current points, candidate points, and GT mesh;
- report scene-level and entity-level metrics separately.
```

## 6.6 Add: `docs-curator`

This is the most urgent ARIA skill.

Use when touching:

```text
docs/
Quarto pages
Typst paper/slides
roadmap/todos/questions
generated context
agent scaffold mirrors
```

It should triage every doc into one of:

```text
KEEP_PUBLIC
MOVE_TO_AGENTS_MEMORY
MOVE_TO_PACKAGE_REQUIREMENTS
MOVE_TO_ARCHIVE
GENERATED_UNTRACKED
DELETE
```

It should enforce:

```text
- no duplicated source of truth;
- no stale roadmap in public docs;
- no agent scaffold mirror in public docs unless explicitly intended;
- no speculative notes presented as current project truth;
- paper > canonical memory > package contract > Quarto summary.
```

## 6.7 Port from PRML: `create-pr`

Port almost directly. It is well structured and reviewer-friendly. ([GitHub][5])

Adapt only:

```text
PRML-specific names → ARIA-NBV
work package examples → oracle, VIN, rollout, docs, cache, entity-aware, CI
validation commands → ARIA make targets
```

## 6.8 Port from PRML: `gh-issue-lifecycle`

Port almost directly. It will help convert vague “docs slop” into bounded issues with acceptance criteria. ([GitHub][6])

## 6.9 Port from PRML but adapt: `agents-db`

Do not copy as-is. ARIA’s current copy is stale. It still says PRML VSLAM. ([GitHub][19])

The ARIA version should say:

```text
Use when working with ARIA-NBV's internal agent backlog:
.agents/issues.toml
.agents/todos.toml
.agents/refactors.toml
.agents/resolved.toml
```

And it should include ARIA-specific categories:

```text
oracle-rri
cache-contract
vin
rollout
entity-aware
docs
ci
streamlit
paper
dataset
external-deps
```

## 6.10 Optional: `graphify`

PRML’s graphify skill gives a useful rule: inspect `graphify-out/GRAPH_REPORT.md` before architecture/codebase questions and rebuild graph artifacts after code edits. ([GitHub][27])

ARIA already has strong context-generation scripts, so Graphify is optional. I would add it only if:

```text
- graphify artifacts are kept fresh automatically;
- agents actually use it;
- it does not create another stale generated-doc tree.
```

## 6.11 Optional: `mempalace-repo`

PRML’s MemPalace setup is powerful: it mines docs and Codex histories into a repo-local palace and exposes search/wake-up workflows. ([GitHub][28])

For ARIA, use it only after docs cleanup. Otherwise it will mine noisy docs and preserve slop as memory.

---

# 7. Oh My Codex / OMX recommendation

OpenAI’s official Codex scaffold already supports:

* `AGENTS.md` repo guidance;
* nested local instructions;
* Codex CLI;
* task-specific skills. ([OpenAI Developers][29])

Oh My Codex / OMX appears to be an external orchestration layer around Codex. Its own docs describe the mental model as: Codex does the actual agent work; OMX adds reusable role keywords, reusable skills, and `.omx/` runtime state. ([GitHub][30])

My recommendation:

```text
Do not make OMX canonical in ARIA-NBV yet.
```

Use this instead:

1. Build first-class `AGENTS.md` + `.agents/skills` + `.agents/issues.toml`.
2. Clean docs.
3. Add CI and pre-commit.
4. Then pilot OMX locally for two tasks:

   * docs cleanup / classification
   * multi-agent PR review
5. Only vendor `.omx/` or OMX conventions if it demonstrably reduces errors.

Why: ARIA already has too many state surfaces. Adding `.omx/` before pruning will create another memory/backlog layer and likely make the “slop” worse.

A safe policy:

```text
OMX may be user-local or optional.
ARIA canonical state remains AGENTS.md + .agents/ + package docs + docs/paper.
```

---

# 8. Concrete ARIA cleanup plan

## PR 1 — Fix obvious scaffold inconsistencies

Acceptance criteria:

```text
- Fix .agents/skills/agents-db/SKILL.md so it says ARIA-NBV, not PRML VSLAM.
- Either add .agents/issues.toml, todos.toml, refactors.toml, resolved.toml and script support, or remove agents-db from ARIA.
- Add docs-curator skill stub.
- Add create-pr and gh-issue-lifecycle skills.
```

## PR 2 — Add README / SETUP split

ARIA root README should become:

```text
# ARIA-NBV

## What this repository does

## Current status

## Documentation map

## Quick entry

## Core workflows

## Evaluation / experiments

## Deliverables

## Starting points
```

Add `SETUP.md` with environment and runbook details. Use PRML’s `SETUP.md` pattern, but adapt it to EFM3D/ATEK/PyTorch3D. ([GitHub][21])

## PR 3 — Add local CI and pre-commit

Acceptance criteria:

```text
- Add .pre-commit-config.yaml.
- Add make fmt, make lint, make test, make ci.
- Add .github/workflows/ci.yml.
- CI can skip GPU-heavy tests but must validate import, lint, unit tests, and docs sanity.
```

PRML’s pre-commit file is minimal and good: merge-conflict check, TOML check, trailing whitespace, Ruff check, Ruff format. ([GitHub][31])

## PR 4 — Docs triage inventory

Run a one-time docs inventory:

```text
docs_path
role
owner
keep/move/delete/archive/generated
reason
target_location
```

Classify all docs as:

```text
PUBLIC
PAPER
SLIDES
REFERENCE
PACKAGE_CONTRACT
AGENT
MEMORY
BACKLOG
GENERATED
ARCHIVE
SCRATCHPAD
```

Do not edit content heavily yet. First classify.

## PR 5 — Move agent/scaffold material out of public docs

Likely moves:

```text
docs/contents/resources/agent_scaffold
  -> remove from public docs or regenerate as untracked mirror only

docs/contents/todos.qmd
  -> .agents/todos.toml or .agents/memory/state/OPEN_QUESTIONS.md

docs/contents/roadmap.qmd
  -> .agents/memory/state/PROJECT_STATE.md or a concise public roadmap

docs/contents/ideas.qmd
  -> .agents/memory/history or notes/archive unless explicitly public
```

The current ARIA docs tree explicitly contains `resources/agent_scaffold`, `ideas.qmd`, `questions.qmd`, `roadmap.qmd`, and `todos.qmd`. ([GitHub][17]) That is the visible slop vector.

## PR 6 — Add ARIA-specific skills

Add:

```text
oracle-rri-contracts
vin-training-diagnostics
counterfactual-rollout-planner
entity-aware-rri
docs-curator
scientific-writing
```

Keep each skill short and trigger-specific. Do not create another giant instruction blob.

## PR 7 — Add package-local `REQUIREMENTS.md`

PRML’s README says package-level `README.md` and `REQUIREMENTS.md` files carry concise contracts and implementation boundaries. ([GitHub][1])

ARIA should add these at least for:

```text
aria_nbv/aria_nbv/data_handling/REQUIREMENTS.md
aria_nbv/aria_nbv/rri_metrics/REQUIREMENTS.md
aria_nbv/aria_nbv/vin/REQUIREMENTS.md
aria_nbv/aria_nbv/rollout/REQUIREMENTS.md
```

These should be engineering contracts, not essays.

---

# 9. Minimal root `AGENTS.md` target shape for ARIA

I would compress ARIA’s root file to something like this:

```markdown
# ARIA-NBV Agent Guidance

This repository develops quality-driven next-best-view planning for egocentric Aria/ASE scenes.

## Sources of truth

- `README.md`: human project front door.
- `SETUP.md`: environment and runbook.
- `docs/typst/paper/main.typ`: thesis/paper narrative.
- `.agents/memory/state/PROJECT_STATE.md`: current project truth.
- `.agents/memory/state/DECISIONS.md`: durable decisions.
- `.agents/memory/state/OPEN_QUESTIONS.md`: unresolved research/system questions.
- `.agents/memory/state/GOTCHAS.md`: recurring pitfalls.
- nearest nested `AGENTS.md`: local rules.

## Repo map

- `aria_nbv/`: Python package workspace.
- `aria_nbv/aria_nbv/data_handling/`: snippets and caches.
- `aria_nbv/aria_nbv/rri_metrics/`: oracle RRI and metrics.
- `aria_nbv/aria_nbv/vin/`: candidate scorer.
- `docs/`: public docs, paper, slides, references.
- `.agents/`: agent skills, memory, backlog, references.

## Default workflow

1. Restate task and acceptance criteria.
2. Read nearest applicable `AGENTS.md`.
3. Use `aria-nbv-context` only when the target surface is unclear.
4. Edit the narrowest owner surface.
5. Run surface-specific validation.
6. Update docs or memory only when current truth changed.

## Commands

- Setup: `cd aria_nbv && uv sync --all-extras`
- Format: `ruff format`
- Lint: `ruff check`
- Test: `uv run pytest`
- Context: `make context`
- CI: `make ci`

## Hard rules

- Do not use destructive git commands unless explicitly requested.
- Preserve `PoseTW` / `CameraTW` and explicit frame semantics.
- Do not hand-edit derived cache artifacts.
- Do not duplicate source-of-truth docs.
- Do not silently add compatibility fallbacks.
```

The current ARIA root has good content, but much of it should move into skills and references.

---

# 10. What I would transfer from PRML now

Transfer immediately:

```text
create-pr skill
gh-issue-lifecycle skill
agents-db pattern, but rewritten for ARIA
.pre-commit-config.yaml pattern
make ci / lint-check pattern
SETUP.md pattern
README Documentation Map pattern
tool-specific CLAUDE.md / GEMINI.md shims
package README + REQUIREMENTS discipline
```

Transfer later:

```text
Graphify
MemPalace
Oh My Codex / OMX
```

Do not transfer:

```text
VSLAM-specific frame rules
ViSTA/Rerun-specific details unless ARIA has equivalent Rerun needs
PRML-specific issue categories
PRML-specific AGENTS_INTERNAL_DB facts
```

---

# 11. Best-practice rules for the new scaffold

The durable rules should be:

1. **One owner per fact.**
   If the same fact appears in README, Quarto, Typst, AGENTS, and memory, it will drift.

2. **Public docs are not scratchpads.**
   Meeting notes, TODOs, active debates, and agent scaffolds belong in `.agents/`, not public docs.

3. **Skills are workflows, not encyclopedias.**
   A skill should say when to use it, what it owns, what to read, what to do, and how to verify.

4. **Root AGENTS is short.**
   Root guidance routes. Nested guidance constrains. Skills execute.

5. **Backlog is structured.**
   Use issues/todos/refactors TOML or GitHub issues. Do not bury active engineering work inside prose docs.

6. **Generated context is generated.**
   Keep it untracked or clearly marked. Do not let agents treat generated context as ground truth.

7. **CI is part of the scaffold.**
   If an agent cannot run a simple `make ci`, the repo will accumulate silent breakage.

8. **Paper and implementation must agree.**
   For ARIA, code changes that affect oracle RRI, VIN, rollouts, or entity objectives should update the paper or implementation docs.

9. **Avoid copied generic skills.**
   The ARIA `agents-db` issue shows why: copied skills silently import another project’s assumptions.

10. **Introduce orchestration only after pruning.**
    OMX, MemPalace, and graph tools are useful only if the underlying source-of-truth structure is clean.

---

# Bottom line

`ARIA-NBV` should inherit PRML’s **operational discipline**, not its domain content. The strongest immediate move is a scaffold cleanup PR series:

```text
1. Fix stale ARIA skills.
2. Add README/SETUP/CI/pre-commit.
3. Introduce ARIA agent DB.
4. Port PR/issue lifecycle skills.
5. Triage docs by role.
6. Move scratchpad/agent/planning docs out of public docs.
7. Add ARIA-specific skills for oracle RRI, VIN, rollouts, entity-aware RRI, and docs curation.
```

Oh My Codex can be useful, but only as an optional local orchestration layer after the canonical scaffold is clean. The canonical repo should remain based on official `AGENTS.md`, local `.agents/skills`, `.agents/memory/state`, and structured backlog files.

[1]: https://github.com/JanDuchscherer104/prml-vslam "GitHub - JanDuchscherer104/prml-vslam: Off-device monocular dense VSLAM benchmark for uncalibrated smartphone video streams, evaluating trajectory accuracy, dense 3D reconstruction quality, and efficiency on ADVIO and custom ARCore-logged datasets. · GitHub"
[2]: https://github.com/JanDuchscherer104/ARIA-NBV "GitHub - JanDuchscherer104/ARIA-NBV · GitHub"
[3]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/AGENTS.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/AGENTS.md "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.agents/skills/create-pr/SKILL.md "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.agents/skills/gh-issue-lifecycle/SKILL.md "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/Makefile "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/Makefile "raw.githubusercontent.com"
[9]: https://github.com/JanDuchscherer104/prml-vslam/tree/main/.agents "prml-vslam/.agents at main · JanDuchscherer104/prml-vslam · GitHub"
[10]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.codex-plugin/plugin.json "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.codex-plugin/hooks.json "raw.githubusercontent.com"
[12]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.agents/memory/state "ARIA-NBV/.agents/memory/state at main · JanDuchscherer104/ARIA-NBV · GitHub"
[13]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.agents/memory/state/PROJECT_STATE.md "raw.githubusercontent.com"
[14]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/aria_nbv/data_handling/AGENTS.md "raw.githubusercontent.com"
[15]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/aria_nbv/vin/AGENTS.md "raw.githubusercontent.com"
[16]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs "ARIA-NBV/docs at main · JanDuchscherer104/ARIA-NBV · GitHub"
[17]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/contents "ARIA-NBV/docs/contents at main · JanDuchscherer104/ARIA-NBV · GitHub"
[18]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.agents/skills "ARIA-NBV/.agents/skills at main · JanDuchscherer104/ARIA-NBV · GitHub"
[19]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.agents/skills/agents-db/SKILL.md "raw.githubusercontent.com"
[20]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.agents "ARIA-NBV/.agents at main · JanDuchscherer104/ARIA-NBV · GitHub"
[21]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/SETUP.md "raw.githubusercontent.com"
[22]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.github/workflows "ARIA-NBV/.github/workflows at main · JanDuchscherer104/ARIA-NBV · GitHub"
[23]: https://github.com/JanDuchscherer104/prml-vslam/tree/main/.github/workflows "prml-vslam/.github/workflows at main · JanDuchscherer104/prml-vslam · GitHub"
[24]: https://developers.openai.com/codex/learn/best-practices?utm_source=chatgpt.com "Best practices – Codex"
[25]: https://developers.openai.com/codex/skills?utm_source=chatgpt.com "Agent Skills – Codex"
[26]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.agents/skills/aria-nbv-context/SKILL.md "raw.githubusercontent.com"
[27]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.agents/skills/graphify/SKILL.md "raw.githubusercontent.com"
[28]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.agents/skills/mempalace-repo/SKILL.md "raw.githubusercontent.com"
[29]: https://developers.openai.com/codex/guides/agents-md?utm_source=chatgpt.com "Custom instructions with AGENTS.md – Codex"
[30]: https://github.com/Yeachan-Heo/oh-my-codex?utm_source=chatgpt.com "Yeachan-Heo/oh-my-codex: OmX"
[31]: https://raw.githubusercontent.com/JanDuchscherer104/prml-vslam/main/.pre-commit-config.yaml "raw.githubusercontent.com"


[User-Turn13]
Given the current state of our repo. please give me comprehensive GH issues for all actionable items in the confines of this project.
give roadmap (till end of september), milestones, action items. and for all (GH issues!)


[Agent-Turn14]
I prepared an issue-ready roadmap and full GH issue register here:

[Download the full issue-ready roadmap + 42 GH issue bodies](sandbox:/mnt/data/aria_nbv_roadmap_issues.md)

I did **not** post these to GitHub; this is the structured issue backlog to copy into GH or feed into `gh issue create`.

## Basis for the issue plan

The repo currently exposes **1 open issue**, **5 open PRs**, and no milestones; the existing labels are only the default GitHub label set. ([GitHub][1]) The only visible workflow is `quarto-publish.yml`, so general CI/pre-commit is still missing as a first-class contract. ([GitHub][2]) The root README is still mostly a command list, while `docs/contents` contains roadmap, TODOs, questions, ideas, generated agent scaffold material, implementation notes, and public docs in one place. ([GitHub][3]) The `.agents/skills` folder exists, but the `agents-db` skill still references **PRML VSLAM** and `.agents/issues.toml`-style DB files that are not visible in the ARIA `.agents/` root. ([GitHub][4])

The research roadmap below follows the current project state: the repo is centered on oracle RRI label generation, VIN-style learned candidate scoring, diagnostics/caching, and incremental non-myopic planning; full continuous-control policy learning, entity-aware NBV, semantic planning, and deployment are extensions rather than already-finished core claims. ([GitHub][5]) This also matches the ARIA paper’s current framing: oracle RRI supervision, an offline cache, a preliminary VIN v3 candidate scorer, and future work around entity-aware objectives, planning, and deployment.

---

# Roadmap to end of September 2026

| Milestone                               |        Window | Goal                                                        | Must exit with                                                                                  |
| --------------------------------------- | ------------: | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **M0 — GH planning and scope lock**     |  Apr 27–May 3 | Convert repo slop into tracked GitHub work.                 | Milestones, labels, project board, issue templates, draft-PR disposition.                       |
| **M1 — Scaffold, CI, docs hygiene**     |  May 4–May 24 | Make the repo reliable for humans and agents.               | `make ci`, README/SETUP split, fixed agent skills, public-docs triage.                          |
| **M2 — Oracle/cache/frame correctness** | May 18–Jun 14 | Lock geometry, cache, device, and candidate contracts.      | Frame tests, cache manifest, oracle profiler, MPS/CUDA/CPU device policy.                       |
| **M3 — VIN baseline and data scaling**  |  Jun 1–Jun 30 | Make VIN v3 a measured baseline.                            | Ranking CLI, candidate shuffling, controlled ablations, expanded cache.                         |
| **M4 — Non-myopic planning baseline**   |  Jul 1–Jul 31 | Produce thesis-grade one-step vs multi-step comparisons.    | Greedy/oracle/beam/stochastic/close-greedy comparisons on fixed split.                          |
| **M5 — Entity-aware extension**         |  Aug 1–Aug 24 | Add task/entity-aware RRI without derailing the core story. | OBB-cropped entity RRI, entity selection, entity-conditioned scorer baseline.                   |
| **M6 — Experiments, thesis, release**   | Aug 25–Sep 30 | Freeze claims, write, polish, and package.                  | Claim matrix, experiment registry, final thesis/report, defense deck, reproducibility snapshot. |

---

# Proposed GH issue register

## M0 — GH planning and scope lock

|     ID | Issue title                                              | Priority | Labels                           |
| -----: | -------------------------------------------------------- | -------- | -------------------------------- |
| GH-001 | Create milestone, label, and project-board structure     | P0       | `area:scaffold`, `type:feature`  |
| GH-002 | Review and disposition all open draft PRs                | P0       | `area:scaffold`, `type:refactor` |
| GH-003 | Add issue templates, PR template, and reviewer checklist | P1       | `area:scaffold`, `area:agent`    |

**GH-001 acceptance:** create milestones M0–M6, add area/type/priority/status labels, assign existing issue #7 to a milestone, and create project-board views by milestone, area, and priority.

**GH-002 acceptance:** every open draft PR has one disposition: merge, split, close as stale, or keep with a linked tracking issue. This specifically covers PRs #3, #5, #6, #8, and #9, which are currently open drafts. ([GitHub][6])

**GH-003 acceptance:** new issue templates capture problem, affected files, acceptance criteria, and verification; PR template includes summary, affected contracts, verification, docs/memory updates, and limitations.

---

## M1 — Scaffold, CI, docs hygiene

|     ID | Issue title                                                 | Priority | Labels                       |
| -----: | ----------------------------------------------------------- | -------- | ---------------------------- |
| GH-004 | Add local CI, pre-commit, and `make ci`                     | P0       | `area:ci`, `type:feature`    |
| GH-005 | Split root README and SETUP into clear front-door docs      | P0       | `area:docs`, `area:scaffold` |
| GH-006 | Fix stale `agents-db` skill or implement the ARIA agent DB  | P0       | `area:agent`, `type:bug`     |
| GH-007 | Add ARIA-specific Codex skills for domain work              | P1       | `area:agent`, `type:feature` |
| GH-008 | Public docs triage: classify, move, archive, or delete slop | P0       | `area:docs`, `type:refactor` |
| GH-009 | Add package-level README and REQUIREMENTS contracts         | P1       | `area:docs`, `area:scaffold` |

**GH-004 acceptance:** `.pre-commit-config.yaml`, `fmt/lint/test/docs-check/paper-pdf/ci` Make targets, and `.github/workflows/ci.yml` exist; GPU/data-heavy tests are skipped by default.

**GH-005 acceptance:** root README becomes a project front door; root `SETUP.md` owns environment setup, CUDA/PyTorch/PyTorch3D, EFM3D/ATEK, data paths, cache setup, W&B, Streamlit, docs build, and common failure modes.

**GH-006 acceptance:** `agents-db` no longer says PRML VSLAM; either ARIA gets `.agents/issues.toml`, `.agents/todos.toml`, `.agents/refactors.toml`, `.agents/resolved.toml`, and `make agents-db`, or the stale skill is deleted.

**GH-007 acceptance:** add `oracle-rri-contracts`, `vin-training-diagnostics`, `counterfactual-rollout-planner`, `entity-aware-rri`, `docs-curator`, `create-pr`, and `gh-issue-lifecycle`.

**GH-008 acceptance:** every `docs/contents` page is classified as PUBLIC, PAPER, REFERENCE, PACKAGE_CONTRACT, AGENT, MEMORY, BACKLOG, GENERATED, ARCHIVE, or SCRATCHPAD; TODO/roadmap/ideas material stops living as uncurated public docs.

**GH-009 acceptance:** core modules have concise contracts for data handling, RRI metrics, rendering, pose generation, VIN, Lightning, and app surfaces.

---

## M2 — Oracle/cache/frame correctness

|     ID | Issue title                                                        | Priority | Labels                               |
| -----: | ------------------------------------------------------------------ | -------- | ------------------------------------ |
| GH-010 | Expand issue #7 into centralized device/backend selection          | P0       | `area:oracle`, `area:ci`, `type:bug` |
| GH-011 | Lock pose-frame and CW90 consistency contracts                     | P0       | `area:geometry`, `area:vin`          |
| GH-012 | Audit oracle RRI metric realism for semi-dense reconstruction      | P1       | `area:oracle`, `type:research`       |
| GH-013 | Add fine-detail supervision guardrails for mesh/point downsampling | P1       | `area:oracle`, `area:experiments`    |
| GH-014 | Candidate generation realism vs generalization experiment suite    | P1       | `area:geometry`, `area:experiments`  |
| GH-015 | Build oracle throughput profiler and optimization report           | P0       | `area:oracle`, `area:cache`          |
| GH-016 | Make `OracleRriLabeler` compatible with multi-worker data loading  | P1       | `area:oracle`, `area:cache`          |
| GH-017 | Add offline cache manifest by config hash and filtering CLI        | P0       | `area:cache`                         |
| GH-018 | Implement hybrid offline + online sample generation path           | P1       | `area:cache`, `area:lightning`       |

**GH-010 acceptance:** issue #7 gets full labels/milestone/body; central device policy covers CPU/CUDA/MPS; unsupported PyTorch3D/MPS combinations fail explicitly.

**GH-011 acceptance:** cached batches pass automated consistency checks for `candidate_poses_world_cam`, `reference_pose_world_rig`, and `p3d_cameras`; display-only CW90 correction cannot leak into render/model/cache paths.

**GH-012 acceptance:** compare current GT-depth oracle against semi-dense-like sampling on low-texture walls, clutter, occlusions, and fine details; document whether the oracle remains idealized or changes.

**GH-013 acceptance:** thesis-grade mesh/point simplification defaults are defined; aggressive downsampling raises warnings or requires explicit override.

**GH-014 acceptance:** candidate-generation profiles are versioned and compared on validity, RRI distribution, collision rate, depth-hit stats, and rank diversity.

**GH-015 acceptance:** profiler reports wall time, memory, point counts, mesh face counts, and candidate counts for candidate generation, rendering, backprojection, point↔mesh scoring, EVL forward, and cache write.

**GH-016 acceptance:** online or hybrid data loading works with `num_workers > 0`, or fails with a documented limitation and actionable error.

**GH-017 acceptance:** cache can list/filter/drop samples by `config_hash`, with a summary manifest of counts and labeler/backbone signatures.

**GH-018 acceptance:** training supports offline-only, online-only, and mixed modes, with sample provenance logged.

---

## M3 — VIN baseline and data scaling

|     ID | Issue title                                                                   | Priority | Labels                           |
| -----: | ----------------------------------------------------------------------------- | -------- | -------------------------------- |
| GH-019 | Scale oracle cache coverage within mesh-supervised ASE subset                 | P0       | `area:cache`, `area:experiments` |
| GH-020 | Shuffle candidates and labels during VIN training                             | P0       | `area:vin`, `area:lightning`     |
| GH-021 | Add standalone VIN ranking evaluation CLI                                     | P0       | `area:vin`, `area:experiments`   |
| GH-022 | Add VIN contract tests for pose descriptors, SH dimensions, and voxel queries | P1       | `area:vin`, `area:geometry`      |
| GH-023 | Run controlled VIN v3 ablation suite                                          | P0       | `area:vin`, `area:experiments`   |
| GH-024 | Stage dependence and label-calibration analysis                               | P1       | `area:vin`, `area:oracle`        |
| GH-025 | Validate CORAL variants and bin-center learning                               | P1       | `area:vin`, `area:rri`           |
| GH-026 | Add richer semi-dense visibility and directional observability features       | P2       | `area:vin`, `area:geometry`      |

**GH-019 acceptance:** cache coverage report lists scenes, snippets, candidate profiles, config hashes, and storage size; expanded split manifests are reproducible.

**GH-020 acceptance:** candidate poses, labels, masks, cameras, and diagnostics are shuffled in lockstep; deterministic seed support exists.

**GH-021 acceptance:** evaluation runs from checkpoint/cache split and reports Spearman, top-1 oracle-view hit, top-k recall, confusion, calibration, and per-stage metrics.

**GH-022 acceptance:** tests fail if pose/ref/camera frame contracts break; SH dimensions and voxel-query masks are tested.

**GH-023 acceptance:** one-factor ablations cover semidense CNN, voxel FiLM, trajectory context, observation counts, global pool resolution, coverage curriculum, CORAL variants, and auxiliary regression.

**GH-024 acceptance:** RRI distributions and bin occupancy are analyzed by stage/snippet/scene; final binning/calibration policy is selected.

**GH-025 acceptance:** fixed bin representatives, monotone learnable bin values, plain CORAL, balanced BCE, and focal thresholds are compared using ranking/calibration metrics.

**GH-026 acceptance:** observation-count features, directional visibility histograms, or candidate-visible point embeddings are optional, ablated, and documented.

---

## M4 — Non-myopic planning baseline

|     ID | Issue title                                                   | Priority | Labels                              |
| -----: | ------------------------------------------------------------- | -------- | ----------------------------------- |
| GH-027 | Define MDP contract for non-myopic planning                   | P0       | `area:planning`, `type:decision`    |
| GH-028 | Benchmark one-step greedy vs non-myopic oracle rollouts       | P0       | `area:planning`, `area:oracle`      |
| GH-029 | Stabilize cumulative-RRI rollout evaluator and plots          | P1       | `area:planning`, `area:experiments` |
| GH-030 | Stabilize discrete-shell Gymnasium / SB3 baseline             | P1       | `area:planning`                     |
| GH-031 | Validate VIN as fast surrogate reward or critic               | P1       | `area:planning`, `area:vin`         |
| GH-032 | Prototype invalid-action penalties and feasibility projection | P2       | `area:planning`, `area:geometry`    |

**GH-027 acceptance:** state/action/transition/reward/termination contract exists; actor-visible, critic-visible, and oracle-only signals are separated.

**GH-028 acceptance:** greedy, oracle beam, stochastic/Gumbel beam, random feasible, and coverage baselines are compared with cumulative RRI, final Chamfer components, path feasibility, and diversity.

**GH-029 acceptance:** rollout results export to CSV/JSON and generate cumulative-RRI, trajectory, selected-candidate, validity, and final-quality plots.

**GH-030 acceptance:** discrete-shell RL runs reproducibly on a fixed cache subset and is explicitly framed as exploratory unless evidence meets the claim bar.

**GH-031 acceptance:** VIN-selected rollouts are oracle-evaluated; rank errors and calibration failures are reported before any surrogate-reward claim.

**GH-032 acceptance:** hard filtering, invalid class/penalty, and feasibility projection are compared; invalid handling is documented for future continuous control.

---

## M5 — Entity-aware extension

|     ID | Issue title                                                        | Priority | Labels                                           |
| -----: | ------------------------------------------------------------------ | -------- | ------------------------------------------------ |
| GH-033 | Implement entity-aware RRI with OBB-cropped mesh and points        | P0       | `area:entity`, `area:oracle`                     |
| GH-034 | Add snippet-level object/entity selection policy                   | P1       | `area:entity`                                    |
| GH-035 | Add entity-conditioned VIN scorer baseline                         | P1       | `area:entity`, `area:vin`                        |
| GH-036 | Extend Streamlit diagnostics for entity-aware and rollout analysis | P2       | `area:streamlit`, `area:entity`, `area:planning` |
| GH-037 | Decide and scope SceneScript / semantic-global planner integration | P2       | `area:entity`, `area:planning`, `type:decision`  |

**GH-033 acceptance:** `RRI_e(q)` computes object-level accuracy/completeness/RRI using OBB-cropped mesh, current points, and candidate points; global and entity metrics are logged separately.

**GH-034 acceptance:** entity selection uses visibility, distance, OBB size, category, current completeness, and candidate visibility; metadata is stored in cache/eval outputs.

**GH-035 acceptance:** scorer supports GT OBB conditioning at minimum; entity Spearman/top-k and scene metrics are reported separately.

**GH-036 acceptance:** Streamlit can visualize selected entities, OBB crops, per-entity RRI, candidate rankings, and rollout trajectories from saved artifacts.

**GH-037 acceptance:** SceneScript is either scoped as backbone/baseline/entity prior or deferred; if semantic-global planning is pursued, outputs are grounded subgoals, not raw continuous poses. SceneScript is especially relevant as an extensible structured-scene representation, but should remain a bounded extension unless it directly supports the thesis claim.

---

## M6 — Experiments, thesis, release

|     ID | Issue title                                         | Priority | Labels                              |
| -----: | --------------------------------------------------- | -------- | ----------------------------------- |
| GH-038 | Create experiment registry and run-report generator | P0       | `area:experiments`, `area:paper`    |
| GH-039 | Lock thesis scope and final claim matrix            | P0       | `area:paper`, `type:decision`       |
| GH-040 | Final paper/thesis docs cleanup and typo pass       | P0       | `area:paper`, `area:docs`           |
| GH-041 | Build final defense slide deck and demo script      | P1       | `area:paper`, `area:docs`           |
| GH-042 | Final release and reproducibility checklist         | P1       | `area:scaffold`, `area:experiments` |

**GH-038 acceptance:** every final figure/table maps to run ID, config hash, cache hash, split, candidate profile, metrics, and source artifact.

**GH-039 acceptance:** final claim matrix lists claim, evidence, metric, figure/table, and status; no claims about continuous control, RL, deployment, or VLMs are made without evidence.

**GH-040 acceptance:** Typst paper and Quarto docs build cleanly; typos, duplicate “Section Section” references, stale paths, TODO prose, and placeholder citations are gone.

**GH-041 acceptance:** final slide deck builds from Typst; demo can run from saved cache artifacts; backup slides cover frame geometry, cache, oracle, and planning details.

**GH-042 acceptance:** final tag/branch, dependency lock state, configs, cache manifests, split manifests, and reproducibility README are present.

---

# Recommended dependency order

Start with **GH-001 → GH-002 → GH-004 → GH-005 → GH-006 → GH-008**. Those unlock sane collaboration and prevent new slop. Then run **GH-010 → GH-011 → GH-015 → GH-017**, because device/frame/cache correctness must be stable before scaling experiments. The first real research block should be **GH-019 → GH-020 → GH-021 → GH-023 → GH-028**. Entity-aware work should wait until the one-step/non-myopic baseline is stable, except for scoping decisions in **GH-033/GH-037**.

[1]: https://github.com/JanDuchscherer104/ARIA-NBV/issues "Issues · JanDuchscherer104/ARIA-NBV · GitHub"
[2]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.github/workflows "ARIA-NBV/.github/workflows at main · JanDuchscherer104/ARIA-NBV · GitHub"
[3]: https://github.com/JanDuchscherer104/ARIA-NBV "GitHub - JanDuchscherer104/ARIA-NBV · GitHub"
[4]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.agents/skills "ARIA-NBV/.agents/skills at main · JanDuchscherer104/ARIA-NBV · GitHub"
[5]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.agents/memory/state/PROJECT_STATE.md "raw.githubusercontent.com"
[6]: https://github.com/JanDuchscherer104/ARIA-NBV/pulls "Pull requests · JanDuchscherer104/ARIA-NBV · GitHub"


[User-Turn15]
Please employ ruthless $simplification and identify all slop and targets for decluttering, removing noise in both the docs and the aria_nbv package!

[Agent-Turn16]

## Ruthless simplification thesis

The repo has **one coherent thesis spine** and too many parallel surfaces around it. The spine is:

> **Oracle RRI supervision → cached candidate labels → VIN v3 candidate scorer → ranking/planning diagnostics → thesis/report figures.**

Everything else should be treated as noise unless it directly improves **oracle correctness**, **VIN ranking evidence**, **non-myopic planning evaluation**, or **reproducibility**. This matches the project’s own current framing: it is not yet a full end-to-end NBV policy, but an oracle RRI labeling pipeline plus a preliminary VIN v3 candidate scorer and diagnostics stack.  The live repo also shows symptoms of sprawl: a tiny command-list root README, many docs branches, tracked logs/checkpoints, many legacy data modules, public agent scaffold pages, and broad future-work surfaces. ([GitHub][1])

My rule: **delete what is stale, archive what is historically useful, merge what duplicates, and keep only what can appear in the final thesis or reproduce a final figure.**

---

# 1. Keep: the minimal project spine

## Code that should survive as first-class

Keep these as the core `aria_nbv` package:

```text
aria_nbv/
  data_handling/        # canonical ASE/EFM snippets, offline cache, VinOracleBatch
  pipelines/            # OracleRriLabeler and cache-building orchestration
  pose_generation/      # candidate pose generation + pruning
  rendering/            # PyTorch3D depth rendering + backprojection
  rri_metrics/          # Chamfer/RRI/CORAL/binning metrics
  vin/                  # VIN v3 scorer only, plus required feature modules
  lightning/            # training/eval harness, if it stays thin
  app/ or streamlit     # diagnostics only, not core research logic
```

The package already implies this shape: `aria_nbv/aria_nbv` contains `data_handling`, `pipelines`, `pose_generation`, `rendering`, `rri_metrics`, `vin`, `lightning`, app surfaces, and utilities. ([GitHub][2]) The project-state file also says the near-term priorities are oracle RRI correctness, bounded VIN improvements, planning rollouts, and docs alignment—not broad RL, semantic planning, or deployment. ([GitHub][3])

## Docs that should survive as first-class

Keep only these public docs:

```text
docs/
  index.qmd                 # short landing page
  setup.qmd                 # environment + data + cache setup
  contents/glossary.qmd     # terms and coordinate conventions
  contents/ase_dataset.qmd  # dataset facts
  contents/experiments/final_results.qmd
  contents/impl/oracle_rri.qmd
  contents/impl/offline_cache.qmd
  contents/impl/vin_v3.qmd
  contents/impl/planning_eval.qmd
  contents/impl/frame_contracts.qmd
  reference/                # generated API docs, only if maintained
  typst/paper/              # thesis/paper source
```

The docs’ own `AGENTS.md` says the Typst paper should be the highest-level ground truth and that Quarto should link to canonical state docs rather than re-explaining internal guidance; it also says generated context/rendered artifacts should not be stored unless explicitly required. ([GitHub][4]) Use that instruction ruthlessly.

---

# 2. Delete or untrack immediately

## P0 deletion targets

| Target                                                                                 | Action                                                          | Why                                                                                                                                                                |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `.logs/checkpoints/*.ckpt`                                                             | **Untrack from Git; move to artifact storage or W&B**           | These are model artifacts, not source. They are tracked through LFS despite `.gitignore` ignoring `.logs/*`. ([GitHub][5])                                         |
| `.logs/ckpts/*.pth`, `.logs/ckpts/*.ckpt`                                              | **Untrack from Git; replace with download instructions**        | Pretrained weights and external checkpoints should not be in the repo tree. ([GitHub][6])                                                                          |
| `docs/literature/tex-src/.DS_Store`                                                    | **Delete**                                                      | Pure OS trash in docs. ([GitHub][7])                                                                                                                               |
| `docs/literature/tex-src/*` extracted arXiv sources                                    | **Archive outside public docs or delete**                       | Keep summaries/citations, not full extracted source trees. They add noise and confuse public docs. ([GitHub][7])                                                   |
| `docs/_freeze/contents/**`                                                             | **Delete unless Quarto requires it for public reproducibility** | Frozen execution artifacts are generated outputs, not core source. ([GitHub][8])                                                                                   |
| Empty `docs/_generated/context` public surface                                         | **Delete links or regenerate properly**                         | The directory only contains `.gitignore`, while root agent instructions refer to generated context files. ([GitHub][9])                                            |
| `docs/contents/resources/agent_scaffold/**` from public nav                            | **Remove from public docs**                                     | The Quarto nav links agent scaffold pages, but the directory is empty except `.gitignore`; agent scaffolding belongs in `.agents`, not public docs. ([GitHub][10]) |
| `docs/contents/ideas.qmd`, `todos.qmd`, `questions.qmd`, `roadmap.qmd` as public pages | **Move to `.agents/memory/state` or `docs/archive`**            | These are internal planning artifacts. Public docs should present stable results and contracts, not wandering backlog. ([GitHub][11])                              |
| `docs/contents/repo_structure.qmd`                                                     | **Delete or regenerate on demand**                              | Repo-structure docs rot immediately. Use tree output in README or generated docs only. ([GitHub][11])                                                              |

---

# 3. Archive, don’t keep in the active surface

These are not necessarily worthless, but they should not be in the main mental path.

| Target                                                             | Action                                                                 | Why                                                                                                                                                             |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docs/contents/impl/vin_v2_*`                                      | Move to `docs/archive/vin_v2/`                                         | VIN v3 is the current baseline. v2 design pages create conceptual drag. ([GitHub][12])                                                                          |
| `docs/contents/impl/optuna_vin_v2_searchspace_2026-01-07.qmd`      | Move to experiment archive                                             | Date-specific sweep notes are historical, not public implementation docs. ([GitHub][12])                                                                        |
| `docs/contents/ext-impl/*`                                         | Collapse into one “external dependencies” reference page               | External implementation notes are useful, but four separate pages are too much surface. ([GitHub][13])                                                          |
| `docs/contents/resources/agent_scaffold/*`                         | Move fully into `.agents/`                                             | Agent internals should not be part of user-facing docs. ([GitHub][14])                                                                                          |
| `docs/contents/theory/*`                                           | Keep only if cited by thesis; otherwise archive                        | Theory pages easily become detached literature notes.                                                                                                           |
| `docs/contents/literature/*`                                       | Collapse to one annotated bibliography                                 | The final paper/thesis should own literature claims.                                                                                                            |
| `docs/contents/diagrams.qmd`                                       | Keep only if all diagrams are final/public                             | Otherwise split final figures from scratch diagrams.                                                                                                            |
| `external/scenescript`                                             | Archive/de-submodule unless actively used in a reproducible experiment | SceneScript is relevant as a structured semantic representation, but it is future-extension material unless it directly powers a final entity-aware NBV result. |
| `external/PointNeXt`                                               | Remove unless PointNeXt ablation is actually run and reported          | Current VIN v3 baseline explicitly has PointNeXt off.                                                                                                           |
| `external/openpoints_shim`                                         | Remove with PointNeXt unless needed by active code                     | A shim for an inactive ablation is maintenance drag.                                                                                                            |
| `.agents/references`, `.agents/memory` docs exposed through Quarto | Keep private, not public                                               | Useful for agents, noisy for readers.                                                                                                                           |

---

# 4. Fix naming and navigation slop

## Repo identity is inconsistent

The root README says **“Seminar: Next Best View Estimation (NBV)”**, while the repo is **ARIA-NBV**, and the Quarto website title is **“Semantic NBV Planning.”** ([GitHub][1]) Pick one public name:

> **ARIA-NBV: Quality-Driven Egocentric Next-Best-View Planning**

Then apply it everywhere: root README, Quarto title, paper title, package docs, GitHub Pages, issue templates.

## Quarto links point to the wrong repo

`docs/_quarto.yml` links to `JanDuchscherer104/NBV`, while the current repo is `JanDuchscherer104/ARIA-NBV`. ([GitHub][14]) This is a P0 cleanup item. Broken identity links make the project look stale.

## Root README is not a README

The current root README is essentially a packed command list with cache and downloader commands. ([GitHub][1]) Replace it with:

```text
README.md
  1. What this project is
  2. What is currently implemented
  3. What is not claimed
  4. Quickstart
  5. Reproduce final cache/training/eval
  6. Repo map
  7. Data/model artifact policy
```

Move the long command block into `SETUP.md` or `docs/contents/setup.qmd`.

---

# 5. Package decluttering targets

## 5.1 `aria_nbv.data` should die or become a tiny compatibility shim

The package itself says `aria_nbv.data` is deprecated and that `aria_nbv.data_handling` is the canonical owner. ([GitHub][15]) The cleanup action is simple:

```text
DELETE / ARCHIVE:
  aria_nbv/data/*

KEEP ONLY IF NEEDED:
  aria_nbv/data/__init__.py
    -> raises clear deprecation error or re-exports minimal stable names
```

Do not keep residual utilities in both places. Every data/cache/snippet import should go through `aria_nbv.data_handling`.

## 5.2 Remove `_legacy_*` after one import-graph pass

`data_handling` contains a large pile of `_legacy_*` modules and the README explicitly marks a cutover sweep with `NBV_LEGACY_OFFLINE_CACHE_REMOVE_AFTER_FULL_MIGRATION`. ([GitHub][16]) This is prime slop.

Delete candidates:

```text
aria_nbv/data_handling/_legacy_cache_api.py
aria_nbv/data_handling/_legacy_dataset_mixins.py
aria_nbv/data_handling/_legacy_offline_cache_coverage.py
aria_nbv/data_handling/_legacy_offline_cache_serialization.py
aria_nbv/data_handling/_legacy_offline_cache_store.py
aria_nbv/data_handling/_legacy_oracle_cache.py
aria_nbv/data_handling/_legacy_vin_cache.py
aria_nbv/data_handling/_legacy_vin_provider.py
aria_nbv/data_handling/_legacy_vin_source.py
```

Target state:

```text
data_handling/
  __init__.py
  efm_dataset.py
  efm_snippet_loader.py
  efm_views.py
  vin_oracle_types.py
  vin_oracle_datasets.py
  cache_contracts.py
  cache_index.py
  offline_cache_store.py
  offline_cache_serialization.py
  offline_cache_coverage.py
  oracle_cache.py
  vin_cache.py
  vin_provider.py
  mesh_cache.py
```

Everything else should justify itself.

## 5.3 Kill hard-coded local paths

`data_handling/README.md` includes hard-coded `/home/jandu/repos/NBV/...` paths in migration/cutover notes. ([GitHub][16]) These should never appear in package docs. Replace with environment variables:

```bash
ARIA_NBV_CACHE_ROOT=/path/to/cache
ARIA_NBV_DATA_ROOT=/path/to/ase
ARIA_NBV_OUTPUT_ROOT=/path/to/runs
```

## 5.4 Collapse plotting sprawl

Plotting helpers appear in multiple modules:

```text
pose_generation/plotting.py
rendering/plotting.py
rri_metrics/plotting.py
utils/plotting.py
utils/data_plotting.py
vin/plotting.py
vin/_plotting_common.py
scripts/plot_vin_*.py
```

The package tree confirms this spread. ([GitHub][17]) Collapse into:

```text
aria_nbv/viz/
  oracle.py
  candidates.py
  vin.py
  cache.py
```

Then make scripts call these helpers. No plotting logic should live in six places.

## 5.5 Move one-off scripts out of package source

`aria_nbv/scripts` contains one-off generation, plotting, repair, summarization, and training scripts. ([GitHub][18]) Keep only stable CLI entrypoints. Everything else moves to `tools/` or `experiments/scripts/archive/`.

Keep:

```text
aria_nbv/scripts/train_vin_lightning.py        # only if console entrypoint uses it
aria_nbv/scripts/run_efm3d_on_ase.py           # only if reproducibility requires it
aria_nbv/scripts/export_paper_figures.py       # only if final figures use it
```

Archive/delete:

```text
generate_vin_v2_arch.py
plot_vin_encodings.py
plot_vin_inputs.py
plot_vin_binning.py
summarize_vin.py
get_context.py
fix_vin_snippet_lengths.py  # unless needed as a migration tool, then move to tools/migrations/
```

## 5.6 `vin/experimental` should not be in the active package

The `vin` package has an `experimental` directory plus v3 code. ([GitHub][19]) If v3 is the baseline, then:

```text
KEEP:
  vin/model_v3.py
  vin/backbone_evl.py
  vin/pose_encoders.py
  vin/pose_encoding.py
  vin/semidense_projection.py
  vin/types.py
  vin/vin_modules.py

ARCHIVE:
  vin/experimental/**
  vin/traj_encoder.py   # if disabled and not part of final ablation
  vin/summarize_v3.py   # move to scripts/tools if needed
```

The current paper describes VIN v3 as the implemented scorer and treats several architecture toggles as planned or uncontrolled ablations, so experimental variants should not sit in the mainline namespace.

## 5.7 `interpretability` and Captum are likely premature

`pyproject.toml` includes Captum and many experiment/diagnostic dependencies. ([GitHub][20]) Unless interpretability produces a final thesis figure, move it to:

```text
experiments/interpretability/
```

Do not keep interpretability as a top-level production package surface.

## 5.8 `utils` is a junk drawer

`utils` contains config, console, plotting, schemas, W&B, summaries, reporting, grad norms, typed payloads, and visualization helpers. ([GitHub][21]) Split it or kill it.

Target:

```text
aria_nbv/config.py
aria_nbv/logging.py
aria_nbv/types.py
aria_nbv/viz/
aria_nbv/experiment_tracking/
```

Delete generic names like `summary.py`, `reporting.py`, `viz_utils.py`, `data_plotting.py` unless they have one stable owner.

## 5.9 Markdown inside source package should move

`rri_metrics/context_pytorch_3d_losses.md` lives inside `aria_nbv/aria_nbv/rri_metrics`. ([GitHub][22]) Move it to docs or delete it. Source packages should not contain context notes unless they are packaged reference assets intentionally.

---

# 6. Docs decluttering targets

## 6.1 Collapse public docs to five lanes

Current `docs/contents` mixes implementation docs, literature, resources, agent scaffolding, theory, roadmap, TODOs, ideas, questions, and setup. ([GitHub][11]) Collapse to:

```text
docs/contents/
  setup.qmd
  glossary.qmd
  ase_dataset.qmd

  impl/
    oracle_rri.qmd
    offline_cache.qmd
    vin_v3.qmd
    planning_eval.qmd
    frame_contracts.qmd

  experiments/
    final_results.qmd
    ablations.qmd
    failure_cases.qmd

  reference/
    api.qmd
```

Everything else is archive or internal.

## 6.2 Remove public “Project State” menus

The Quarto navbar exposes Roadmap, Ideas, Questions, and Findings under “Project State.” ([GitHub][14]) That is internal project management, not documentation. Move to `.agents/memory/state` or GitHub issues.

Public docs should say:

```text
What is implemented?
How do I run it?
What results are final?
What are the known limitations?
```

Not:

```text
What were we thinking about last week?
What might be interesting someday?
What did agents generate?
```

## 6.3 Fix typo/stale filenames

Immediate renames:

```text
docs/contents/impl/coral_intergarion.qmd
  -> docs/contents/impl/coral_integration.qmd

docs/contents/ext-impl/prj_aria_tools_impl.qmd
  -> docs/contents/ext_impl/projectaria_tools.qmd

docs/contents/impl/data_pipeline_overview.quarto_ipynb
  -> docs/contents/impl/data_pipeline_overview.qmd
```

The typo and `.quarto_ipynb` appear in the live docs tree. ([GitHub][12])

## 6.4 Do not track generated notebooks

`.gitignore` already ignores `*.quarto_ipynb`, yet a `.quarto_ipynb` appears in docs. ([GitHub][23]) Remove it from the repo and keep the source `.qmd`.

## 6.5 The docs landing page is trying to be a paper, README, and project-state memo

`docs/index.qmd` currently contains project overview, current state, done/WIP/future work, implementation links, agent links, and typos. ([GitHub][24]) Replace it with:

```markdown
# ARIA-NBV

One sentence mission.

## Current implemented scope
- Oracle RRI labels
- Offline cache
- VIN v3 candidate scorer
- Ranking/planning diagnostics

## Not claimed
- No full end-to-end action policy yet
- No real-device deployment claim
- No SceneScript semantic planner claim

## Start here
- Setup
- Reproduce final experiments
- Read the paper
```

## 6.6 Keep SceneScript out of the core docs unless used

SceneScript is valuable because it provides compact structured scene commands and extensible semantic/entity representations.  But for this project, it should be a **future extension** or **external reference**, not a core lane, unless there is an implemented experiment connecting SceneScript outputs to entity-aware RRI or planning.

---

# 7. External dependencies: prune aggressively

The `external` directory contains ATEK, PointNeXt, EFM3D, openpoints shim, Project Aria Tools, and SceneScript. ([GitHub][25]) Keep only what is needed to run the final thesis pipeline.

| External dependency          |                              Keep? | Action                                                                       |
| ---------------------------- | ---------------------------------: | ---------------------------------------------------------------------------- |
| `external/efm3d`             |                                Yes | Core EVL backbone dependency.                                                |
| `external/ATEK`              | Yes, if data loading depends on it | Keep as pinned submodule or document install.                                |
| `external/projectaria_tools` |                              Maybe | Keep only if local source checkout is needed. Otherwise use package install. |
| `external/scenescript`       |                        No for core | Move to archive/reference unless active entity-aware experiment exists.      |
| `external/PointNeXt`         |                        No for core | Remove unless final ablation uses it.                                        |
| `external/openpoints_shim`   |                        No for core | Remove with PointNeXt.                                                       |

---

# 8. Target simplified repo tree

This is the shape I would aim for:

```text
ARIA-NBV/
  README.md
  SETUP.md
  Makefile
  pyproject.toml
  .github/
    workflows/
      ci.yml
      quarto-publish.yml

  aria_nbv/
    aria_nbv/
      __init__.py
      config.py
      types.py

      data_handling/
      pipelines/
      pose_generation/
      rendering/
      rri_metrics/
      vin/
      lightning/
      viz/

    scripts/
      train_vin_lightning.py
      export_paper_figures.py
      run_efm3d_on_ase.py

  docs/
    index.qmd
    contents/
      setup.qmd
      glossary.qmd
      ase_dataset.qmd
      impl/
        oracle_rri.qmd
        offline_cache.qmd
        vin_v3.qmd
        planning_eval.qmd
        frame_contracts.qmd
      experiments/
        final_results.qmd
        ablations.qmd
        failure_cases.qmd
    typst/
      paper/

  tools/
    migrations/
    dev/

  experiments/
    archived/
    scratch/

  external/
    efm3d/
    ATEK/
```

Everything not fitting this structure should be forced to answer: **Does it support a final figure, a final claim, or a reproducibility contract?** If not, archive or delete.

---

# 9. Issue-ready cleanup backlog

## P0 — stop the bleeding

1. **Untrack checkpoint/model artifacts from `.logs`**

   * Remove `.logs/checkpoints/*.ckpt` and `.logs/ckpts/*` from Git/LFS.
   * Add artifact download instructions.
   * Ensure `.gitignore` and `.gitattributes` agree.

2. **Replace root README with real project front door**

   * State current implemented scope.
   * State non-claims.
   * Move command dump to `SETUP.md`.

3. **Fix repo identity and docs links**

   * Rename docs title from “Semantic NBV Planning” to ARIA-NBV.
   * Replace stale `/NBV` GitHub links with `/ARIA-NBV`.

4. **Remove public agent scaffold and empty generated-context pages**

   * Delete Quarto nav/sidebar entries for generated agent scaffold.
   * Keep agent memory under `.agents`, not docs.

5. **Archive roadmap/ideas/TODO/questions public docs**

   * Move to `.agents/memory/state` or GitHub issues.
   * Keep only final public roadmap if needed.

6. **Delete tracked generated/frozen docs artifacts**

   * Remove `docs/_freeze/**` unless required for Quarto publishing.
   * Remove `*.quarto_ipynb`.

## P1 — simplify package internals

7. **Finish `aria_nbv.data` → `aria_nbv.data_handling` migration**

   * Delete deprecated data package or reduce to explicit compatibility shim.

8. **Delete `_legacy_*` data-handling modules**

   * Run import graph.
   * Replace remaining callers.
   * Remove all `_legacy_*` modules in one sweep.

9. **Collapse plotting into `aria_nbv.viz`**

   * Move module-specific plotting logic into a single visualization namespace.
   * Convert plotting scripts into CLI calls.

10. **Archive inactive VIN experiments**

* Move `vin/experimental/**` and VIN v2 docs to archive.
* Keep VIN v3 as the only active architecture.

11. **Move one-off scripts to `tools/` or archive**

* Keep only stable CLI-backed scripts in package source.

12. **Split or delete `utils` junk drawer**

* Move W&B, plotting, reporting, schema, and console helpers to named owners.

## P2 — polish and thesis alignment

13. **Collapse external implementation docs**

* One external dependencies page, not many implementation archaeology pages.

14. **Prune external submodules**

* Keep EFM3D/ATEK.
* Remove PointNeXt/openpoints/SceneScript unless a final experiment depends on them.

15. **Fix docs typos and stale filenames**

* `coral_intergarion.qmd` → `coral_integration.qmd`.
* Remove typo-heavy prose from landing page.
* Make paper and docs terminology match.

16. **Create final claim matrix**

* Every claim maps to a run, cache hash, figure, table, or code path.
* Anything without evidence becomes “future work.”

---

# 10. The most important cut

The highest-leverage simplification is this:

> **Stop presenting the repo as a broad semantic/agent/NBV platform. Present it as a thesis-grade oracle-RRI + VIN-v3 + planning-evaluation system.**

That means SceneScript, LLM/VLA, real-device deployment, PointNeXt, broad RL, full continuous planning, and human-in-the-loop UI should all move out of the active path unless they are implemented, evaluated, and tied to final claims. The project’s own paper and state docs already say the implemented contribution is narrower and stronger: oracle RRI labels, offline cache, VIN v3 scoring, and diagnostics.

[1]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/README.md "raw.githubusercontent.com"
[2]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv "ARIA-NBV/aria_nbv/aria_nbv at main · JanDuchscherer104/ARIA-NBV · GitHub"
[3]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.agents/memory/state/PROJECT_STATE.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/AGENTS.md "raw.githubusercontent.com"
[5]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.logs/checkpoints "ARIA-NBV/.logs/checkpoints at main · JanDuchscherer104/ARIA-NBV · GitHub"
[6]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/.logs/ckpts "ARIA-NBV/.logs/ckpts at main · JanDuchscherer104/ARIA-NBV · GitHub"
[7]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/literature/tex-src "ARIA-NBV/docs/literature/tex-src at main · JanDuchscherer104/ARIA-NBV · GitHub"
[8]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/_freeze/contents "ARIA-NBV/docs/_freeze/contents at main · JanDuchscherer104/ARIA-NBV · GitHub"
[9]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/_generated/context "ARIA-NBV/docs/_generated/context at main · JanDuchscherer104/ARIA-NBV · GitHub"
[10]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/contents/resources/agent_scaffold "ARIA-NBV/docs/contents/resources/agent_scaffold at main · JanDuchscherer104/ARIA-NBV · GitHub"
[11]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/contents "ARIA-NBV/docs/contents at main · JanDuchscherer104/ARIA-NBV · GitHub"
[12]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/contents/impl "ARIA-NBV/docs/contents/impl at main · JanDuchscherer104/ARIA-NBV · GitHub"
[13]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/docs/contents/ext-impl "ARIA-NBV/docs/contents/ext-impl at main · JanDuchscherer104/ARIA-NBV · GitHub"
[14]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/_quarto.yml "raw.githubusercontent.com"
[15]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/data "ARIA-NBV/aria_nbv/aria_nbv/data at main · JanDuchscherer104/ARIA-NBV · GitHub"
[16]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/data_handling "ARIA-NBV/aria_nbv/aria_nbv/data_handling at main · JanDuchscherer104/ARIA-NBV · GitHub"
[17]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/pose_generation "ARIA-NBV/aria_nbv/aria_nbv/pose_generation at main · JanDuchscherer104/ARIA-NBV · GitHub"
[18]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/scripts "ARIA-NBV/aria_nbv/scripts at main · JanDuchscherer104/ARIA-NBV · GitHub"
[19]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/vin "ARIA-NBV/aria_nbv/aria_nbv/vin at main · JanDuchscherer104/ARIA-NBV · GitHub"
[20]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/aria_nbv/pyproject.toml "raw.githubusercontent.com"
[21]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/utils "ARIA-NBV/aria_nbv/aria_nbv/utils at main · JanDuchscherer104/ARIA-NBV · GitHub"
[22]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/aria_nbv/aria_nbv/rri_metrics "ARIA-NBV/aria_nbv/aria_nbv/rri_metrics at main · JanDuchscherer104/ARIA-NBV · GitHub"
[23]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/.gitignore "raw.githubusercontent.com"
[24]: https://raw.githubusercontent.com/JanDuchscherer104/ARIA-NBV/main/docs/index.qmd "raw.githubusercontent.com"
[25]: https://github.com/JanDuchscherer104/ARIA-NBV/tree/main/external "ARIA-NBV/external at main · JanDuchscherer104/ARIA-NBV · GitHub"
