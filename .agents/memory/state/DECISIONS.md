---
id: decisions
updated: 2026-05-05
scope: repo
owner: jan
status: active
tags: [codex, workflow, architecture]
---

# Decisions

## Durable Repo Decisions
- Codex repo guidance uses repo-root and nested `AGENTS.md` files, not `.codex/AGENTS.md`.
- Repo skills live in `.agents/skills/` and use progressive disclosure.
- Shared repo guidance must stay machine-portable; operator-specific interpreter or host paths belong in `.agents/references/` or user-local notes, not repo-root or nested `AGENTS.md` files.
- Canonical project memory lives in `.agents/memory/state/`; episodic notes live in `.agents/memory/history/`.
- Generated context is derived output under `docs/_generated/context/` and should remain untracked.
- `make context` is the lightweight scaffold refresh for `source_index.md`, `literature_index.md`, and `data_contracts.md`.
- `make context-heavy` is explicit fallback for bundled heavy artifacts such as UML, bulk docstrings, and directory trees.
- `docs/_generated/context/source_index.md` is a compact routing index; broad file inventories stay discoverable through commands, not the hot path.
- The Codex hot path stays limited to `docs/typst/seminar_paper/main.typ` + `.agents/memory/state/` + `docs/_generated/context/source_index.md`.
- Progressive disclosure routes from the root `AGENTS.md` into package, docs, and module-specific guides only after the touched surface is localized; agents should not load all nested guides up front.
- `.agents/references/` holds operator aids and long-form conventions; those docs are on-demand references, not default bootstrap context.
- `docs/typst/seminar_paper/main.typ` is the highest-level project truth when it disagrees with Quarto summaries.
- Native debriefs under `.agents/memory/history/` must include `canonical_updates_needed`; existing `status: legacy-imported` notes are grandfathered archive evidence.
- Ad hoc `.codex/*.md` notes are invalid; migrate them into `.agents/memory/history/` or archive them under `archive/codex-legacy/`.
- Verification in shared repo guidance is selected by touched surface rather than by a single global checklist.

## Technical Decisions
- Runtime objects are instantiated through config `.setup_target()` factories.
- Pose and camera representations use `PoseTW` and `CameraTW`.
- Package verification uses `ruff format`, `ruff check`, and targeted `pytest`.
- The tracked Python workspace and package root are `aria_nbv/` and `aria_nbv/aria_nbv`; repo tooling and docs should refer to that layout.
- Documentation changes should update Quarto/Typst sources directly, not ad hoc notes under `.codex/`.
- The published Quarto site refreshes `aria_nbv` API reference pages from docstrings via `quartodoc` during the Pages workflow, with `docs/reference/index.qmd` as the human-authored landing page.
- Generated agent-scaffold pages are internal operator artifacts under
  `.agents/generated/agent_scaffold/`; the published Quarto site must not
  regenerate or render them.
- Retained QMD docs remain renderable, but current thesis pages use
  `docs/contents/thesis/`, past seminar material uses `docs/contents/seminar/`,
  and only curated public archive summaries use `docs/contents/archive/` with
  explicit `phase`, `audience`, `status`, and `owner` frontmatter. Raw
  scratch/history belongs under `.agents/archive/docs/`.
- Human-owner preferences that are durable but not public narrative or workflow
  rules live in `.agents/references/human_owner_intent.md`.
- OMX remains optional operator orchestration; it does not own canonical memory,
  backlog, docs, or repo state.

## Working Project Decisions
- The thesis/system name is **ARIA-NBV**.
- The core thesis claim is target-conditioned, quality-driven NBV on ASE/EFM with strict M1 data/cache/oracle contracts.
- RRI is the primary project objective for next-best-view research in this repo. Coverage-style objectives remain baselines or diagnostics, not the main thesis target.
- The canonical training and evaluation surface remains finite candidate ranking/selection anchored on prerecorded ASE trajectory snippets with oracle supervision derived from GT meshes where available.
- The thesis-core simulator substrate is the ASE mesh/oracle counterfactual rollout loop. Habitat, Isaac, external datasets, online Gymnasium/SB3, continuous actor-critic policies, SceneScript, VLM planning, and real-device guidance are stretch or bridge work unless later evidence changes the scope.
- The proposal/thesis boundary has four tiers: implemented substrate; prerequisite core covering proposal freeze, M1, V0/V1 target contracts, rollout/Q storage, and LRZ gates; hard thesis core covering observed target selection, target-conditioned scorer/Q_H, greedy/softmax/Gumbel rollouts, fitted Double-Q, and full-scale generation; stretch covering IQL second ablation, actor-critic bridge, continuous policies, external simulators, SceneScript, and real-device guidance.
- Proposal freeze precedes M1 scale-up. The proposal must use primary metadata in the bibliography, remove generated/Wikipedia citation slop, render with `make proposal-pdf`, and track the generated proposal PDF once frozen.
- M1 is a hard stop before target/RL scaling: offline store, split, frame/CW90, candidate-label ordering, depth/backprojection, Rerun normal/boundary/failure recordings, and oracle throughput evidence must be reported in a public M1 contract artifact or explicit blockers.
- The final experiment scale bar is full 100 GT-mesh ASE scenes and 4,608 snippet windows after small-subset correctness. Final supervision coverage must be reported exactly, and train/validation/test boundaries must be scene-level; the exact held-out test split remains an advisor decision.
- LRZ deterministic sharding, Slurm/DSS staging, resumable writes, storage budgeting, and Zarr-first rollout/Q schema are hard M2/M3 gates before full-scale target/RL generation. No workstation should be assumed as the thesis scale path.
- The target contract separates actor input from oracle labels: V0 uses GT OBB input plus GT crop/evaluation as a sanity/upper-bound path; V1 is mandatory for the main result and uses observed/predicted OBB inputs matched to GT-OBB target-RRI labels.
- The main target protocol is OBS-SEL / PRED-Q / GT-EVAL: observed-only target selection, predicted/observed target-conditioned scoring or Q_H, and GT target-crop evaluation.
- Automatic target selection is mandatory and actor-visible only: use predicted OBBs/classes/confidence, projected area, current semidense/EVL point support, and related observed signals. GT is label/evaluation only.
- Target matching starts with compatible class, OBB IoU, visibility/support, projected area, and semidense/EVL point support. Extra criterion `X` is deferred.
- Candidate generation is a mandatory mixed observed candidate set with categorical-probability strategy hyperparameters. The thesis-core vocabulary follows the current ARIA-NBV generator modes: `TARGET_POINT`, `RADIAL_AWAY`, `RADIAL_TOWARDS`, `FORWARD_RIG`, `UNIFORM_SPHERE`, `FORWARD_POWERSPHERICAL`, and bounded view jitter. Frontier or missing-surface samplers are stretch unless implemented and validated.
- RQ4 is a target-conditioned one-step scoring question only. Fitted Double-Q / Q_H policy success belongs to RQ5, where Q_H must beat one-step greedy/model scoring on cumulative target RRI under equal acquisition budget.
- Invalidity is a hard mask plus explicit reason-code contract, not a low RRI class. Validity heads or scalar penalty rewards are ablations after masks/reasons are reliable.
- The first thesis-grade non-myopic comparison is bounded oracle-RRI lookahead versus one-step greedy under equal acquisition and candidate budget; bounded oracle lookahead is an upper-bound baseline for learned Q_H.
- Rollout data must include random-valid, one-step greedy/model scoring, bounded oracle lookahead, temperature-softmax, and Gumbel-Top-k traces with deterministic replay and per-step branch schedules before Q_H training.
- Multi-step reward and Q targets start with bounded cumulative target RRI. Gamma remains symbolic in formulas, but the thesis-core comparison reports cumulative target RRI under equal acquisition budget. Log-improvement, episode normalization, and scalar motion/rule/validity/diversity penalties are extensions; the current one-step RRI label remains unchanged for VIN compatibility.
- A target-conditioned fitted Double-Q / Q_H model over finite candidate sets is a mandatory M5 thesis deliverable, trained from ASE oracle rollout data. Q_H must beat one-step greedy/model scoring on cumulative target RRI under equal acquisition budget, while oracle lookahead is reported as the upper bound.
- Fitted Double-Q is the first Q algorithm. IQL is a second offline-RL ablation only after Q_H is stable; SB3 DQN/PPO/SAC are deferred until an online Gymnasium simulator exists.
- Zarr is the first-choice rollout/Q store. It should contain replay/training/evaluation payloads and lineage without duplicating raw ASE/ATEK; full meshes are external path/hash/version references, and high-detail target mesh crops are embedded once per target with crop metadata.
- The public glossary is a tiered math lookup table generated from `docs/typst/shared/glossary.typ`: core thesis math terms render first with shared symbol/equation refs, support terms remain normal glossary entries, and peripheral background terms stay linkable but visually demoted.
- CI/pre-commit becomes required before full-scale generation, not before proposal/M1 groundwork. GitHub issue mirroring remains a local TODO; `.agents/*.toml` stays the source of truth.
