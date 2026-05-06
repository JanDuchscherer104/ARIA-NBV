# ARIA-NBV Verification Matrix

Use the narrowest checks that cover the changed surface. Add broader checks only
when behavior crosses boundaries.

## Agent Scaffold, Skills, Memory

- Skill validation:
  `python3 /home/jd/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/<skill>`
- Agent memory:
  `make check-agent-memory`
- Agents DB:
  `make agents-db AGENTS_ARGS='validate'`
  `make agents-db`

## litkg Guidance Or KG Config

- `make kg-capabilities KG_FORMAT=json`
- `make kg-route KG_TASK="<task>" KG_FORMAT=json`
- `make kg-claim-check KG_CLAIM="<claim>"` for advisor-facing claims

## Public Docs

- Frontmatter:
  `make qmd-frontmatter-check`
- Focused Quarto render:
  `cd docs && quarto render <page.qmd>`
- Focused Typst render:
  `cd docs && typst compile typst/seminar_paper/main.typ --root .`
  or `cd docs && typst compile typst/thesis/proposal.typ --root .`

## Python Package

- Format:
  `ruff format <file>`
- Lint:
  `ruff check <file>`
- Targeted tests:
  `cd aria_nbv && uv run pytest <path>`

## Research Contract Examples

- Data handling/offline store:
  `cd aria_nbv && uv run pytest tests/data_handling/test_vin_offline_store.py`
- RRI:
  `cd aria_nbv && uv run pytest tests/rri_metrics`
- Rollouts:
  `cd aria_nbv && uv run pytest tests/pose_generation/test_counterfactuals.py`
- Rerun inspector:
  run the focused inspector tests or smoke command named by
  `rerun-nbv-inspector`.
