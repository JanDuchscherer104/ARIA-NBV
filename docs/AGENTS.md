# Docs Guidance

Apply this file when working under `docs/`.

## Priorities
- Treat source order as role-split; see `.agents/references/source_order.md`.
- Treat `docs/typst/seminar_paper/main.typ` as the implemented-substrate
  ground truth.
- Treat `docs/contents/thesis/roadmap.qmd`,
  `docs/contents/thesis/questions.qmd`, and `.agents/memory/state/` as the
  current thesis-direction ground truth.
- Treat `docs/typst/thesis/proposal.typ` as the advisor proposal narrative
  when proposal work is in scope.
- Keep Quarto docs aligned to the correct source role instead of introducing
  competing top-level narratives.
- Keep `docs/references.bib` as the single bibliography source of truth.
- Preserve established Quarto and Typst structure unless the task explicitly changes it.
- Prefer links to canonical state docs in `.agents/memory/state/` over re-explaining the same guidance in multiple places.
- Keep internal agent guidance, generated context, and OMX runtime notes out of
  public Quarto navigation. If a generated agent mirror is needed, regenerate
  it under `.agents/generated/`; do not write agent mirrors under
  `docs/contents/**`.

## Default Workflow
- Start from `.agents/references/source_order.md` and then the source that owns
  the touched role.
- Use `docs/typst/seminar_paper/main.typ` for implemented substrate claims.
- Use `scripts/nbv_qmd_outline.sh --compact` to localize the exact Quarto page before opening it.
- Use `scripts/nbv_typst_includes.py --paper --mode outline` to localize the exact Typst section before opening it.
- Open `docs/index.qmd`, `docs/contents/thesis/roadmap.qmd`, and
  `docs/contents/thesis/questions.qmd` only when the task is about project
  narrative, priorities, or roadmap. Historical scratch pages live under
  `.agents/archive/docs/`.
- If you need current project truth beyond the paper, open the relevant doc in `.agents/memory/state/` instead of scanning broad doc trees.
- Run `make kg-claim-check KG_CLAIM="..."` for advisor-facing proposal,
  roadmap, research-question, or literature-synthesis claims.

## Commands
- Context refresh: `make context`
- API reference refresh: `./scripts/quarto_generate_api_docs.sh`
- Internal agent scaffold refresh: `./scripts/quarto_generate_agent_docs.py`
- QMD frontmatter check: `make qmd-frontmatter-check`
- Quarto render: `cd docs && quarto render .`
- Quarto preview: `cd docs && quarto preview`
- Quarto check: `quarto check`
- Typst paper: `cd docs && typst compile typst/seminar_paper/main.typ --root .`
- Typst slides: `cd docs && typst compile typst/seminar_slides/<file>.typ --root .`
- Typst fallback on sandboxed snap installs: `/snap/typst/current/bin/typst compile <file>.typ --root docs`
- QMD tree: `make context-qmd-tree`
- Outline-first routing: `scripts/nbv_qmd_outline.sh`, `scripts/nbv_typst_includes.py`

## Diagram Rules
- Validate Mermaid before committing diagram edits.
- For non-trivial Mermaid edits, validate standalone first with `npx -y @mermaid-js/mermaid-cli -i /tmp/diagram.mmd -o /tmp/diagram.svg`.
- Use `{mermaid}` fences in Quarto.
- Use `<br/>` for Mermaid line breaks and Mermaid-safe node ids.

## Writing Rules
- Keep cross-references and bibliography entries synchronized.
- Add new references to `docs/references.bib` when introducing important concepts or papers.
- Replace temporary citation placeholders such as `cite…` before finishing.
- Use links to relevant internal docs or authoritative external references when introducing non-obvious concepts.
- Keep Quarto source files (`*.qmd`) separate from rendered site output. Published HTML belongs under `docs/_site/`, not next to the sources.
- Treat `docs/_freeze/` as tracked execution state for code-backed pages when needed; treat `docs/_site/`, `site_libs/`, `index_files/`, and `*_files/` as generated publish artifacts.
- Do not store generated context or rendered artifacts in tracked docs paths unless the task explicitly requires it.
- Treat generated agent scaffold mirrors as internal operator artifacts under
  `.agents/generated/`; do not expose them in public Quarto content.
- Retained QMD pages should carry minimal ownership metadata: `title`, `phase`,
  `audience`, `status`, and `owner`. Use
  `phase: thesis | seminar | archive | generated`,
  `audience: public | advisor | developer | agent`,
  `status: current | planned | scratch | deprecated`, and
  `owner: paper | docs | code | agent | generated | jan`.
- Current thesis pages live under `docs/contents/thesis/`; past seminar
  material belongs under `docs/contents/seminar/`; raw scratch or stale history
  belongs under `.agents/archive/docs/`. Only curated public archive summaries
  may live under `docs/contents/archive/`.
- Active tasks belong in `.agents/*.toml`, not public TODO pages.
- For larger doc changes, run `make qmd-frontmatter-check`, `quarto render`,
  and `quarto check` before finishing.
