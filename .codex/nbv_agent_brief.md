# NBV Agent Quickstart (re-init)

- **Env & kickoff**: Use `/home/jandu/miniforge3/envs/aria-nbv/bin/python` (py3.11). On arrival run `make context`; read `docs/index.qmd` and `docs/contents/todos.qmd` before coding.
- **Workflow ritual**: Condense the problem, explore relevant files, outline solution with acceptance/termination criteria, then implement incrementally; keep a running task list. Pause to #think about alignment regularly.
- **Core conventions**: Config-as-factory (`BaseConfig.setup_target()`), no direct class instantiation. Full type hints, Google-style docstrings with tensor shapes. Prefer `Path`, `Enum`, `match-case`, `Literal`. Use ARIA constants, `PoseTW`, `CameraTW`, EFM3D/ATEK utilities instead of re‑implementing. Log via `oracle_rri.utils.Console`.
- **Coding style**: Minimal comments; document tricky blocks. Keep ASCII. Session/state objects fully typed. Follow column enums and coordinate-frame notes (world/rig/cam, RDF, gravity -Z).
- **Testing & quality**: Work test-driven; for touched files run `ruff format <file>` → `ruff check <file>` → `pytest <file>` (real data where possible). Do not ship untested changes.
- **Documentation**: Update docs when behavior changes; prefer existing pattern examples in `docs/contents/impl/*.qmd`.
- **MCP/tools**: Use code-index/find, Context7 docs, and web search when needed; prefer external implementations over reinventing. Use `make context-dir-tree` or `make context-external` for overviews.
- **Scope guardrails**: Do not revert user changes; avoid destructive git commands; ask when unsure. Keep visualization tabs typed and cache-aware; Streamlit vars should be typed views.
