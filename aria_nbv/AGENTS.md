# Package Guidance

Apply this file when working under `aria_nbv/`.

## Commands
- Python: `aria_nbv/.venv/bin/python`
- Format: `ruff format <file>`
- Lint: `ruff check <file>`
- Tests: `uv run pytest <path>` or `aria_nbv/.venv/bin/python -m pytest <path>`
- Context: `make context`

## Invariants
- Use `PoseTW` for poses and `CameraTW` for cameras.
- Use `pathlib.Path` for filesystem paths.
- Instantiate runtime objects through config `.setup_target()` methods.
- Prefer `Field(default_factory=...)` for computed defaults.
- Use `Console` from `aria_nbv.utils` for structured logging.
- Prefer existing utilities from `efm3d`, `atek`, and `projectaria_tools` over reimplementation.
- Use ARIA constants from `efm3d.aria.aria_constants` for dataset keys.
- Follow EFM3D / ATEK coordinate conventions and document tensor shapes plus coordinate frames where they are not obvious.

## Style
- Keep public signatures fully typed and use modern builtins such as `list[str]` and `dict[str, Any]`.
- Use `TYPE_CHECKING` guards for type-only imports.
- Use `Literal` for constrained string values.
- Prefer `Enum` for categorical values and `match-case` for multi-branch logic when it improves clarity.
- Prefer vectorized implementations over helper-heavy or loop-heavy versions when readability remains acceptable.
- Public methods should have Google-style docstrings.
- For config models, document fields with attribute docstrings instead of `Field(..., description=...)` for ordinary primitive fields.
- Use `field_validator` and `model_validator` when validation belongs in the config layer.
- See [python_conventions.md](/home/jandu/repos/NBV/.agents/references/python_conventions.md) for full examples and anti-patterns.

## Recent Gotchas
- See [GOTCHAS.md](/home/jandu/repos/NBV/.agents/memory/state/GOTCHAS.md) for the maintained full list.
- Validation is disabled by default unless `trainer_config.enable_validation=true`.
- `Field(default=<callable>)` stores the callable; use `Field(default_factory=...)` for computed defaults.
- `uv run pytest` is preferred over the system interpreter; use the repo venv python when uv resolution is suspect.
- Offline cache splits are file-backed and may create or update `train_index.jsonl` and `val_index.jsonl`.
- `VinOracleBatch.collate` expects cache-ready `VinSnippetView` snippets rather than full `EfmSnippetView` instances.
- EVL OBB outputs are not batch-collatable yet; entity-aware runs may need `batch_size=None` or OBB outputs disabled.

## Verification
- For package changes, run format -> lint -> targeted pytest on the changed surface.
- Prefer real-data or integration-style tests when feasible.
- Keep public signatures typed and public methods documented.
