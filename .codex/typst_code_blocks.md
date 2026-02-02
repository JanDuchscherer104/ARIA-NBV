## Typst slides: Python code blocks as figures (Codly)

### Decision

Use the Typst Universe package `@preview/codly` for slide code blocks. It upgrades fenced
`raw` blocks with a consistent framed layout and (optionally) line-referencing.

### Repo changes

- `docs/typst/slides/template.typ`
  - Import `@preview/codly:1.3.0`.
  - Enable Codly globally via `#show: codly-init.with()`.
  - Configure slide defaults (no language tag box, no line numbers, rounded corners).
  - Add a helper `#code-figure(...)` that wraps a fenced code block in `#figure(kind: raw, ...)`
    and shrinks the code font for slides.
- `docs/typst/slides/slides_4.typ`
  - Replace the manual `#figure([...```python ...```...])` with `#code-figure(...)`.

### Recommended usage pattern

```typ
#code-figure(caption: [What this snippet illustrates.])[
  ```python
  def f(x: int) -> int:
      return x + 1
  ```
] <lst:my-snippet>
```

For minimal styling, use the raw-based `code-block` / `code-figure` helpers.

### Compile note

`slides_4.typ` uses absolute paths like `/typst/...` that assume the Typst root is `docs/`:

```bash
typst compile --root docs docs/typst/slides/slides_4.typ /tmp/slides_4.pdf
```
