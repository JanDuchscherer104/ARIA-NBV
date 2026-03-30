# Aria-NBV Autoimprove Workspace

This workspace adapts the core idea of [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) to a Python codebase that cannot be optimized by a single scalar training metric alone.

The ground-truth program file is:

- [`autoimprove.md`](/home/jandu/repos/NBV/autoimprove.md)

That Markdown file, with YAML front matter, defines:

- the editable and protected code scope
- the redundancy cost function
- adjacent module groups that should collapse to one owner
- helper-root policy
- mode-specific goals and verification commands

## Key Difference From `autoresearch`

`autoresearch` optimizes one mutable training file against a short-run validation metric.

Aria-NBV `autoimprove` optimizes:

- lower Python LOC
- fewer duplicate module pairs
- fewer repeated contract definitions
- fewer repeated helper definitions
- fewer helper-dense feature modules
- unchanged behavior proven by mode-specific test commands

The executor is Codex itself. Each turn picks one `mode`, reads [`autoimprove.md`](/home/jandu/repos/NBV/autoimprove.md), edits code within the declared scope, runs the configured checks, and emits a report.

## Commands

Render the active prompt:

```bash
cd /home/jandu/repos/NBV/aria_nbv
.venv/bin/python -m aria_nbv.utils.autoimprove --spec ../autoimprove.md prompt --mode simplify
```

Generate the current Markdown inventory:

```bash
cd /home/jandu/repos/NBV/aria_nbv
.venv/bin/python -m aria_nbv.utils.autoimprove --spec ../autoimprove.md report --mode simplify
```

Compute the current score:

```bash
cd /home/jandu/repos/NBV/aria_nbv
.venv/bin/python -m aria_nbv.utils.autoimprove --spec ../autoimprove.md score --mode simplify
```

Or use the convenience wrapper:

```bash
/home/jandu/repos/NBV/.agents/workspace/autoimprove/run_autoimprove.sh simplify
```

## Outputs

Generated reports live under:

- `.agents/workspace/autoimprove/reports/`

The report file is mode-specific and intentionally checked into the workspace, not the package runtime.
