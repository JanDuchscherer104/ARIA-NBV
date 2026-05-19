"""Small Typer runner utilities for stable console-script wrappers."""

from __future__ import annotations

from typing import Any

import click
import typer
from typer.main import get_command


def run_typer_app(
    app: typer.Typer,
    argv: list[str],
    *,
    prog_name: str,
    obj: dict[str, Any] | None = None,
) -> None:
    """Run a Typer app without raising ``SystemExit`` on successful commands."""

    command = get_command(app)
    try:
        command.main(args=argv, prog_name=prog_name, standalone_mode=False, obj=obj)
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from None
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None
    except click.Abort:
        raise SystemExit(1) from None


__all__ = ["run_typer_app"]
