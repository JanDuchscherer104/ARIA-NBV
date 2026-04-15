#!/usr/bin/env python3
"""Inactive Codex PostToolUse template for scaffold-check reminders."""

from __future__ import annotations

import json
import re
import sys


SCAFFOLD_PATTERN = re.compile(
    r"(AGENTS\.md|\.agents/|scripts/(validate_agent|quarto_generate_agent)|Makefile|\.pre-commit-config\.yaml)"
)


def main() -> int:
    payload = json.load(sys.stdin)
    command = str(payload.get("tool_input", {}).get("command", ""))
    if not SCAFFOLD_PATTERN.search(command):
        return 0
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": "Scaffold files may have changed. Run make check-agent-scaffold before finishing.",
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
