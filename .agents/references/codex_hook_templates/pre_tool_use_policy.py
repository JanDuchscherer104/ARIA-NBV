#!/usr/bin/env python3
"""Inactive Codex PreToolUse template for NBV shell guardrails."""

from __future__ import annotations

import json
import re
import sys


BLOCK_PATTERNS = (
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+restore\b",
    r"\bgit\s+clean\s+-[^\s]*f",
    r"\brm\s+-rf\s+(/|\.)",
)


def main() -> int:
    payload = json.load(sys.stdin)
    command = str(payload.get("tool_input", {}).get("command", ""))
    for pattern in BLOCK_PATTERNS:
        if re.search(pattern, command):
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": "NBV hook template blocked a destructive shell command.",
                        }
                    }
                )
            )
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
