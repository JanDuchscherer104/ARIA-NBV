#!/usr/bin/env python3
"""Inactive Codex Stop hook template for scaffold verification reminders."""

from __future__ import annotations

import json
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    message = str(payload.get("last_assistant_message") or "")
    if "AGENTS.md" in message and "check-agent-scaffold" not in message:
        print(
            json.dumps(
                {
                    "decision": "block",
                    "reason": "Scaffold guidance was mentioned; verify whether make check-agent-scaffold should be run or reported.",
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
