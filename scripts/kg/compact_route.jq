### Compact kg-route summary (<=15 lines).
### Default output for `make kg-route`; opt out via KG_VERBOSE=1 or KG_FORMAT=json.
###
### Invariant: this filter MUST NOT surface ContextPack fields
###   evidence_spans, backend_status, action_plan, assumptions,
###   missing_leaves, missing_context_leaves, profile, budget_tokens, truncated.
### Those are either legacy aliases (missing_*), debug metadata (profile,
### budget_tokens, truncated), or bulk data the agent shouldn't read
### unfiltered. Use top_sources, suggested_next_action, required_reads,
### risk_flags, and the active backlog instead.

def cap($n): if length > $n then .[:$n] else . end;
def or_none: if . == null or . == "" then "(none)" else . end;
def truncate($n): if length > $n then .[:$n] + "..." else . end;

# Header.
"task: " + (.task_summary // .task // "(none)" | truncate(96)),
"verb: " + (.verb // "(none)"),
"",

# Top sources (max 3): path [tier] score=X.XX
"top_sources:",
( .top_sources // []
  | cap(3)
  | map(
      "  - " + (.path // "(no-path)")
      + " [" + (.scores.authority // "default") + "]"
      + " score=" + ((.scores.score_final // 0) | tostring | .[0:5])
    )
  | if length == 0 then ["  (no high-signal sources; consider aria-nbv-context)"] else . end
  | .[]
),
"",

# Active backlog (max 4 across issues + todos).
( ((.active_issues // []) + (.active_todos // []))
  | cap(4)
  | if length == 0 then "active_backlog: (none)"
    else "active_backlog: " + (map(.id) | join(", "))
    end
),

# Risk flags inline.
"risk_flags: " + ((.risk_flags // []) | if length == 0 then "(none)" else join("; ") end),

# Suggested next action.
( .suggested_next_action // {}
  | "next: " + ((.summary // "(none)") | truncate(100))
    + (if .skill then "  (skill: " + .skill + ")" else "" end)
),

# First required read (path only; objects collapsed).
"read_first: " + ((.required_reads // [])
  | if length == 0 then "(none)"
    else (.[0] | if type == "object" then (.path // .title // "(no-path)") else . end)
    end),

# Footer hint.
"",
"# For full context pack: re-run with KG_VERBOSE=1 or KG_FORMAT=json."
| @text
