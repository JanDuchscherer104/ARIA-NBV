### Compact kg-route summary.
### Default output for `make kg-route`; opt out via KG_VERBOSE=1 or KG_FORMAT=json.
###
### For each top source, surface:
###   - path with line range (path:start-end), the actually-actionable locator
###   - paper_id when the source is a paper:* node (parity with kg-search)
###   - the authority tier
###   - one short "why" line preferring backlog-injection context over the
###     generic "boosted X source" entries, which carry the real signal
###     ("this hit is here because matched backlog record todo-027")
###
### For active backlog: show id, priority, and a short title - the bare id
### list ("issue-019, issue-020, ...") forced agents to context-switch to
### the TOML files to know what they were looking at.
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

# Extract paper_id from a paper:*:section:N or paper:* node id.
def paper_id_from_path:
  if (. // "") | startswith("paper:")
  then sub("^paper:"; "") | sub(":section:.*$"; "")
  else null
  end;

# Pick the most informative why_relevant entry. Prefer entries that name a
# specific backlog record ("injected from matched backlog record todo-XYZ");
# fall back to the first non-generic entry; only emit anything when there
# is a clear signal-bearing line.
def best_why_relevant:
  ( . // [] ) as $why
  | ( $why | map(select(. | test("matched backlog|injected from|verification target|missing"; "i"))) ) as $signal
  | ( $signal[0] // ($why | map(select(. | test("boosted "; "i") | not))[0])
      // null )
  ;

# path:line_start-line_end when source_span carries a real range, otherwise just path.
def path_with_range:
  (.path // "(no-path)") as $p
  | (.source_span.line_start // 0) as $s
  | (.source_span.line_end // 0) as $e
  | if ($s > 0 and $e > 0 and $s != $e) then $p + ":" + ($s|tostring) + "-" + ($e|tostring)
    elif ($s > 0) then $p + ":" + ($s|tostring)
    else $p
    end;

# Header.
"task: " + (.task_summary // .task // "(none)" | truncate(96)),
"verb: " + (.verb // "(none)"),
"",

# Top sources (max 3): include line range, authority, and a 1-line "why".
"top_sources:",
( .top_sources // []
  | cap(3)
  | map(
      (.path | paper_id_from_path) as $pid
      | (.why_relevant | best_why_relevant) as $why
      | "  - " + (. | path_with_range)
        + (if $pid then "  (paper: " + $pid + ")" else "" end)
        + "  [" + (.scores.authority // "default") + "]"
        + (if $why then "\n      why: " + ($why | truncate(110)) else "" end)
    )
  | if length == 0 then ["  (no high-signal sources; consider aria-nbv-context)"] else . end
  | .[]
),
"",

# Active backlog (max 4): show id, priority, and a short title.
"active_backlog:",
( ((.active_issues // []) + (.active_todos // []))
  | cap(4)
  | map(
      "  - " + (.id // "?")
      + " [" + (.priority // "?") + "]"
      + " " + (.title // "(no title)" | truncate(80))
    )
  | if length == 0 then ["  (none)"] else . end
  | .[]
),
"",

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
