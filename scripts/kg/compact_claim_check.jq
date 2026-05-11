### Compact kg-claim-check summary (<=14 lines).
### Default output for `make kg-claim-check`; opt out via KG_VERBOSE=1 or KG_FORMAT=json.

def cap($n): if length > $n then .[:$n] else . end;
def truncate($n): if length > $n then .[:$n] + "..." else . end;

"claim: " + ((.task // .task_summary // "(none)") | truncate(140)),
"verdict: " + (.verdict // "unverifiable")
  + " (confidence=" + ((.confidence // 0) | tostring | .[0:4]) + ")",
"",

"supporting:",
( .supporting_evidence // []
  | cap(2)
  | if length == 0 then ["  (none)"]
    else map(
      "  - " + (.source_path // .path // "(no-path)")
        + (if .line_start then ":" + (.line_start | tostring) else "" end)
        + " [" + (.scores.authority // .authority // "default") + "]"
    )
    end
  | .[]
),

"contradicting:",
( .contradicting_evidence // []
  | cap(2)
  | if length == 0 then ["  (none)"]
    else map(
      "  - " + (.source_path // .path // "(no-path)")
        + (if .line_start then ":" + (.line_start | tostring) else "" end)
        + " [" + (.scores.authority // .authority // "default") + "]"
    )
    end
  | .[]
),
"",
( if (.verdict // "") == "unverifiable"
  then "# Verdict unverifiable: literature `paper:*` nodes lack source paths today.\n# Run `make kg-search KG_QUERY='<claim keywords>'` to inspect raw evidence directly."
  else "# For full evidence + spans: re-run with KG_VERBOSE=1 or KG_FORMAT=json."
  end
)
| @text
