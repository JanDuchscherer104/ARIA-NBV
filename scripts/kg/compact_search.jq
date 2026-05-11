### Compact kg-search summary (<=12 lines).
### Default output for `make kg-search`; opt out via KG_VERBOSE=1 or KG_FORMAT=json.

def cap($n): if length > $n then .[:$n] else . end;
def truncate($n): if length > $n then .[:$n] + "..." else . end;

"top_hits:",
( . // []
  | cap(5)
  | if length == 0 then ["  (no matches)"]
    else (
      to_entries
      | map(
          "  " + (.key + 1 | tostring) + ". "
          + (.value.title // .value.node_id // "(no-title)" | truncate(60))
          + " [" + (.value.authority // "default") + "/" + (.value.source_type // "?") + "]"
          + " score=" + ((.value.score_final // .value.score // 0) | tostring | .[0:6])
          + (if .value.repo_path then "\n     " + .value.repo_path else "" end)
        )
    )
    end
  | .[]
),
"",
"# For full search payload: re-run with KG_VERBOSE=1 or KG_FORMAT=json."
| @text
