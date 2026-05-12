### Compact kg-search summary.
### Default output for `make kg-search`; opt out via KG_VERBOSE=1 or KG_FORMAT=json.
###
### For each top hit, surface:
###   - hit index + a 1-token "kind tag" (paper / mem / code / doc)
###   - paper_id (when node_id is paper:*; the title alone is ambiguous because
###     PaperSection titles repeat across papers)
###   - title or symbol path, lightly truncated
###   - the snippet (first ~140 chars); this is the actually-useful evidence
###   - the on-disk repo_path when present (paper:* nodes carry null today
###     pending todo-062 provenance fix)
###
### When Phase B (todo-067) ships, kg-search may emit a wrapper object with
### .results + .search_mode + .query_fixups + .mode_reason fields. This filter
### handles both shapes: bare array (today) and wrapped object (future).

def cap($n): if length > $n then .[:$n] else . end;
def truncate($n; $cap): if length > $cap then .[:$cap] + "..." else . end;

# Extract paper_id from a node_id like "paper:pb-nbv-jia2025:section:3" or
# "paper:pb-nbv-jia2025". Returns null for non-paper nodes.
def paper_id_from_node_id:
  if startswith("paper:")
  then (sub("^paper:"; "") | sub(":section:.*$"; ""))
  else null
  end;

# One-token kind tag for the hit (mem / paper / code / doc / ?).
def kind_tag:
  if (.kind // "") == "PaperSection" or (.kind // "") == "Paper" then "paper"
  elif (.kind // "") == "ProjectMemory" then "mem"
  elif (.kind // "") == "CodeSymbol" or (.kind // "") == "CodeFile" then "code"
  elif (.kind // "") == "Document" or (.kind // "") == "DocSection" then "doc"
  elif (.source_type // "") == "code" then "code"
  elif (.source_type // "") == "canonical_memory" or (.source_type // "") == "agent_backlog" then "mem"
  else "?"
  end;

# Normalize input: bare array OR {results, search_mode, ...} wrapper.
( if type == "array" then {results: ., search_mode: "lexical_only", mode_reason: "kg-search has not yet adopted hybrid output (todo-067 pending)", query_fixups: []}
  else . + {results: (.results // []), search_mode: (.search_mode // "lexical_only"), mode_reason: (.mode_reason // ""), query_fixups: (.query_fixups // [])}
  end
) as $env
| ($env.results // []) as $hits

| "top_hits:",
( $hits
  | cap(5)
  | if length == 0 then ["  (no matches)"]
    else (
      to_entries
      | map(
          (.value.node_id // "") as $nid
          | ($nid | paper_id_from_node_id) as $pid
          | (.value | kind_tag) as $tag
          | (.value.snippet // "" | gsub("\\s+"; " ") | truncate(160; 160)) as $snip
          | (.value.title // .value.node_id // "(no-title)" | gsub("\\s+"; " ")) as $title
          # For ProjectMemory the title is just the content prefix - drop it
          # in favor of the snippet line so we don't print the same text twice.
          | (if $tag == "mem"
               then (.value.node_id // "(memory)" | sub(":[0-9]+$"; "") | sub("^memory:[^:]+:"; ""))
               else $title
             end) as $display_title
          | "  " + (.key + 1 | tostring) + ". [" + $tag + "] "
            + (if $pid then $pid + " :: " else "" end)
            + ($display_title | truncate(80; 80))
            + (if .value.repo_path
                 then "\n     -> " + .value.repo_path
                 else (if $pid then "\n     -> (no repo_path; see docs/contents/literature/ or docs/references.bib)" else "" end)
               end)
            + (if ($snip | length) > 0 then "\n     \"" + $snip + "\"" else "" end)
        )
    )
    end
  | .[]
),
"",

# Query fixups (Phase A fuzzy fallback surfaces these here when shipped).
( if ($env.query_fixups | length) > 0
    then "# query_fixups: " + ($env.query_fixups | join(", "))
    else empty
  end ),

# Search-mode hint: tells the agent which retrieval path produced the hits.
"# search_mode: " + $env.search_mode
  + (if ($env.mode_reason // "") != "" then "  (reason: " + $env.mode_reason + ")" else "" end),
"# For full search payload: re-run with KG_VERBOSE=1 or KG_FORMAT=json."
| @text
