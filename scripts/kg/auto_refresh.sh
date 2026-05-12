#!/usr/bin/env bash
# Background-safe KG refresh.
#
# Designed for the Claude/Codex Stop hook. Never blocks the calling session,
# never errors out, never produces stdout/stderr the user has to read. All
# state goes to .agents/kg/.refresh.log.
#
# Skips silently when:
#   - the Mac/remote ollama endpoint at 127.0.0.1:11434 is not reachable
#     (your `ssh -N -R 11434:127.0.0.1:11434 ubuntu` tunnel is down);
#   - no tracked source under docs/, aria_nbv/, .agents/memory/state/, or
#     .agents/*.toml has been modified since .agents/kg/.last-refresh;
#   - another refresh is already in flight (.agents/kg/.refresh.lock present).
#
# Otherwise spawns `make kg-refresh-light` in the background via nohup; the
# detached process logs success/failure and releases the lock.
#
# Force a refresh (skip ollama probe and staleness check):
#   KG_FORCE=1 scripts/kg/auto_refresh.sh
#
# Manually clear a stale lock:
#   rm .agents/kg/.refresh.lock

set -u

REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
cd "$REPO" || exit 0

KG_DIR="$REPO/.agents/kg"
LOG="$KG_DIR/.refresh.log"
STAMP="$KG_DIR/.last-refresh"
LOCK="$KG_DIR/.refresh.lock"
TS="$(date -Iseconds)"
FORCE="${KG_FORCE:-0}"

mkdir -p "$KG_DIR" 2>/dev/null || exit 0

log() {
  echo "$TS $*" >> "$LOG" 2>/dev/null || true
}

# 1) Ollama tunnel reachable? Skip on first failure (don't retry).
if [ "$FORCE" != "1" ]; then
  if ! curl -sSf --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    log "skip: ollama tunnel unreachable at 127.0.0.1:11434"
    exit 0
  fi
fi

# 1a) Optional Neo4j warm-start. Opt-in via KG_NEO4J_AUTO_UP=1.
# When set, ensure the local Neo4j Docker is running so that future kg-search
# queries can use the vector index (kg_embedding_index_2560). When the runtime
# is up, hybrid (lexical + cosine) search becomes available; when it isn't,
# search falls back to lexical-only with an explicit mode hint.
if [ "${KG_NEO4J_AUTO_UP:-0}" = "1" ]; then
  if ! curl -sSf --max-time 1 http://127.0.0.1:7474/ >/dev/null 2>&1; then
    log "neo4j down and KG_NEO4J_AUTO_UP=1 set; attempting make kg-up"
    if command -v docker >/dev/null 2>&1; then
      ( nohup make kg-up >> "$LOG" 2>&1 & disown 2>/dev/null || true )
      log "kg-up dispatched in background; vector search available after warm-up"
    else
      log "skip: docker not on PATH; cannot warm-start Neo4j"
    fi
  fi
fi

# 2) Anything to do? Compare tracked sources against the stamp.
if [ "$FORCE" != "1" ] && [ -f "$STAMP" ]; then
  # Use mtime comparison; bound the search depth to keep this fast.
  newest=$(find \
    docs/contents docs/typst aria_nbv/aria_nbv \
    .agents/memory/state \
    .agents/issues.toml .agents/todos.toml .agents/refactors.toml \
    -type f -newer "$STAMP" \
    \( -name '*.md' -o -name '*.qmd' -o -name '*.typ' \
       -o -name '*.py' -o -name '*.toml' \) \
    2>/dev/null | head -1)
  if [ -z "$newest" ]; then
    log "skip: no changed sources since last refresh"
    exit 0
  fi
fi

# 3) Lock — abort if a previous refresh is still running.
if ! ( set -o noclobber; echo "$$" > "$LOCK" ) 2>/dev/null; then
  existing=$(cat "$LOCK" 2>/dev/null || echo "?")
  log "skip: refresh already running (pid $existing); leave it alone"
  exit 0
fi

# 4) Detach and run.
log "start: kg-refresh-light (background pid $$)"
(
  trap 'rm -f "$LOCK"' EXIT
  if make kg-refresh-light >> "$LOG" 2>&1; then
    touch "$STAMP"
    log "done: kg-refresh-light ok"
  else
    log "fail: kg-refresh-light non-zero exit (see lines above)"
  fi
) </dev/null >/dev/null 2>&1 &
disown 2>/dev/null || true
exit 0
