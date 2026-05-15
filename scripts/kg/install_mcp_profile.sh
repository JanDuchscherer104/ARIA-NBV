#!/usr/bin/env bash
# Install the litkg-cypher MCP profile.
#
# Registers the neo4j-contrib mcp-neo4j-cypher server against the MCP_DOCKER
# gateway, pointed at the local litkg Neo4j (bolt://127.0.0.1:7687, user
# neo4j, db neo4j). Read-only by default. Snapshots the result as the
# `litkg-cypher` profile so any Claude / Codex / Gemini session can load it
# with `mcp-activate-profile litkg-cypher`.
#
# Prerequisites
#   - MCP_DOCKER gateway already wired in your CLI config (Gemini already
#     references it in .gemini/settings.json; Claude / Codex configurations
#     should also point at the gateway).
#   - Neo4j running: `make kg-up`. APOC required for get_neo4j_schema; the
#     repo's docker-compose at .agents/external/litkg-rs/infra/neo4j/
#     loads it via NEO4J_PLUGINS='["apoc"]'.
#
# Env knobs
#   NEO4J_URL              Bolt URL (default: bolt://127.0.0.1:7687)
#   NEO4J_USERNAME         Neo4j user (default: neo4j)
#   NEO4J_DATABASE         Database name (default: neo4j)
#   NEO4J_PASSWORD         Password for the secret store (default: litkglocal)
#   KG_MCP_PROFILE         Profile name to snapshot (default: litkg-cypher)
#   KG_MCP_READ_ONLY       1 / true to lock writes (default: 1)
#
# Idempotent: re-runs update config + re-snapshot the profile.

set -eu

REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO"

NEO4J_URL="${NEO4J_URL:-bolt://127.0.0.1:7687}"
NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-litkglocal}"
KG_MCP_PROFILE="${KG_MCP_PROFILE:-litkg-cypher}"
KG_MCP_READ_ONLY="${KG_MCP_READ_ONLY:-1}"

case "$KG_MCP_READ_ONLY" in
  1|true|TRUE|yes) read_only=true ;;
  *)               read_only=false ;;
esac

cat <<EOF
litkg-cypher MCP install
  Neo4j URL:      $NEO4J_URL
  Database:       $NEO4J_DATABASE
  Username:       $NEO4J_USERNAME
  Read-only:      $read_only
  Profile name:   $KG_MCP_PROFILE
  Server (catalog): neo4j-cypher (neo4j-contrib mcp-neo4j-cypher)

This script DOES NOT call mcp-add directly because the MCP gateway lives
inside the agent's tool surface, not the shell. Run the following from
an agent session that has the MCP_DOCKER gateway tools loaded
(Claude / Codex / Gemini):

  # 1. Register the server
  mcp-add neo4j-cypher

  # 2. Configure the connection
  mcp-config-set neo4j-cypher \\
      url=$NEO4J_URL \\
      username=$NEO4J_USERNAME \\
      database=$NEO4J_DATABASE \\
      read_only=$read_only \\
      schema_sample_size=1000 \\
      read_timeout=30s

  # 3. Set the password (secret store)
  #    The secret name is neo4j-cypher.password
  #    Use your usual secret-store tool; for the docker-compose default:
  #      value = $NEO4J_PASSWORD

  # 4. Snapshot the active gateway as a reusable profile
  mcp-create-profile $KG_MCP_PROFILE

Once snapshotted, future sessions activate with:
  mcp-activate-profile $KG_MCP_PROFILE

Available tools after activation:
  - read_neo4j_cypher(query, params)       # read-only Cypher
  - get_neo4j_schema(sample_param=1000)    # requires APOC
  - write_neo4j_cypher(query, params)      # only if read_only=false

See .agents/references/litkg_quick_reference.md (Typed-query escape
hatch) for example queries.
EOF
