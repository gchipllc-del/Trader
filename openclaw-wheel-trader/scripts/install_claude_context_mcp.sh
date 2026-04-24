#!/usr/bin/env bash
#
# Install zilliztech/claude-context as a Claude Code MCP server.
#
# Claude-context provides semantic code search over this repo (and polybot),
# with ~40% token reduction on large-scope questions. It runs as an MCP
# server (Node.js) backed by Milvus vector DB + an embedding provider.
#
# Requirements (you need ONE from each pair):
#   - Embeddings:  OPENAI_API_KEY   OR   VOYAGE_API_KEY   OR   Ollama running locally
#   - Vector DB:   Zilliz Cloud (free tier OK)  OR  self-hosted Milvus
#
# Cheapest path if you don't want new sign-ups:
#   1. Ollama for embeddings (fully local, no API key):
#        brew install ollama && ollama pull nomic-embed-text
#   2. Zilliz Cloud free tier for the vector DB:
#        https://cloud.zilliz.com/  -> create free cluster
#
# Usage:
#   ./scripts/install_claude_context_mcp.sh
#
# The script will prompt for missing values. Aborts cleanly if prerequisites
# aren't met. Safe to re-run — uses `claude mcp remove` + `claude mcp add`.
#
set -euo pipefail

NAME="claude-context"

# --- prereqs ---------------------------------------------------------------
if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: Node.js not found. Install Node 20.x-23.x first." >&2
  exit 1
fi
NODE_MAJOR=$(node -v | sed -E 's/^v([0-9]+).*/\1/')
if [ "$NODE_MAJOR" -lt 20 ] || [ "$NODE_MAJOR" -gt 23 ]; then
  echo "ERROR: Node v${NODE_MAJOR} not supported by @zilliz/claude-context-mcp (needs 20.x-23.x)." >&2
  exit 1
fi
if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not found. Install Claude Code first." >&2
  exit 1
fi

# --- gather env vars -------------------------------------------------------
EMBED_PROVIDER="${EMBED_PROVIDER:-}"
if [ -z "${EMBED_PROVIDER}" ]; then
  if [ -n "${OPENAI_API_KEY:-}" ]; then
    EMBED_PROVIDER="openai"
  elif [ -n "${VOYAGE_API_KEY:-}" ]; then
    EMBED_PROVIDER="voyageai"
  else
    echo "Pick embedding provider:"
    echo "  1) OpenAI (needs OPENAI_API_KEY)"
    echo "  2) VoyageAI (needs VOYAGE_API_KEY)"
    echo "  3) Ollama (fully local, needs 'ollama pull nomic-embed-text')"
    read -rp "Choice [1/2/3]: " choice
    case "$choice" in
      1) EMBED_PROVIDER="openai" ;;
      2) EMBED_PROVIDER="voyageai" ;;
      3) EMBED_PROVIDER="ollama" ;;
      *) echo "Invalid choice." >&2; exit 1 ;;
    esac
  fi
fi

MILVUS_ADDRESS="${MILVUS_ADDRESS:-}"
MILVUS_TOKEN="${MILVUS_TOKEN:-}"
if [ -z "$MILVUS_ADDRESS" ] || [ -z "$MILVUS_TOKEN" ]; then
  echo ""
  echo "Zilliz Cloud endpoint required. Free tier at https://cloud.zilliz.com/"
  [ -z "$MILVUS_ADDRESS" ] && read -rp "MILVUS_ADDRESS (public endpoint URL): " MILVUS_ADDRESS
  [ -z "$MILVUS_TOKEN" ]   && read -rsp "MILVUS_TOKEN (API key, hidden): " MILVUS_TOKEN && echo
fi

# --- build env-arg list ----------------------------------------------------
ENV_ARGS=(
  -e "MILVUS_ADDRESS=${MILVUS_ADDRESS}"
  -e "MILVUS_TOKEN=${MILVUS_TOKEN}"
  -e "EMBEDDING_PROVIDER=${EMBED_PROVIDER}"
)
case "$EMBED_PROVIDER" in
  openai)
    if [ -z "${OPENAI_API_KEY:-}" ]; then
      read -rsp "OPENAI_API_KEY (hidden): " OPENAI_API_KEY && echo
    fi
    ENV_ARGS+=(-e "OPENAI_API_KEY=${OPENAI_API_KEY}")
    ;;
  voyageai)
    if [ -z "${VOYAGE_API_KEY:-}" ]; then
      read -rsp "VOYAGE_API_KEY (hidden): " VOYAGE_API_KEY && echo
    fi
    ENV_ARGS+=(-e "VOYAGE_API_KEY=${VOYAGE_API_KEY}")
    ;;
  ollama)
    ENV_ARGS+=(-e "OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}")
    ENV_ARGS+=(-e "OLLAMA_MODEL=${OLLAMA_MODEL:-nomic-embed-text}")
    ;;
esac

# --- install ---------------------------------------------------------------
echo ""
echo "Removing any prior '$NAME' MCP registration (idempotent)..."
claude mcp remove "$NAME" 2>/dev/null || true

echo "Registering MCP server..."
claude mcp add "$NAME" \
  "${ENV_ARGS[@]}" \
  -- npx @zilliz/claude-context-mcp@latest

echo ""
echo "✓ Done. Verify with:   claude mcp list"
echo ""
echo "Next: inside a 'claude' session in this repo, prompt:"
echo "    Index this codebase using claude-context"
echo "Then: ask things like 'find all code that handles earnings filters'."
