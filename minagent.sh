#!/usr/bin/env bash
# Launch MinAgent through uv, keeping your current directory as the workspace.
#
# Usage:
#   cd /path/to/your/project
#   /path/to/MinAgent/minagent.sh
#
# The workspace is the directory you run this from, not the MinAgent directory.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
	echo "minagent: uv is required but was not found on PATH." >&2
	echo "  Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
	exit 1
fi

# Tell MinAgent where its .env and .minagent/mcp.json live, not the workspace.
export MINAGENT_ROOT="$SCRIPT_DIR"

# Create .env from the example on first run so the model setting is discoverable.
if [ ! -f "$SCRIPT_DIR/.env" ] && [ -f "$SCRIPT_DIR/.env.example" ]; then
	echo "minagent: no .env found; copy $SCRIPT_DIR/.env.example to $SCRIPT_DIR/.env and set OPENAI_MODEL." >&2
fi

# --project keeps your cwd as the workspace; only the environment comes from MinAgent.
exec uv run --project "$SCRIPT_DIR" minagent "$@"
