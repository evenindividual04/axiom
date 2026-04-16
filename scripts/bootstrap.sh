#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
	echo "uv is required. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
	exit 1
fi

uv sync

echo "Bootstrap complete (uv)"
