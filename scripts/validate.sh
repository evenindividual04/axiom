#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m compileall src evals models data/synthetic main.py
uv run python main.py status

echo "Validation complete"
