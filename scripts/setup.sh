#!/usr/bin/env bash
# Local (sim / eval / demo / OpenVINO export) environment. See README.md.
set -euo pipefail
command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
uv sync --extra local
uv run python scripts/check_env.py
