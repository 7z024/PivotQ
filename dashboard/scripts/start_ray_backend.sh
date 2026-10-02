#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${FUSION_RAY_PYTHON:-$ROOT/.venv/bin/python}"
export FUSION_DEVICE_REGISTRATION="${FUSION_DEVICE_REGISTRATION:-network}"
exec "$PY" "$ROOT/dashboard/serve.py" --mode ray "$@"
