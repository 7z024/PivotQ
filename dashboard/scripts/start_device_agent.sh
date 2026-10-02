#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${FUSION_AGENT_PYTHON:-${ROOT}/.venv/bin/python}"
export PYTHONPATH="$ROOT/dashboard:${PYTHONPATH:-}"
if [[ -z "${FUSION_REGISTRATION_TOKEN_FILE:-}" && -f "$ROOT/fusion-platform/secrets/registration.token" ]]; then
  export FUSION_REGISTRATION_TOKEN_FILE="$ROOT/fusion-platform/secrets/registration.token"
fi
ARGS=(--registry "${FUSION_REGISTRY_URL:-http://127.0.0.1:8787}" --token-file "${FUSION_REGISTRATION_TOKEN_FILE:-$ROOT/dashboard/secrets/registration.token}")
if [[ -n "${FUSION_QPU_CONFIG_FILE:-}" ]]; then ARGS+=(--qpu-config "$FUSION_QPU_CONFIG_FILE"); fi
exec "$PY" -m backend.device_agent "${ARGS[@]}" "$@"
