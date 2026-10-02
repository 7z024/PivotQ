#!/usr/bin/env bash
set -euo pipefail

RAY_BIN="${RAY_QUANTUM_RAY_BIN:-/opt/ray-quantum/venv/bin/ray}"
test -x "${RAY_BIN}"
"${RAY_BIN}" stop --force

RAY_PROCESS_COUNT="$(pgrep -af '[r]aylet|[g]cs_server|[d]ashboard[.]py|[r]untime_env' | wc -l || true)"
echo "RAY_PROCESS_COUNT=${RAY_PROCESS_COUNT}"
if [[ "${RAY_PROCESS_COUNT}" != "0" ]]; then
  echo "STOP_RESULT=FAIL"
  exit 1
fi
echo "STOP_RESULT=PASS"
