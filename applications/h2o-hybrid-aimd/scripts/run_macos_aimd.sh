#!/usr/bin/env bash
# Run the existing standalone scientific pipeline with an isolated Mac environment.
set -euo pipefail

AIMD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIMD_STEPS="${1:-10}"
if [[ $# -gt 1 || ! "${AIMD_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash scripts/run_macos_aimd.sh [positive step count; default 10]" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This environment is for Apple Silicon Macs (macOS 14 or newer)." >&2
  exit 2
fi
command -v uv >/dev/null || { echo "Install uv before running this script." >&2; exit 2; }
cd "${AIMD_ROOT}"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Expected Python 3.12"'
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple \
  -r requirements-macos-cpu.txt

export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
mkdir -p outputs
AIMD_OUTPUT="$(mktemp -d "${AIMD_ROOT}/outputs/macos_cpu_${AIMD_STEPS}_steps_XXXXXX")"
uv pip freeze --python .venv/bin/python > "${AIMD_OUTPUT}/requirements-installed.txt"
echo "AIMD output: ${AIMD_OUTPUT}"
.venv/bin/python -B scripts/run_aimd.py \
  --config configs/h2o_aimd.yaml \
  --steps "${AIMD_STEPS}" \
  --output-dir "${AIMD_OUTPUT}" 2>&1 | tee "${AIMD_OUTPUT}/run.log"
echo "AIMD completed: ${AIMD_OUTPUT}/aimd/run_summary.json"
