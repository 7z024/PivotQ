#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
if [[ "${ROLE}" != "head" && "${ROLE}" != "worker" ]]; then
  echo "usage: $0 head|worker" >&2
  exit 2
fi

UV_BIN="${RAY_QUANTUM_UV_BIN:-/opt/ray-quantum/bin/uv}"
PYTHON_BIN="${RAY_QUANTUM_PYTHON_BIN:-/opt/ray-quantum/venv/bin/python}"
TORCH_WHEEL="https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp312-cp312-linux_x86_64.whl"
PYPI_INDEX_URL="${RAY_QUANTUM_PYPI_INDEX_URL:-https://pypi.org/simple}"

test -x "${UV_BIN}"
test -x "${PYTHON_BIN}"

echo "INSTALL_ROLE=${ROLE}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PYPI_INDEX_URL=${PYPI_INDEX_URL}"
"${PYTHON_BIN}" --version
"${UV_BIN}" pip install \
  --python "${PYTHON_BIN}" \
  --index-url "${PYPI_INDEX_URL}" \
  "numpy==1.26.4"

if [[ "${ROLE}" == "worker" ]]; then
  echo "Installing the pinned official PyTorch 2.5.1 CUDA 12.1 CPython 3.12 wheel"
  "${UV_BIN}" pip install \
    --python "${PYTHON_BIN}" \
    --index-url "${PYPI_INDEX_URL}" \
    "${TORCH_WHEEL}"
fi

"${PYTHON_BIN}" -m examples.hybrid_cpu_gpu_experiments.preflight --role "${ROLE}"
echo "DEPENDENCY_INSTALL_RESULT=PASS"
