#!/usr/bin/env bash
# One-command setup + evaluation; ACTPROBE_PYTHON uses an existing environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "${ACTPROBE_PYTHON:-python3}" "$ROOT/scripts/reproduce_table1.py" --help
fi
if [[ -n "${ACTPROBE_PYTHON:-}" ]]; then
  TASK_PYTHON="$ACTPROBE_PYTHON"
else
  TASK_PYTHON="$ROOT/.venv-table1/bin/python"
  if [[ ! -x "$TASK_PYTHON" ]]; then
    python3 -m venv "$ROOT/.venv-table1"
  fi
  if ! "$TASK_PYTHON" -c 'import torch,numpy,scipy,sklearn; assert torch.__version__.split("+")[0]=="2.10.0" and numpy.__version__=="2.1.3" and scipy.__version__=="1.15.3" and sklearn.__version__=="1.7.2"' 2>/dev/null; then
    "$TASK_PYTHON" -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
    "$TASK_PYTHON" -m pip install -r "$ROOT/requirements-table1.txt"
  fi
fi
exec "$TASK_PYTHON" "$ROOT/scripts/reproduce_table1.py" "$@"
