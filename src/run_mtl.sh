#!/usr/bin/env bash
# run_mtl.sh — train and evaluate the PyTorch multi-task model.
#
# Usage:
#   ./src/run_mtl.sh
#   ./src/run_mtl.sh --epochs 30 --batch-size 4096

set -euo pipefail
cd "$(dirname "$0")/.."

resolve_python() {
  if [[ -n "${PY:-}" ]]; then
    echo "$PY"
    return
  fi
  if [[ -f ".python_path" ]]; then
    # shellcheck disable=SC1091
    source ".python_path" || true
    if [[ -n "${PY:-}" ]]; then
      echo "$PY"
      return
    fi
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    echo "$VIRTUAL_ENV/bin/python"
    return
  fi
  if [[ -x "./venv/bin/python" ]]; then
    echo "./venv/bin/python"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    echo "$(command -v python)"
    return
  fi
  echo "/opt/tljh/user/bin/python"
}

PY="$(resolve_python)"

echo "========================================"
echo " PyTorch Multi-Task Pipeline"
echo "========================================"
echo " Python:  $PY"
echo " Started: $(date)"
echo "========================================"

echo ""
echo "[1/2] Train MTL model ..."
"$PY" -m src.train_mtl "$@"

echo ""
echo "[2/2] Evaluate MTL model ..."
"$PY" -m src.eval_mtl

echo ""
echo "========================================"
echo " MTL pipeline complete: $(date)"
echo "========================================"
echo ""
echo "Outputs:"
echo "  outputs/models/mtl_best.pt"
echo "  outputs/models/mtl_preprocessor.pkl"
echo "  outputs/metrics/mtl_training_history.csv"
echo "  outputs/metrics/mtl_eval_table.csv"
