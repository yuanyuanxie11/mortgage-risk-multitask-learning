#!/usr/bin/env bash
# run_baseline.sh — execute the full XGBoost baseline pipeline in order.
#
# Usage:
#   ./run_baseline.sh                    # full run
#   ./run_baseline.sh --skip-quality     # skip data checks (faster dev loop)
#   ./run_baseline.sh --target delinq    # one target only
#
# Expected runtime on deepdish (32-core):
#   data_quality.py   ~5 min
#   train_xgb.py      ~20-40 min per target (early stopping)
#   eval.py           ~5 min
#   shap_report.py    ~10 min (5K sample)
#   Total             ~1.5h for both targets

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
SKIP_QUALITY=0
TARGET_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-quality) SKIP_QUALITY=1; shift ;;
        --target) TARGET_ARG="--target $2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "========================================"
echo " Fannie Mae XGBoost Baseline Pipeline"
echo "========================================"
echo " Python:  $PY"
echo " Started: $(date)"
echo "========================================"

# Step 1: Data quality contracts
if [[ $SKIP_QUALITY -eq 0 ]]; then
    echo ""
    echo "[1/4] Running data quality checks ..."
    "$PY" -m src.data_quality
    echo "      Done."
else
    echo "[1/4] Skipping data quality checks."
fi

# Step 2: Train XGBoost baselines
echo ""
echo "[2/4] Training XGBoost baselines ..."
"$PY" -m src.train_xgb $TARGET_ARG
echo "      Done."

# Step 3: Evaluate — produce baseline table
echo ""
echo "[3/4] Evaluating on val + test splits ..."
"$PY" -m src.eval $TARGET_ARG
echo "      Done."

# Step 4: SHAP reports
echo ""
echo "[4/4] Generating SHAP reports ..."
"$PY" -m src.shap_report $TARGET_ARG
echo "      Done."

echo ""
echo "========================================"
echo " Pipeline complete: $(date)"
echo "========================================"
echo ""
echo " Key outputs:"
echo "   outputs/metrics/baseline_table.csv   ← the anchor benchmark"
echo "   outputs/metrics/data_quality_report.csv"
echo "   outputs/metrics/prevalence_drift.csv"
echo "   outputs/plots/pr_curve_*.png"
echo "   outputs/plots/calibration_*.png"
echo "   outputs/shap/shap_comparison_val.png ← MTL motivation chart"
echo "   outputs/shap/shap_top20_*.csv"
echo ""
echo " Next step: train MTL model and compare against baseline_table.csv"
