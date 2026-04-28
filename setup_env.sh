#!/usr/bin/env bash
# setup_env.sh — configure Python env and install dependencies.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#   ./setup_env.sh --venv        # force local virtualenv ./venv
#
# Default behavior prefers a local virtualenv for reproducibility and to avoid
# mutating system Python packages.

set -euo pipefail

USE_VENV=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) USE_VENV=1; shift ;;
    --system) USE_VENV=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
REQ="$ROOT/requirements.txt"
VENV_DIR="$ROOT/venv"

find_python() {
  for candidate in \
    /opt/tljh/user/bin/python3 \
    /opt/tljh/user/bin/python \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"; do
    if [[ -n "${candidate:-}" && -x "$candidate" ]]; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3,10) else 1)
PY
      then
        echo "$candidate"
        return
      fi
    fi
  done
  echo ""
}

PY="$(find_python)"
if [[ -z "$PY" ]]; then
  echo "ERROR: No Python >= 3.10 found." >&2
  exit 1
fi
echo "Python: $PY ($("$PY" --version))"

if [[ $USE_VENV -eq 1 ]]; then
  echo "Mode: virtualenv ($VENV_DIR)"
  "$PY" -m venv "$VENV_DIR"
  PY="$VENV_DIR/bin/python"
  PIP="$VENV_DIR/bin/pip"
else
  echo "Mode: system install"
  PIP="$PY -m pip"
fi

echo "[1/3] Upgrade pip"
$PIP install --upgrade pip

echo "[2/3] Install requirements"
if ! $PIP install -r "$REQ"; then
  echo "Retrying with --break-system-packages (system environment workaround)"
  $PIP install --break-system-packages -r "$REQ"
fi

echo "[3/3] Verify imports"
"$PY" - <<'PY'
import importlib, sys
mods = [
  ("duckdb", "duckdb"),
  ("pyarrow", "pyarrow"),
  ("polars", "polars"),
  ("pandas", "pandas"),
  ("xgboost", "xgboost"),
  ("sklearn", "scikit-learn"),
  ("shap", "shap"),
  ("matplotlib", "matplotlib"),
  ("requests", "requests"),
  ("tqdm", "tqdm"),
  ("torch", "torch"),
]
failed=[]
for mod,name in mods:
  try:
    m=importlib.import_module(mod)
    print(f"  ✓ {name:<16} {getattr(m,'__version__','?')}")
  except Exception:
    print(f"  ✗ {name:<16} MISSING")
    failed.append(name)
if failed:
  # torch can fail when a system-wide CUDA wheel is injected; surface clearly.
  non_torch = [x for x in failed if x != "torch"]
  if non_torch:
    print("Missing:", non_torch)
    sys.exit(1)
  print("\nPyTorch import failed. This is often a CUDA wheel mismatch.")
  print("Recommended fix (CPU wheel):")
  print("  pip uninstall -y torch torchvision torchaudio")
  print("  pip install --index-url https://download.pytorch.org/whl/cpu torch")
print("All imports OK.")
PY

echo "PY=$PY" > "$ROOT/.python_path"
echo "Done."
if [[ $USE_VENV -eq 1 ]]; then
  echo "Activate with: source $VENV_DIR/bin/activate"
fi
