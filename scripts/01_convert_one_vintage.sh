#!/usr/bin/env bash
# Convert a single vintage (e.g. 2018Q1) from the zip to Parquet.
#
# Usage:
#   scripts/01_convert_one_vintage.sh 2018Q1 [--overwrite]

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="${PY:-/opt/tljh/user/bin/python}"
VINTAGE="${1:?vintage required, e.g. 2018Q1}"
shift || true

mkdir -p logs
LOG="logs/convert_${VINTAGE}.log"

echo "[$(date -Iseconds)] converting ${VINTAGE} -> ${LOG}"
"$PY" -u -m src.ingest "$VINTAGE" --threads 16 --memory-limit 60GB "$@" \
    >"$LOG" 2>&1
echo "[$(date -Iseconds)] done ${VINTAGE}"
tail -1 "$LOG"
