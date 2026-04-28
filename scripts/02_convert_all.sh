#!/usr/bin/env bash
# Parallel conversion of every vintage in Performance_All.zip to Hive-partitioned
# Parquet.
#
# Conservative defaults favor stability on large vintages. Tune via env vars:
#   PARALLEL_WORKERS, INGEST_THREADS, INGEST_MEMORY, INGEST_TMP_ROOT
#
# Skip already-completed vintages. To force a re-conversion pass --overwrite.
#
# Usage:
#   scripts/02_convert_all.sh                          # all vintages, skip done
#   scripts/02_convert_all.sh --parallel 6             # tune concurrency
#   scripts/02_convert_all.sh --only 2015,2016,2017    # year filter
#   scripts/02_convert_all.sh --overwrite              # force re-run

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${PY:-/opt/tljh/user/bin/python}"
ZIP="${ZIP:-$ROOT/Performance_All.zip}"
OUT="${OUT:-$ROOT/data/raw_parquet}"

PARALLEL="${PARALLEL_WORKERS:-2}"
INGEST_THREADS="${INGEST_THREADS:-4}"
INGEST_MEMORY="${INGEST_MEMORY:-20GB}"
INGEST_TMP_ROOT="${INGEST_TMP_ROOT:-$ROOT/data/duckdb_tmp}"
ONLY_YEARS=""
OVERWRITE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel) PARALLEL="$2"; shift 2 ;;
        --only)     ONLY_YEARS="$2"; shift 2 ;;
        --overwrite) OVERWRITE="--overwrite"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p logs
SUMMARY="logs/convert_all_summary.tsv"
: >"$SUMMARY"

# enumerate vintages from the zip (2000Q1 ... 2025Q3) in ascending order
mapfile -t VINTAGES < <(
    unzip -l "$ZIP" \
      | awk 'match($NF, /^([0-9]{4})Q[1-4]\.csv$/, m){print $NF}' \
      | sed 's/\.csv$//' \
      | sort -u
)
echo "found ${#VINTAGES[@]} vintages in $ZIP"

run_one() {
    local v="$1"
    local year="${v:0:4}"
    local qtr="${v:4}"
    local out_pq="$OUT/orig_year=$year/orig_qtr=$qtr/part-0.parquet"

    if [[ -f "$out_pq" && -z "$OVERWRITE" ]]; then
        printf '%s\tSKIP\t%s\texisting\n' "$v" "$out_pq" >>"$SUMMARY"
        return 0
    fi

    if [[ -n "$ONLY_YEARS" ]]; then
        if ! grep -q -w "$year" <<<"${ONLY_YEARS//,/ }"; then
            printf '%s\tFILTER\t-\tnot-in-only\n' "$v" >>"$SUMMARY"
            return 0
        fi
    fi

    local log="logs/convert_${v}.log"
    local tmp_dir="$INGEST_TMP_ROOT/$v"
    local t0=$(date +%s)
    if "$PY" -u -m src.ingest "$v" \
          --threads "$INGEST_THREADS" \
          --memory-limit "$INGEST_MEMORY" \
          --tmp-root "$tmp_dir" \
          $OVERWRITE \
          >"$log" 2>&1; then
        local t1=$(date +%s)
        local line
        line=$(tail -1 "$log" | tr -d '\n')
        printf '%s\tOK\t%ss\t%s\n' "$v" "$((t1-t0))" "$line" >>"$SUMMARY"
        echo "[$(date -Iseconds)] OK   $v ($((t1-t0))s)"
    else
        printf '%s\tFAIL\t-\tsee %s\n' "$v" "$log" >>"$SUMMARY"
        echo "[$(date -Iseconds)] FAIL $v (see $log)" >&2
    fi
}

export -f run_one
export PY ZIP OUT OVERWRITE ONLY_YEARS SUMMARY INGEST_THREADS INGEST_MEMORY INGEST_TMP_ROOT

# GNU parallel if available, else xargs
if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${VINTAGES[@]}" \
        | parallel -j "$PARALLEL" --lb run_one {}
else
    printf '%s\n' "${VINTAGES[@]}" \
        | xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}
fi

echo
echo "=== summary ==="
column -t -s $'\t' "$SUMMARY" | head -120
echo "..."
echo "full summary: $SUMMARY"
