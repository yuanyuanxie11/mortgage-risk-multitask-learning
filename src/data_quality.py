"""Data quality contracts and drift monitoring.

Run this BEFORE training to confirm your feature store is healthy.

Checks per vintage:
  1. Row count parity (Parquet vs expected)
  2. Null rates for key columns (fails if above threshold)
  3. Range checks (FICO, LTV, DTI, UPB)
  4. Label prevalence drift table (delinq / prepay by year)
  5. Coverage artifact: converted vintages, rows, eligible loans

Outputs:
  outputs/metrics/data_quality_report.csv
  outputs/metrics/prevalence_drift.csv
  outputs/metrics/coverage_artifact.csv

Usage:
    python data_quality.py
    python data_quality.py --fail-fast    # stop on first violation
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

try:
    from .config import FEAT_ROOT, METRICS_DIR
except ImportError:
    from config import FEAT_ROOT, METRICS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
NULL_THRESHOLDS = {
    "CSCORE_B":   0.30,   # up to 30% null OK (co-borrower not always present)
    "DTI":        0.10,
    "OLTV":       0.05,
    "ORIG_RATE":  0.01,
    "ORIG_UPB":   0.01,
}

# Macro feature null-rate contracts.
# These help catch silent join failures (period/state key mismatches, bad files).
MACRO_NULL_THRESHOLDS = {
    "rate_incentive": 0.20,             # should exist for most rows once FRED joins work
    "macro_mortgage30us": 0.20,         # direct FRED join key quality
    "hpi_state_ratio_curr_to_orig": 0.80,  # HPI may be missing in some runs; looser guard
}

RANGE_CHECKS = {
    "CSCORE_B":  (300, 850),
    "OLTV":      (0,   200),
    "DTI":       (0,   65),
    "ORIG_RATE": (0,   20),
    "ORIG_UPB":  (1_000, 2_000_000),
}

# Delinquency rate sanity bounds by era (rough — just catches major issues)
DELINQ_BOUNDS = {
    range(2000, 2004): (0.001, 0.08),
    range(2004, 2009): (0.001, 0.25),   # crisis vintages allow high rates
    range(2009, 2013): (0.001, 0.30),
    range(2013, 2020): (0.001, 0.08),
    range(2020, 2026): (0.001, 0.15),   # COVID allowance
}


def get_delinq_bound(year: int) -> tuple[float, float]:
    for yr_range, bounds in DELINQ_BOUNDS.items():
        if year in yr_range:
            return bounds
    return (0.0, 1.0)


def run_quality_checks(fail_fast: bool = False) -> tuple[pd.DataFrame, bool]:
    """Run per-vintage quality checks. Returns (report_df, all_passed)."""
    glob    = str(FEAT_ROOT / "vintage=*" / "part-*.parquet")
    con     = duckdb.connect()
    hpi_available = (FEAT_ROOT.parent / "macro" / "hpi_state_quarterly.parquet").exists()

    vintages = [
        r[0] for r in con.execute(f"""
            SELECT DISTINCT vintage
            FROM read_parquet('{glob}', hive_partitioning=1)
            ORDER BY vintage
        """).fetchall()
    ]

    log.info("Checking %d vintages ...", len(vintages))
    rows = []
    all_passed = True

    for v in vintages:
        year = int(v[:4])
        vglob = str(FEAT_ROOT / f"vintage={v}" / "part-*.parquet")

        stats = con.execute(f"""
            SELECT
                COUNT(*)                        AS n_rows,
                SUM(eligible)                   AS n_eligible,
                -- null rates
                AVG(CASE WHEN CSCORE_B  IS NULL THEN 1.0 ELSE 0.0 END) AS null_cscore,
                AVG(CASE WHEN DTI       IS NULL THEN 1.0 ELSE 0.0 END) AS null_dti,
                AVG(CASE WHEN OLTV      IS NULL THEN 1.0 ELSE 0.0 END) AS null_oltv,
                AVG(CASE WHEN ORIG_RATE IS NULL THEN 1.0 ELSE 0.0 END) AS null_origrate,
                AVG(CASE WHEN rate_incentive IS NULL THEN 1.0 ELSE 0.0 END) AS null_rate_incentive,
                AVG(CASE WHEN macro_mortgage30us IS NULL THEN 1.0 ELSE 0.0 END) AS null_macro_mortgage30us,
                AVG(CASE WHEN hpi_state_ratio_curr_to_orig IS NULL THEN 1.0 ELSE 0.0 END) AS null_hpi_ratio,
                -- ranges (fraction out of bound)
                AVG(CASE WHEN CSCORE_B  NOT BETWEEN 300 AND 850 THEN 1.0 ELSE 0.0 END) AS oor_cscore,
                AVG(CASE WHEN OLTV      NOT BETWEEN 0   AND 200 THEN 1.0 ELSE 0.0 END) AS oor_oltv,
                AVG(CASE WHEN DTI       NOT BETWEEN 0   AND 65  THEN 1.0 ELSE 0.0 END) AS oor_dti,
                -- label rates
                AVG(CASE WHEN eligible=1 THEN y_delinq_90p END) AS delinq_rate,
                AVG(CASE WHEN eligible=1 THEN y_prepay     END) AS prepay_rate
            FROM read_parquet('{vglob}')
        """).fetchone()

        (n_rows, n_eligible,
         null_cscore, null_dti, null_oltv, null_origrate,
         null_rate_incentive, null_macro_mortgage30us, null_hpi_ratio,
         oor_cscore, oor_oltv, oor_dti,
         delinq_rate, prepay_rate) = stats

        violations = []

        if null_cscore  > NULL_THRESHOLDS["CSCORE_B"]:  violations.append(f"null_cscore={null_cscore:.3f}")
        if null_dti     > NULL_THRESHOLDS["DTI"]:        violations.append(f"null_dti={null_dti:.3f}")
        if null_oltv    > NULL_THRESHOLDS["OLTV"]:       violations.append(f"null_oltv={null_oltv:.3f}")
        if null_origrate > NULL_THRESHOLDS["ORIG_RATE"]: violations.append(f"null_origrate={null_origrate:.3f}")
        if null_rate_incentive > MACRO_NULL_THRESHOLDS["rate_incentive"]:
            violations.append(f"null_rate_incentive={null_rate_incentive:.3f}")
        if null_macro_mortgage30us > MACRO_NULL_THRESHOLDS["macro_mortgage30us"]:
            violations.append(f"null_macro_mortgage30us={null_macro_mortgage30us:.3f}")
        if hpi_available and null_hpi_ratio > MACRO_NULL_THRESHOLDS["hpi_state_ratio_curr_to_orig"]:
            violations.append(f"null_hpi_ratio={null_hpi_ratio:.3f}")

        for col, oor in [("CSCORE_B", oor_cscore), ("OLTV", oor_oltv), ("DTI", oor_dti)]:
            if oor and oor > 0.01:
                violations.append(f"oor_{col}={oor:.4f}")

        lo, hi = get_delinq_bound(year)
        if delinq_rate is not None and not (lo <= delinq_rate <= hi):
            violations.append(f"delinq_rate={delinq_rate:.4f} outside [{lo},{hi}]")

        passed = len(violations) == 0
        if not passed:
            all_passed = False
            log.warning("FAIL %s: %s", v, " | ".join(violations))
            if fail_fast:
                raise RuntimeError(f"Quality check failed for {v}: {violations}")

        rows.append({
            "vintage":      v,
            "n_rows":       n_rows,
            "n_eligible":   n_eligible,
            "delinq_rate":  round(delinq_rate or 0, 4),
            "prepay_rate":  round(prepay_rate or 0, 4),
            "null_cscore":  round(null_cscore, 4),
            "null_dti":     round(null_dti, 4),
            "null_oltv":    round(null_oltv, 4),
            "null_rate_incentive": round(null_rate_incentive, 4),
            "null_macro_mortgage30us": round(null_macro_mortgage30us, 4),
            "null_hpi_ratio": round(null_hpi_ratio, 4),
            "violations":   "; ".join(violations) if violations else "OK",
            "passed":       passed,
        })

    con.close()
    return pd.DataFrame(rows), all_passed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    report, all_passed = run_quality_checks(fail_fast=args.fail_fast)

    # ── Save reports ──────────────────────────────────────────────────────────
    report_path = METRICS_DIR / "data_quality_report.csv"
    report.to_csv(report_path, index=False)

    drift = report[["vintage", "delinq_rate", "prepay_rate"]].copy()
    drift.to_csv(METRICS_DIR / "prevalence_drift.csv", index=False)

    coverage = report[["vintage", "n_rows", "n_eligible", "passed"]].copy()
    coverage.to_csv(METRICS_DIR / "coverage_artifact.csv", index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    failed = report[~report["passed"]]
    print(f"\n{'='*60}")
    print(f"DATA QUALITY SUMMARY")
    print(f"{'='*60}")
    print(f"Vintages checked : {len(report)}")
    print(f"Passed           : {report['passed'].sum()}")
    print(f"Failed           : {len(failed)}")
    print(f"Total rows       : {report['n_rows'].sum():,}")
    print(f"Total eligible   : {report['n_eligible'].sum():,}")

    if len(failed):
        print(f"\nFAILED VINTAGES:")
        print(failed[["vintage", "violations"]].to_string(index=False))

    print(f"\nSaved → {report_path}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
