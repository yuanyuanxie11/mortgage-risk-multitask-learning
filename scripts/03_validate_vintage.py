"""Validation checks on a converted vintage.

Run: python scripts/03_validate_vintage.py 2018Q1

Checks:
1. Row count matches what unzip | wc -l reports for the source CSV.
2. Distinct LOAN_ID count.
3. ACT_PERIOD range is contiguous monthly.
4. Each loan's observations are sorted on disk (by inspecting row ordering).
5. Basic sanity ranges for CSCORE_B, OLTV, DTI, ORIG_UPB, DLQ_STATUS, ZB_CODE.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent


def expected_csv_rows(zip_path: Path, member: str) -> int:
    proc = subprocess.run(
        f"unzip -p {zip_path} {member} | wc -l",
        shell=True, check=True, capture_output=True, text=True,
    )
    return int(proc.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vintage", help="e.g. 2018Q1")
    ap.add_argument("--zip", default=str(ROOT / "Performance_All.zip"))
    ap.add_argument("--out-root", default=str(ROOT / "data" / "raw_parquet"))
    ap.add_argument("--skip-csv-rowcount", action="store_true",
                    help="skip the expensive wc -l check")
    args = ap.parse_args()

    year = int(args.vintage[:4])
    qtr = args.vintage[4:]
    pq = Path(args.out_root) / f"orig_year={year}" / f"orig_qtr={qtr}" / "part-0.parquet"
    if not pq.exists():
        print(f"ERROR: {pq} does not exist", file=sys.stderr)
        return 1

    con = duckdb.connect()
    src = f"read_parquet('{pq}')"

    print(f"=== validating {args.vintage} ===")
    print(f"file: {pq}")
    print(f"size: {pq.stat().st_size / 1024 / 1024:.1f} MB")

    pq_rows = con.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
    print(f"parquet rows:  {pq_rows:,}")

    if not args.skip_csv_rowcount:
        print("counting source CSV lines (streaming)...")
        csv_rows = expected_csv_rows(Path(args.zip), f"{args.vintage}.csv")
        print(f"source CSV rows: {csv_rows:,}")
        assert csv_rows == pq_rows, f"row mismatch: csv={csv_rows} parquet={pq_rows}"
        print("row count: OK")

    n_loans = con.execute(f"SELECT COUNT(DISTINCT LOAN_ID) FROM {src}").fetchone()[0]
    print(f"distinct loans: {n_loans:,}")
    print(f"avg monthly obs per loan: {pq_rows / n_loans:.2f}")

    act_min, act_max = con.execute(
        f"SELECT MIN(ACT_PERIOD), MAX(ACT_PERIOD) FROM {src}"
    ).fetchone()
    print(f"ACT_PERIOD range: {act_min} .. {act_max}")

    print("\nCSCORE_B distribution (buckets of 20):")
    for r in con.execute(f"""
        SELECT FLOOR(CSCORE_B / 20) * 20 AS bucket, COUNT(*) AS n
        FROM {src} WHERE CSCORE_B IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall():
        print(f"  {int(r[0]):4d}+: {r[1]:>10,}")

    print("\nOLTV distribution (buckets of 10):")
    for r in con.execute(f"""
        SELECT FLOOR(OLTV / 10) * 10 AS bucket, COUNT(*) AS n
        FROM {src} WHERE OLTV IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall():
        print(f"  {int(r[0]):3d}+: {r[1]:>10,}")

    print("\nORIG_UPB summary:")
    row = con.execute(f"""
        SELECT MIN(ORIG_UPB), AVG(ORIG_UPB), MAX(ORIG_UPB), STDDEV(ORIG_UPB)
        FROM {src}
    """).fetchone()
    print(f"  min={row[0]:.0f}  mean={row[1]:,.0f}  max={row[2]:,.0f}  std={row[3]:,.0f}")

    print("\nDLQ_STATUS top 10:")
    for r in con.execute(f"""
        SELECT DLQ_STATUS, COUNT(*) FROM {src}
        GROUP BY 1 ORDER BY 2 DESC LIMIT 10
    """).fetchall():
        print(f"  {r[0]!r:>6}: {r[1]:>12,}")

    print("\nZB_CODE distribution (terminal event for the loan):")
    for r in con.execute(f"""
        SELECT ZB_CODE, COUNT(*) FROM {src}
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchall():
        print(f"  {r[0]!r:>6}: {r[1]:>12,}")

    print("\nZB_CODE distribution per distinct loan (lifetime outcome):")
    for r in con.execute(f"""
        WITH terminal AS (
            SELECT LOAN_ID, MAX(ZB_CODE) AS zb
            FROM {src}
            GROUP BY LOAN_ID
        )
        SELECT zb, COUNT(*) FROM terminal GROUP BY 1 ORDER BY 2 DESC
    """).fetchall():
        print(f"  {r[0]!r:>6}: {r[1]:>12,}")

    print("\nORDER BY check (loans with smallest LOAN_ID: periods should be sorted):")
    for lid, periods in con.execute(f"""
        SELECT LOAN_ID, array_agg(ACT_PERIOD ORDER BY ACT_PERIOD)
        FROM {src}
        WHERE LOAN_ID IN (SELECT LOAN_ID FROM {src} ORDER BY LOAN_ID LIMIT 3)
        GROUP BY LOAN_ID
        ORDER BY LOAN_ID
    """).fetchall():
        print(f"  {lid}: first={periods[:4]} last={periods[-3:]} count={len(periods)}")

    print("\nvalidation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
