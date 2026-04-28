"""Vintage-based train / val / test / stress splits.

Rule: a loan's split is determined by its origination vintage (`orig_year`),
never by a random row sample. This prevents any form of temporal leakage
(feature info from later observations bleeding into training).

Default bands:

    train  : orig_year in [2000, 2015]   (16 years, the bulk of the data)
    val    : orig_year in [2016, 2017]
    test   : orig_year in [2018, 2019]
    stress : orig_year == 2020           (COVID forbearance regime)
    holdout: orig_year >= 2021           (very recent; label window may be short)

Everything in this module is pure metadata — it returns DuckDB SQL WHERE
clauses or Polars filter expressions. No data is read until you execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEAT_ROOT_DEFAULT = PROJECT_ROOT / "data" / "features"


@dataclass(frozen=True)
class SplitBands:
    train: tuple[int, int] = (2000, 2015)
    val: tuple[int, int] = (2016, 2017)
    test: tuple[int, int] = (2018, 2019)
    stress: tuple[int, int] = (2020, 2020)
    # holdout = anything >= 2021 until we have enough horizon

    def split_of(self, orig_year: int) -> str:
        if self.train[0] <= orig_year <= self.train[1]:
            return "train"
        if self.val[0] <= orig_year <= self.val[1]:
            return "val"
        if self.test[0] <= orig_year <= self.test[1]:
            return "test"
        if self.stress[0] <= orig_year <= self.stress[1]:
            return "stress"
        return "holdout"


DEFAULT_BANDS = SplitBands()


def vintage_dir_pattern(feat_root: Path, year_lo: int, year_hi: int) -> str:
    """Return a glob over feature parquet files for a year range."""
    return str(feat_root / f"vintage={{{','.join(f'{y}Q{q}' for y in range(year_lo, year_hi + 1) for q in (1, 2, 3, 4))}}}" / "part-*.parquet")


def load_split(
    split: str,
    feat_root: Path = FEAT_ROOT_DEFAULT,
    bands: SplitBands = DEFAULT_BANDS,
    only_eligible: bool = True,
    columns: list[str] | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
):
    """Load a named split ('train' | 'val' | 'test' | 'stress') as a Polars DataFrame.

    Only loads parquet partitions whose origination year falls in the band;
    partitioning pruning happens via explicit glob, not WHERE clause. The
    DuckDB connection is created on demand if not provided.
    """
    lo, hi = {
        "train": bands.train,
        "val": bands.val,
        "test": bands.test,
        "stress": bands.stress,
    }[split]

    pattern = vintage_dir_pattern(feat_root, lo, hi)
    close = False
    if con is None:
        con = duckdb.connect()
        close = True

    col_sql = ",".join(columns) if columns else "*"
    q = f"SELECT {col_sql} FROM read_parquet('{pattern}', hive_partitioning=1)"
    if only_eligible:
        q += " WHERE eligible = 1"

    try:
        return con.execute(q).pl()
    finally:
        if close:
            con.close()


def describe_splits(
    feat_root: Path = FEAT_ROOT_DEFAULT,
    bands: SplitBands = DEFAULT_BANDS,
) -> "duckdb.DuckDBPyConnection":
    """Print a split summary table for diagnostics."""
    con = duckdb.connect()
    pattern = str(feat_root / "vintage=*" / "part-*.parquet")
    con.execute(f"""
        CREATE VIEW feat AS
        SELECT
            *,
            CAST(REGEXP_EXTRACT(vintage, '^([0-9]{{4}})', 1) AS INTEGER) AS orig_year
        FROM read_parquet('{pattern}', hive_partitioning=1)
    """)
    con.execute(f"""
        CREATE VIEW feat_split AS
        SELECT
            vintage, orig_year,
            CASE
                WHEN orig_year BETWEEN {bands.train[0]} AND {bands.train[1]} THEN 'train'
                WHEN orig_year BETWEEN {bands.val[0]}   AND {bands.val[1]}   THEN 'val'
                WHEN orig_year BETWEEN {bands.test[0]}  AND {bands.test[1]}  THEN 'test'
                WHEN orig_year BETWEEN {bands.stress[0]} AND {bands.stress[1]} THEN 'stress'
                ELSE 'holdout'
            END AS split,
            eligible, y_prepay, y_delinq_90p
        FROM feat
    """)
    summary = con.execute("""
        SELECT
            split,
            COUNT(*)                                      AS loans,
            SUM(eligible)                                 AS eligible,
            SUM(CASE WHEN eligible = 1 THEN y_prepay END) AS prepays,
            SUM(CASE WHEN eligible = 1 THEN y_delinq_90p END) AS delinq90p,
            ROUND(AVG(CASE WHEN eligible = 1 THEN y_prepay END), 4) AS prepay_rate,
            ROUND(AVG(CASE WHEN eligible = 1 THEN y_delinq_90p END), 4) AS delinq_rate
        FROM feat_split
        GROUP BY split
        ORDER BY CASE split
            WHEN 'train' THEN 0
            WHEN 'val'   THEN 1
            WHEN 'test'  THEN 2
            WHEN 'stress' THEN 3
            ELSE 4 END
    """).fetchall()
    print(f"{'split':<8} {'loans':>12} {'eligible':>12} {'prepay':>10} {'delq90p':>10} {'p_rate':>8} {'d_rate':>8}")
    for r in summary:
        print(f"{r[0]:<8} {r[1]:>12,} {r[2]:>12,} {r[3]:>10,} {r[4]:>10,} {r[5]!s:>8} {r[6]!s:>8}")
    return con


if __name__ == "__main__":
    describe_splits()
