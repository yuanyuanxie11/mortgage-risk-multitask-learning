"""Shared DuckDB view catalog for the Fannie Mae feature store.

Central place where Projects 1 (MTL), 2 (intervention), and notebooks all
register the same set of SQL views against the Hive-partitioned Parquet.
Avoids each consumer re-implementing glob paths and hive-partitioning flags.

Usage:

    from src.views import open_catalog
    con = open_catalog()
    con.execute("SELECT COUNT(*) FROM raw_monthly WHERE orig_year = 2018").fetchone()
    con.execute("SELECT * FROM features LIMIT 5").pl()
    con.execute("SELECT * FROM features_train LIMIT 5").pl()

Views defined:

    raw_monthly     - all monthly rows across every vintage; partition
                      pruning via `orig_year`, `orig_qtr` columns.
    features        - per-loan feature+label table (all vintages).
    features_eligible - features filtered to eligible=1 (valid labels).
    features_train  - eligible, orig_year in [2000, 2015].
    features_val    - eligible, orig_year in [2016, 2017].
    features_test   - eligible, orig_year in [2018, 2019].
    features_stress - eligible, orig_year == 2020.

A helper registers `vintage_orig_year(vintage VARCHAR)` derived column so
you can slice by orig_year on the features table even though its on-disk
partition key is `vintage=YYYYQn`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .splits import DEFAULT_BANDS, SplitBands

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT_DEFAULT = PROJECT_ROOT / "data" / "raw_parquet"
FEAT_ROOT_DEFAULT = PROJECT_ROOT / "data" / "features"


def open_catalog(
    raw_root: Path = RAW_ROOT_DEFAULT,
    feat_root: Path = FEAT_ROOT_DEFAULT,
    bands: SplitBands = DEFAULT_BANDS,
    threads: int = 16,
    memory_limit: str = "60GB",
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")

    raw_glob = str(raw_root / "orig_year=*" / "orig_qtr=*" / "part-*.parquet")
    feat_glob = str(feat_root / "vintage=*" / "part-*.parquet")

    con.execute(f"""
        CREATE OR REPLACE VIEW raw_monthly AS
        SELECT * FROM read_parquet('{raw_glob}', hive_partitioning=1)
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW features AS
        SELECT
            *,
            CAST(SUBSTR(vintage, 1, 4) AS INTEGER) AS orig_year,
            SUBSTR(vintage, 5, 2)                  AS orig_qtr
        FROM read_parquet('{feat_glob}', hive_partitioning=1)
    """)

    con.execute("""
        CREATE OR REPLACE VIEW features_eligible AS
        SELECT * FROM features WHERE eligible = 1
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW features_train AS
        SELECT * FROM features_eligible
        WHERE orig_year BETWEEN {bands.train[0]} AND {bands.train[1]}
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW features_val AS
        SELECT * FROM features_eligible
        WHERE orig_year BETWEEN {bands.val[0]} AND {bands.val[1]}
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW features_test AS
        SELECT * FROM features_eligible
        WHERE orig_year BETWEEN {bands.test[0]} AND {bands.test[1]}
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW features_stress AS
        SELECT * FROM features_eligible
        WHERE orig_year BETWEEN {bands.stress[0]} AND {bands.stress[1]}
    """)

    return con


def describe(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_type = 'VIEW'
        ORDER BY table_name
    """).fetchall()
    print("views:")
    for (v,) in rows:
        n = con.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0]
        print(f"  {v:<22} rows={n:>14,}")


if __name__ == "__main__":
    c = open_catalog()
    describe(c)
