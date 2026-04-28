"""Load train/val/test splits from the shared DuckDB feature store.

Reads the Hive-partitioned Parquet produced by src/features.py.
Returns pandas DataFrames with features + labels, only eligible loans.

Usage:
    from src.data import load_split
    X_train, y_train = load_split("train", target="delinq")
    X_val,   y_val   = load_split("val",   target="delinq")
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

try:
    from .config import FEAT_ROOT, ALL_FEATURES, CAT_FEATURES, TARGETS
except ImportError:
    from config import FEAT_ROOT, ALL_FEATURES, CAT_FEATURES, TARGETS

log = logging.getLogger(__name__)

# Vintage year bands — must match src/splits.py
SPLIT_BANDS = {
    "train":  (2000, 2015),
    "val":    (2016, 2017),
    "test":   (2018, 2019),
    "stress": (2020, 2020),
}

SplitName  = Literal["train", "val", "test", "stress"]
TargetName = Literal["delinq", "prepay"]


def _glob(feat_root: Path) -> str:
    return str(feat_root / "vintage=*" / "part-*.parquet")


def load_split(
    split: SplitName,
    target: TargetName,
    feat_root: Path = FEAT_ROOT,
    threads: int = 16,
    memory_limit: str = "60GB",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the requested split and target.

    Only eligible loans (eligible=1) are returned.
    Categorical columns are cast to pandas Categorical dtype so XGBoost
    can handle them natively with enable_categorical=True.

    Parameters
    ----------
    split   : "train" | "val" | "test" | "stress"
    target  : "delinq" (y_delinq_90p) | "prepay" (y_prepay)
    columns : optional feature subset; defaults to ALL_FEATURES from config
    """
    lo, hi = SPLIT_BANDS[split]
    glob = _glob(feat_root)
    label_col = TARGETS[target]
    feat_cols = columns or ALL_FEATURES

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")

    select_cols = ", ".join(feat_cols + [label_col, "LOAN_ID", "vintage"])
    sql = f"""
        SELECT
            {select_cols},
            CAST(SUBSTR(vintage, 1, 4) AS INTEGER) AS orig_year
        FROM read_parquet('{glob}', hive_partitioning=1)
        WHERE eligible = 1
          AND CAST(SUBSTR(vintage, 1, 4) AS INTEGER) BETWEEN {lo} AND {hi}
    """
    log.info("Loading %s split (%d–%d) target=%s ...", split, lo, hi, target)
    df = con.execute(sql).df()
    con.close()
    if df.empty:
        raise RuntimeError(
            f"No rows found for split={split} ({lo}-{hi}). "
            "Build features for vintages in this band first."
        )

    # Cast categoricals for XGBoost native handling
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    X = df[feat_cols]
    y = df[label_col].astype(int)

    log.info(
        "  %s: %d loans | %s prevalence=%.4f",
        split, len(X), target, y.mean()
    )
    return X, y


def load_all_splits(target: TargetName) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Convenience: load train/val/test/stress in one call."""
    return {s: load_split(s, target) for s in ["train", "val", "test", "stress"]}


def prevalence_table(feat_root: Path = FEAT_ROOT) -> pd.DataFrame:
    """Return per-vintage label prevalence — useful for drift monitoring."""
    glob = str(feat_root / "vintage=*" / "part-*.parquet")
    con  = duckdb.connect()
    df   = con.execute(f"""
        SELECT
            vintage,
            COUNT(*)                                          AS loans,
            SUM(eligible)                                     AS eligible,
            ROUND(AVG(CASE WHEN eligible=1 THEN y_delinq_90p END), 4) AS delinq_rate,
            ROUND(AVG(CASE WHEN eligible=1 THEN y_prepay     END), 4) AS prepay_rate
        FROM read_parquet('{glob}', hive_partitioning=1)
        GROUP BY vintage
        ORDER BY vintage
    """).df()
    con.close()
    return df
