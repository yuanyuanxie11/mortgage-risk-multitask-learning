"""Leakage-safe feature & label construction, per origination vintage.

Produces a single per-loan feature table from the Hive-partitioned raw Parquet.

Windows (measured in LOAN_AGE months since origination):

    feature window: LOAN_AGE in [0, 12]    (inclusive, 13 observations at most)
    label   window: LOAN_AGE in [13, 24]   (inclusive, 12 observations at most)

Labels are defined as competing risks:

    y_prepay   = 1 iff ZB_CODE == '01'          in the label window
    y_delinq   = 1 iff numeric DLQ_STATUS >= 3  in the label window
                     (i.e. 90+ days past due at any point)
    y_censored = neither event and no forbearance code during the label window

A loan is "eligible" (i.e. has observable labels) iff:

    max(LOAN_AGE) >= 24                -- observed long enough to see the window
    OR any terminal event (ZB_CODE IS NOT NULL) -- a terminated loan is always
                                          resolved regardless of horizon

Non-eligible loans are *kept* in the feature table with `eligible = 0` so we
can partition them off downstream (and report coverage).

Write path:

    data/features/vintage={YYYY}{Qn}/part-0.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT_DEFAULT = PROJECT_ROOT / "data" / "raw_parquet"
FEAT_ROOT_DEFAULT = PROJECT_ROOT / "data" / "features"
MACRO_ROOT_DEFAULT = PROJECT_ROOT / "data" / "macro"
TMP_DEFAULT = PROJECT_ROOT / "data" / "duckdb_tmp"

FEATURE_WINDOW_MAX = 12
LABEL_WINDOW_MIN = 13
LABEL_WINDOW_MAX = 24


def build_features_sql(
    raw_glob: str,
    rates_glob: str | None = None,
    hpi_glob: str | None = None,
) -> str:
    """Return the SQL that turns raw monthly rows into one row per loan.

    All aggregation happens inside DuckDB (vectorized, parallel). No Python
    loops over rows. We key on LOAN_ID and carry origination fields forward
    from LOAN_AGE = 0 (or the earliest available observation per loan).
    """
    # Fannie Mae writes LOAN_AGE = NULL on the terminal (zero-balance) row.
    # We need that row — it carries ZB_CODE. So we recompute a canonical
    # loan age from MMYYYY dates:
    #     month_idx(d) = year(d)*12 + month(d)
    #     age = month_idx(ACT_PERIOD) - month_idx(ORIG_DATE)
    # and COALESCE to the file's LOAN_AGE when present.
    use_rates = bool(rates_glob)
    use_hpi = bool(hpi_glob)
    use_macro = use_rates or use_hpi
    macro_ctes = ""
    macro_join = ""
    macro_select = """
        CAST(NULL AS DOUBLE) AS macro_mortgage30us,
        CAST(NULL AS DOUBLE) AS macro_unrate,
        CAST(NULL AS DOUBLE) AS macro_dgs10,
        CAST(NULL AS DOUBLE) AS macro_term_spread_10y3m,
        CAST(NULL AS DOUBLE) AS hpi_state_curr,
        CAST(NULL AS DOUBLE) AS hpi_state_orig,
        CAST(NULL AS DOUBLE) AS hpi_state_growth_1y,
        CAST(NULL AS DOUBLE) AS hpi_state_ratio_curr_to_orig,
        CAST(NULL AS DOUBLE) AS rate_incentive,
        CAST(NULL AS DOUBLE) AS est_current_ltv_from_oltv
    """
    if use_macro:
        rates_cte = ""
        hpi_cte = ""
        rates_join = ""
        hpi_curr_join = ""
        hpi_orig_join = ""
        if use_rates:
            rates_cte = f""",
    rates AS (
        SELECT
            period_mmyyyy,
            mortgage30us,
            unrate,
            dgs10,
            gs3m,
            dgs10 - gs3m AS term_spread_10y3m
        FROM read_parquet('{rates_glob}')
    )"""
            rates_join = """
        LEFT JOIN rates r
          ON r.period_mmyyyy = s2.ACT_PERIOD
"""
        rate_m30_expr = "MAX_BY(r.mortgage30us, s2.loan_age_eff)" if use_rates else "CAST(NULL AS DOUBLE)"
        rate_unrate_expr = "MAX_BY(r.unrate, s2.loan_age_eff)" if use_rates else "CAST(NULL AS DOUBLE)"
        rate_dgs10_expr = "MAX_BY(r.dgs10, s2.loan_age_eff)" if use_rates else "CAST(NULL AS DOUBLE)"
        rate_term_expr = "MAX_BY(r.term_spread_10y3m, s2.loan_age_eff)" if use_rates else "CAST(NULL AS DOUBLE)"
        if use_hpi:
            hpi_cte = f""",
    hpi_state AS (
        SELECT
            state,
            quarter_key,
            hpi_index,
            hpi_yoy
        FROM read_parquet('{hpi_glob}')
    )"""
            hpi_curr_join = """
        LEFT JOIN hpi_state hc
          ON hc.state = s2.STATE
         AND hc.quarter_key = (
            CAST(SUBSTR(s2.ACT_PERIOD, 3, 4) AS INTEGER) * 10
            + CAST(CEIL(CAST(SUBSTR(s2.ACT_PERIOD, 1, 2) AS DOUBLE) / 3.0) AS INTEGER)
         )
"""
            hpi_orig_join = """
        LEFT JOIN hpi_state ho
          ON ho.state = s2.STATE
         AND ho.quarter_key = (
            CAST(SUBSTR(s2.ORIG_DATE, 3, 4) AS INTEGER) * 10
            + CAST(CEIL(CAST(SUBSTR(s2.ORIG_DATE, 1, 2) AS DOUBLE) / 3.0) AS INTEGER)
         )
"""
        hpi_curr_expr = "MAX_BY(hc.hpi_index, s2.loan_age_eff)" if use_hpi else "CAST(NULL AS DOUBLE)"
        hpi_yoy_expr = "MAX_BY(hc.hpi_yoy, s2.loan_age_eff)" if use_hpi else "CAST(NULL AS DOUBLE)"
        hpi_orig_expr = "MAX_BY(ho.hpi_index, s2.loan_age_eff)" if use_hpi else "CAST(NULL AS DOUBLE)"
        macro_ctes = f"""
    {rates_cte}
    {hpi_cte},
    macro AS (
        SELECT
            s2.LOAN_ID,
            {rate_m30_expr} AS macro_mortgage30us,
            {rate_unrate_expr} AS macro_unrate,
            {rate_dgs10_expr} AS macro_dgs10,
            {rate_term_expr} AS macro_term_spread_10y3m,
            {hpi_curr_expr} AS hpi_state_curr,
            {hpi_yoy_expr} AS hpi_state_growth_1y,
            {hpi_orig_expr} AS hpi_state_orig
        FROM src2 s2
        {rates_join}
        {hpi_curr_join}
        {hpi_orig_join}
        WHERE s2.loan_age_eff <= {FEATURE_WINDOW_MAX}
        GROUP BY s2.LOAN_ID
    )
"""
        macro_join = "LEFT JOIN macro m USING (LOAN_ID)"
        macro_select = """
        m.macro_mortgage30us,
        m.macro_unrate,
        m.macro_dgs10,
        m.macro_term_spread_10y3m,
        m.hpi_state_curr,
        m.hpi_state_orig,
        m.hpi_state_growth_1y,
        CASE
            WHEN m.hpi_state_orig > 0
            THEN m.hpi_state_curr / m.hpi_state_orig
            ELSE NULL
        END AS hpi_state_ratio_curr_to_orig,
        CASE
            WHEN m.macro_mortgage30us IS NOT NULL
            THEN o.ORIG_RATE - m.macro_mortgage30us
            ELSE NULL
        END AS rate_incentive,
        CASE
            WHEN o.OLTV > 0 AND m.hpi_state_orig > 0 AND m.hpi_state_curr > 0
            THEN o.OLTV / (m.hpi_state_curr / m.hpi_state_orig)
            ELSE NULL
        END AS est_current_ltv_from_oltv
"""

    return f"""
    WITH src AS (
        SELECT
            LOAN_ID,
            ACT_PERIOD,
            CURRENT_UPB,
            DLQ_STATUS,
            ZB_CODE,
            MOD_FLAG,
            -- origination features repeated every row; pick any row
            ORIG_RATE, ORIG_UPB, ORIG_TERM,
            OLTV, OCLTV, DTI, NUM_BO,
            CSCORE_B, CSCORE_C,
            PURPOSE, PROP, OCC_STAT, FIRST_FLAG, CHANNEL,
            STATE, MSA,
            SELLER, SERVICER,
            ORIG_DATE, FIRST_PAY,
            TRY_CAST(DLQ_STATUS AS INTEGER) AS dlq_num,
            COALESCE(
                LOAN_AGE,
                (CAST(SUBSTR(ACT_PERIOD, 3, 4) AS INTEGER) * 12
                 + CAST(SUBSTR(ACT_PERIOD, 1, 2) AS INTEGER))
              - (CAST(SUBSTR(ORIG_DATE, 3, 4) AS INTEGER) * 12
                 + CAST(SUBSTR(ORIG_DATE, 1, 2) AS INTEGER))
            ) AS loan_age_eff
        FROM read_parquet('{raw_glob}', hive_partitioning=1)
    ),
    src2 AS (
        SELECT * FROM src
        WHERE loan_age_eff IS NOT NULL
          AND loan_age_eff >= 0   -- drop pre-origination 'LOAN_AGE = -1' rows
    ),

    -- static origination snapshot per loan (take any row)
    orig AS (
        SELECT
            LOAN_ID,
            ANY_VALUE(ORIG_RATE) AS ORIG_RATE,
            ANY_VALUE(ORIG_UPB)  AS ORIG_UPB,
            ANY_VALUE(ORIG_TERM) AS ORIG_TERM,
            ANY_VALUE(OLTV)      AS OLTV,
            ANY_VALUE(OCLTV)     AS OCLTV,
            ANY_VALUE(DTI)       AS DTI,
            ANY_VALUE(NUM_BO)    AS NUM_BO,
            ANY_VALUE(CSCORE_B)  AS CSCORE_B,
            ANY_VALUE(CSCORE_C)  AS CSCORE_C,
            ANY_VALUE(PURPOSE)   AS PURPOSE,
            ANY_VALUE(PROP)      AS PROP,
            ANY_VALUE(OCC_STAT)  AS OCC_STAT,
            ANY_VALUE(FIRST_FLAG) AS FIRST_FLAG,
            ANY_VALUE(CHANNEL)   AS CHANNEL,
            ANY_VALUE(STATE)     AS STATE,
            ANY_VALUE(MSA)       AS MSA,
            ANY_VALUE(SELLER)    AS SELLER,
            ANY_VALUE(SERVICER)  AS SERVICER,
            ANY_VALUE(ORIG_DATE) AS ORIG_DATE,
            ANY_VALUE(FIRST_PAY) AS FIRST_PAY
        FROM src2
        GROUP BY LOAN_ID
    ),

    -- feature window aggregates (0..12)
    feat AS (
        SELECT
            LOAN_ID,
            COUNT(*)                                           AS f12_n_obs,
            MAX(dlq_num)                                       AS f12_dlq_max,
            COUNT(*) FILTER (WHERE dlq_num >= 1)               AS f12_months_30p,
            COUNT(*) FILTER (WHERE dlq_num >= 2)               AS f12_months_60p,
            COUNT(*) FILTER (WHERE dlq_num >= 3)               AS f12_months_90p,
            MAX(CASE WHEN dlq_num >= 1 THEN 1 ELSE 0 END)      AS f12_ever_30,
            MAX(CASE WHEN dlq_num >= 2 THEN 1 ELSE 0 END)      AS f12_ever_60,
            MAX(CASE WHEN dlq_num >= 3 THEN 1 ELSE 0 END)      AS f12_ever_90,
            MAX(CASE WHEN MOD_FLAG = 'Y' THEN 1 ELSE 0 END)    AS f12_mod_flag,
            MAX(CASE WHEN DLQ_STATUS = 'XX' THEN 1 ELSE 0 END) AS f12_forbearance,
            arg_max(CURRENT_UPB, loan_age_eff)                 AS f12_last_upb,
            MAX(loan_age_eff)                                  AS f12_last_age_seen
        FROM src2
        WHERE loan_age_eff <= {FEATURE_WINDOW_MAX}
        GROUP BY LOAN_ID
    ),

    -- label window aggregates (13..24)
    lab AS (
        SELECT
            LOAN_ID,
            COUNT(*)                                           AS lw_n_obs,
            MAX(CASE WHEN dlq_num >= 3 THEN 1 ELSE 0 END)      AS y_delinq_90p,
            MAX(CASE WHEN ZB_CODE = '01' THEN 1 ELSE 0 END)    AS y_prepay,
            MAX(CASE WHEN ZB_CODE IN ('02','03','09','15')
                     THEN 1 ELSE 0 END)                        AS y_default_zb,
            MAX(CASE WHEN DLQ_STATUS = 'XX' THEN 1 ELSE 0 END) AS lw_forbearance
        FROM src2
        WHERE loan_age_eff BETWEEN {LABEL_WINDOW_MIN} AND {LABEL_WINDOW_MAX}
        GROUP BY LOAN_ID
    ),

    -- per-loan horizon: latest age observed and whether any terminal event
    horizon AS (
        SELECT
            LOAN_ID,
            MAX(loan_age_eff)                                      AS max_age,
            MAX(CASE WHEN ZB_CODE IS NOT NULL THEN 1 ELSE 0 END)   AS ever_terminal,
            MIN(CASE WHEN ZB_CODE IS NOT NULL THEN loan_age_eff END) AS first_terminal_age
        FROM src2
        GROUP BY LOAN_ID
    ){macro_ctes}

    SELECT
        o.LOAN_ID,
        -- origination features
        o.ORIG_RATE, o.ORIG_UPB, o.ORIG_TERM,
        o.OLTV, o.OCLTV, o.DTI, o.NUM_BO,
        o.CSCORE_B, o.CSCORE_C,
        o.PURPOSE, o.PROP, o.OCC_STAT, o.FIRST_FLAG, o.CHANNEL,
        o.STATE, o.MSA, o.SELLER, o.SERVICER,
        o.ORIG_DATE, o.FIRST_PAY,
        -- feature window
        COALESCE(f.f12_n_obs, 0)            AS f12_n_obs,
        f.f12_dlq_max,
        COALESCE(f.f12_months_30p, 0)       AS f12_months_30p,
        COALESCE(f.f12_months_60p, 0)       AS f12_months_60p,
        COALESCE(f.f12_months_90p, 0)       AS f12_months_90p,
        COALESCE(f.f12_ever_30, 0)          AS f12_ever_30,
        COALESCE(f.f12_ever_60, 0)          AS f12_ever_60,
        COALESCE(f.f12_ever_90, 0)          AS f12_ever_90,
        COALESCE(f.f12_mod_flag, 0)         AS f12_mod_flag,
        COALESCE(f.f12_forbearance, 0)      AS f12_forbearance,
        f.f12_last_upb,
        CASE
            WHEN o.ORIG_UPB > 0 AND f.f12_last_upb IS NOT NULL
            THEN (o.ORIG_UPB - f.f12_last_upb) / o.ORIG_UPB
            ELSE NULL
        END                                  AS f12_upb_paid_ratio,
        f.f12_last_age_seen,
        {macro_select},
        -- horizon
        h.max_age                            AS observed_through_age,
        h.ever_terminal,
        h.first_terminal_age,
        -- labels (only valid if eligible)
        COALESCE(l.y_delinq_90p, 0)          AS y_delinq_90p,
        COALESCE(l.y_prepay, 0)              AS y_prepay,
        COALESCE(l.y_default_zb, 0)          AS y_default_zb,
        COALESCE(l.lw_forbearance, 0)        AS lw_forbearance,
        COALESCE(l.lw_n_obs, 0)              AS lw_n_obs,
        -- eligibility: observed through the end of the label window, or
        -- terminated at any point (then the label is unambiguously resolved).
        CASE
            WHEN h.max_age >= {LABEL_WINDOW_MAX}
              OR h.ever_terminal = 1
            THEN 1 ELSE 0
        END                                  AS eligible
    FROM orig o
    LEFT JOIN feat f    USING (LOAN_ID)
    LEFT JOIN lab  l    USING (LOAN_ID)
    LEFT JOIN horizon h USING (LOAN_ID)
    {macro_join}
    """


def build_one(
    vintage: str,
    raw_root: Path = RAW_ROOT_DEFAULT,
    feat_root: Path = FEAT_ROOT_DEFAULT,
    macro_root: Path = MACRO_ROOT_DEFAULT,
    tmp_root: Path = TMP_DEFAULT,
    threads: int = 16,
    memory_limit: str = "60GB",
    compression: str = "snappy",
    overwrite: bool = False,
) -> dict:
    year = int(vintage[:4])
    qtr = vintage[4:]
    raw_glob = str(raw_root / f"orig_year={year}" / f"orig_qtr={qtr}" / "part-*.parquet")
    dst_dir = feat_root / f"vintage={vintage}"
    dst = dst_dir / "part-0.parquet"
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists (use --overwrite)")

    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{tmp_root}'")

    rates_path = macro_root / "fred_monthly.parquet"
    hpi_path = macro_root / "hpi_state_quarterly.parquet"
    use_rates = rates_path.exists()
    use_hpi = hpi_path.exists()

    sql = f"""
        COPY (
            {build_features_sql(
                raw_glob,
                str(rates_path) if use_rates else None,
                str(hpi_path) if use_hpi else None,
            )}
        ) TO '{dst}'
        (FORMAT PARQUET, COMPRESSION {compression.upper()}, ROW_GROUP_SIZE 100000);
    """
    t0 = time.time()
    con.execute(sql)
    elapsed = time.time() - t0

    n_loans, n_eligible, n_prepay, n_delinq = con.execute(f"""
        SELECT
            COUNT(*),
            SUM(eligible),
            SUM(CASE WHEN eligible = 1 THEN y_prepay END),
            SUM(CASE WHEN eligible = 1 THEN y_delinq_90p END)
        FROM read_parquet('{dst}')
    """).fetchone()
    return {
        "vintage": vintage,
        "loans": n_loans,
        "eligible": n_eligible,
        "prepay_rate": (n_prepay / n_eligible) if n_eligible else None,
        "delinq_rate": (n_delinq / n_eligible) if n_eligible else None,
        "elapsed_sec": round(elapsed, 1),
        "out_path": str(dst),
        "size_mb": round(dst.stat().st_size / 1024 / 1024, 1),
        "macro_used": use_rates or use_hpi,
        "macro_rates_used": use_rates,
        "macro_hpi_used": use_hpi,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vintage", help="e.g. 2018Q1")
    ap.add_argument("--raw-root", default=str(RAW_ROOT_DEFAULT))
    ap.add_argument("--feat-root", default=str(FEAT_ROOT_DEFAULT))
    ap.add_argument("--macro-root", default=str(MACRO_ROOT_DEFAULT))
    ap.add_argument("--tmp-root", default=str(TMP_DEFAULT))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--memory-limit", default="60GB")
    ap.add_argument("--compression", default="snappy")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    info = build_one(
        vintage=a.vintage,
        raw_root=Path(a.raw_root),
        feat_root=Path(a.feat_root),
        macro_root=Path(a.macro_root),
        tmp_root=Path(a.tmp_root),
        threads=a.threads,
        memory_limit=a.memory_limit,
        compression=a.compression,
        overwrite=a.overwrite,
    )
    pr = f"{info['prepay_rate']:.4f}" if info['prepay_rate'] is not None else "n/a"
    dr = f"{info['delinq_rate']:.4f}" if info['delinq_rate'] is not None else "n/a"
    print(
        f"[features] vintage={info['vintage']} loans={info['loans']:,} "
        f"eligible={info['eligible']:,} prepay={pr} delinq90p={dr} "
        f"macro={info['macro_used']} rates={info['macro_rates_used']} hpi={info['macro_hpi_used']} "
        f"size={info['size_mb']} MB elapsed={info['elapsed_sec']}s "
        f"-> {info['out_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
