"""Fannie Mae Single-Family Loan Performance Data schema.

The modern (post-May 2023) combined Acquisition + Performance layout has 110
pipe-delimited columns, no header row. Origination fields are repeated on every
monthly observation row.

Column 1 (POOL_ID) is frequently empty for conventional SF data.
LOAN_ID (col 2) + ACT_PERIOD (col 3) form the natural primary key.

Dates are stored as MMYYYY strings in the source file; we keep them as VARCHAR
and parse to real DATE downstream. This avoids lossy conversion during ingest
and matches the official Fannie Mae layout document.
"""

from __future__ import annotations

SCHEMA: list[tuple[str, str]] = [
    ("POOL_ID", "VARCHAR"),
    ("LOAN_ID", "VARCHAR"),
    ("ACT_PERIOD", "VARCHAR"),
    ("CHANNEL", "VARCHAR"),
    ("SELLER", "VARCHAR"),
    ("SERVICER", "VARCHAR"),
    ("MASTER_SERVICER", "VARCHAR"),
    ("ORIG_RATE", "DOUBLE"),
    ("CURR_RATE", "DOUBLE"),
    ("ORIG_UPB", "DOUBLE"),
    ("ISSUANCE_UPB", "DOUBLE"),
    ("CURRENT_UPB", "DOUBLE"),
    ("ORIG_TERM", "INTEGER"),
    ("ORIG_DATE", "VARCHAR"),
    ("FIRST_PAY", "VARCHAR"),
    ("LOAN_AGE", "INTEGER"),
    ("REM_MONTHS", "INTEGER"),
    ("ADJ_REM_MONTHS", "INTEGER"),
    ("MATR_DT", "VARCHAR"),
    ("OLTV", "INTEGER"),
    ("OCLTV", "INTEGER"),
    ("NUM_BO", "INTEGER"),
    ("DTI", "INTEGER"),
    ("CSCORE_B", "INTEGER"),
    ("CSCORE_C", "INTEGER"),
    ("FIRST_FLAG", "VARCHAR"),
    ("PURPOSE", "VARCHAR"),
    ("PROP", "VARCHAR"),
    ("NO_UNITS", "INTEGER"),
    ("OCC_STAT", "VARCHAR"),
    ("STATE", "VARCHAR"),
    ("MSA", "VARCHAR"),
    ("ZIP", "VARCHAR"),
    ("MI_PCT", "DOUBLE"),
    ("PRODUCT", "VARCHAR"),
    ("PPMT_FLG", "VARCHAR"),
    ("IO", "VARCHAR"),
    ("FIRST_PAY_IO", "VARCHAR"),
    ("MNTHS_TO_AMTZ_IO", "INTEGER"),
    ("DLQ_STATUS", "VARCHAR"),
    ("PMT_HISTORY", "VARCHAR"),
    ("MOD_FLAG", "VARCHAR"),
    ("MI_CANCEL_FLAG", "VARCHAR"),
    ("ZB_CODE", "VARCHAR"),
    ("ZB_DTE", "VARCHAR"),
    ("LAST_UPB", "DOUBLE"),
    ("RPRCH_DTE", "VARCHAR"),
    ("CURR_SCHD_PRNCPL", "DOUBLE"),
    ("TOT_SCHD_PRNCPL", "DOUBLE"),
    ("UNSCHD_PRNCPL_CURR", "DOUBLE"),
    ("LAST_PAID_INSTALLMENT_DATE", "VARCHAR"),
    ("FORECLOSURE_DATE", "VARCHAR"),
    ("DISPOSITION_DATE", "VARCHAR"),
    ("FORECLOSURE_COSTS", "DOUBLE"),
    ("PROPERTY_PRESERVATION_AND_REPAIR_COSTS", "DOUBLE"),
    ("ASSET_RECOVERY_COSTS", "DOUBLE"),
    ("MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS", "DOUBLE"),
    ("ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", "DOUBLE"),
    ("NET_SALES_PROCEEDS", "DOUBLE"),
    ("CREDIT_ENHANCEMENT_PROCEEDS", "DOUBLE"),
    ("REPURCHASES_MAKE_WHOLE_PROCEEDS", "DOUBLE"),
    ("OTHER_FORECLOSURE_PROCEEDS", "DOUBLE"),
    ("NON_INTEREST_BEARING_UPB", "DOUBLE"),
    ("PRINCIPAL_FORGIVENESS_AMOUNT", "DOUBLE"),
    ("ORIGINAL_LIST_START_DATE", "VARCHAR"),
    ("ORIGINAL_LIST_PRICE", "DOUBLE"),
    ("CURRENT_LIST_START_DATE", "VARCHAR"),
    ("CURRENT_LIST_PRICE", "DOUBLE"),
    ("ISSUE_SCOREB", "INTEGER"),
    ("ISSUE_SCOREC", "INTEGER"),
    ("CURR_SCOREB", "INTEGER"),
    ("CURR_SCOREC", "INTEGER"),
    ("MI_TYPE", "VARCHAR"),
    ("SERV_IND", "VARCHAR"),
    ("CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT", "DOUBLE"),
    ("CUMULATIVE_MODIFICATION_LOSS_AMOUNT", "DOUBLE"),
    ("CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS", "DOUBLE"),
    ("CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS", "DOUBLE"),
    ("HOMEREADY_PROGRAM_INDICATOR", "VARCHAR"),
    ("FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT", "DOUBLE"),
    ("RELOCATION_MORTGAGE_INDICATOR", "VARCHAR"),
    ("ZERO_BALANCE_CODE_CHANGE_DATE", "VARCHAR"),
    ("LOAN_HOLDBACK_INDICATOR", "VARCHAR"),
    ("LOAN_HOLDBACK_EFFECTIVE_DATE", "VARCHAR"),
    ("DELINQUENT_ACCRUED_INTEREST", "DOUBLE"),
    ("PROPERTY_INSPECTION_WAIVER_INDICATOR", "VARCHAR"),
    ("HIGH_BALANCE_LOAN_INDICATOR", "VARCHAR"),
    ("ARM_5_YR_INDICATOR", "VARCHAR"),
    ("ARM_PRODUCT_TYPE", "VARCHAR"),
    ("MONTHS_UNTIL_FIRST_PAYMENT_RESET", "INTEGER"),
    ("MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET", "INTEGER"),
    ("INTEREST_RATE_CHANGE_DATE", "VARCHAR"),
    ("PAYMENT_CHANGE_DATE", "VARCHAR"),
    ("ARM_INDEX", "VARCHAR"),
    ("ARM_CAP_STRUCTURE", "VARCHAR"),
    ("INITIAL_INTEREST_RATE_CAP", "DOUBLE"),
    ("PERIODIC_INTEREST_RATE_CAP", "DOUBLE"),
    ("LIFETIME_INTEREST_RATE_CAP", "DOUBLE"),
    ("MARGIN", "DOUBLE"),
    ("BALLOON_INDICATOR", "VARCHAR"),
    ("PLAN_NUMBER", "VARCHAR"),
    ("FORBEARANCE_INDICATOR", "VARCHAR"),
    ("HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR", "VARCHAR"),
    ("DEAL_NAME", "VARCHAR"),
    ("RE_PROCS_FLAG", "VARCHAR"),
    ("ADR_TYPE", "VARCHAR"),
    ("ADR_COUNT", "INTEGER"),
    ("ADR_UPB", "DOUBLE"),
    ("PAYMENT_DEFERRAL_MOD_EVENT_FLAG", "VARCHAR"),
    ("REPAYMENT_PLAN_INDICATOR", "VARCHAR"),
]

assert len(SCHEMA) == 110, f"schema must have 110 columns, got {len(SCHEMA)}"
assert len({n for n, _ in SCHEMA}) == 110, "duplicate column names"


def duckdb_columns_clause() -> str:
    """Render the schema as a DuckDB `columns={...}` STRUCT literal.

    DuckDB's read_csv accepts an ordered STRUCT of name -> type; it applies the
    types positionally when header=false.
    """
    parts = [f"'{name}': '{dtype}'" for name, dtype in SCHEMA]
    return "{" + ", ".join(parts) + "}"


def column_names() -> list[str]:
    return [name for name, _ in SCHEMA]


CORE_FEATURE_COLS: list[str] = [
    # identifiers / time
    "LOAN_ID", "ACT_PERIOD", "ORIG_DATE", "FIRST_PAY", "LOAN_AGE", "REM_MONTHS",
    # origination features
    "ORIG_RATE", "ORIG_UPB", "ORIG_TERM", "OLTV", "OCLTV", "DTI",
    "CSCORE_B", "CSCORE_C", "PURPOSE", "PROP", "OCC_STAT",
    "STATE", "MSA", "NUM_BO", "FIRST_FLAG", "CHANNEL", "SELLER", "SERVICER",
    # monthly performance
    "CURRENT_UPB", "DLQ_STATUS", "MOD_FLAG",
    "ZB_CODE", "ZB_DTE",
    "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE",
    # loss fields for Project 2
    "NET_SALES_PROCEEDS", "NON_INTEREST_BEARING_UPB",
    "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT",
    "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
    "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS",
]


if __name__ == "__main__":
    import sys
    print(f"schema rows: {len(SCHEMA)}", file=sys.stderr)
    for i, (n, t) in enumerate(SCHEMA, 1):
        print(f"{i:3d}  {n:55s}  {t}")
