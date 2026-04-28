"""Download and normalize macro datasets for feature enrichment.

Outputs (Parquet):
  - data/macro/fred_monthly.parquet
  - data/macro/hpi_state_quarterly.parquet

Sources:
  - FRED (mortgage rate, unemployment, treasury rates)
  - FHFA House Price Index (state-level quarterly)

Usage:
  python scripts/download_macro.py
  python scripts/download_macro.py --check
  python scripts/download_macro.py --fred-only
  python scripts/download_macro.py --hpi-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACRO_ROOT = PROJECT_ROOT / "data" / "macro"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def fetch_fred(start: str = "2000-01-01") -> pd.DataFrame:
    series = {
        "mortgage30us": "MORTGAGE30US",
        "unrate": "UNRATE",
        "dgs10": "DGS10",
        "gs3m": "GS3M",
    }
    frames: list[pd.DataFrame] = []
    start_iso = pd.to_datetime(start).strftime("%Y-%m-%d")
    for out_col, fred_id in tqdm(series.items(), desc="FRED series"):
        # public CSV endpoint; no API key required
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={fred_id}&cosd={start_iso}"
        )
        df = pd.read_csv(url)
        df.columns = ["date", out_col]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[out_col] = pd.to_numeric(df[out_col], errors="coerce")
        df = df.dropna(subset=["date"])
        frames.append(df)

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    out = out.sort_values("date")
    out["mortgage30us"] = out["mortgage30us"].ffill()
    out["dgs10"] = out["dgs10"].ffill()
    out["gs3m"] = out["gs3m"].ffill()
    out["period_mmyyyy"] = out["date"].dt.strftime("%m%Y")
    out = out[["date", "period_mmyyyy", "mortgage30us", "unrate", "dgs10", "gs3m"]]
    # monthly collapse: pick last non-null value of each series in month
    out = (
        out.sort_values("date")
        .groupby("period_mmyyyy", as_index=False)
        .agg(
            date=("date", "max"),
            mortgage30us=("mortgage30us", "last"),
            unrate=("unrate", "last"),
            dgs10=("dgs10", "last"),
            gs3m=("gs3m", "last"),
        )
    )
    return out


def _download_to(url: str, path: Path) -> bool:
    try:
        with urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
        return True
    except Exception:
        return False


def fetch_hpi_state() -> pd.DataFrame:
    """Get FHFA state-level quarterly HPI and normalize keys.

    The FHFA URLs have changed over time; we try several known endpoints.
    """
    csv_candidates = [
        # Current FHFA endpoint family (as of 2026)
        "https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_at_state.csv",
        "https://www.fhfa.gov/hpi/download/quarterly_datasets/hpi_po_state.txt",
        # Legacy fallbacks
        "https://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_AT_state.csv",
        "https://www.fhfa.gov/data/hpi/datasets/hpi_po_state.csv",
        "https://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_PO_state.csv",
    ]
    tmp = MACRO_ROOT / "_hpi_state_raw.csv"
    if tmp.exists():
        ok = True
    else:
        ok = False
        for u in csv_candidates:
            log.info("Trying FHFA URL: %s", u)
            if _download_to(u, tmp):
                ok = True
                break
    if not ok:
        return pd.DataFrame(columns=["state", "quarter_key", "hpi_index", "hpi_yoy"])

    df = pd.read_csv(tmp)

    cols = {c.lower(): c for c in df.columns}
    state_col = cols.get("state") or cols.get("st")
    year_col = cols.get("year")
    qtr_col = cols.get("qtr") or cols.get("quarter")
    hpi_col = (
        cols.get("index_nsa")
        or cols.get("index_sa")
        or cols.get("hpi")
        or cols.get("index")
    )
    if not (state_col and year_col and qtr_col and hpi_col):
        # Newer FHFA quarterly datasets can be headerless 4-column files:
        # state, year, quarter, hpi_index
        # Example first row: AK,1975,1,62.03
        df = pd.read_csv(tmp, header=None, names=["state", "year", "qtr", "hpi_index"])
        cols = {c.lower(): c for c in df.columns}
        state_col = cols.get("state")
        year_col = cols.get("year")
        qtr_col = cols.get("qtr")
        hpi_col = cols.get("hpi_index")
        if not (state_col and year_col and qtr_col and hpi_col):
            raise RuntimeError(f"Unrecognized FHFA HPI schema: {list(df.columns)}")
    tmp.unlink(missing_ok=True)

    out = df[[state_col, year_col, qtr_col, hpi_col]].copy()
    out.columns = ["state", "year", "qtr", "hpi_index"]
    out["state"] = out["state"].astype(str).str.upper().str.strip()
    out["year"] = out["year"].astype(int)
    out["qtr"] = (
        out["qtr"].astype(str).str.extract(r"([1-4])")[0].astype(int)
    )
    out["hpi_index"] = pd.to_numeric(out["hpi_index"], errors="coerce")
    out = out.dropna(subset=["hpi_index"])
    out["quarter_key"] = out["year"] * 10 + out["qtr"]
    out = out.sort_values(["state", "quarter_key"])
    out["hpi_yoy"] = out.groupby("state")["hpi_index"].pct_change(4)
    return out[["state", "quarter_key", "hpi_index", "hpi_yoy"]]


def check_files() -> int:
    paths = {
        "FRED monthly": MACRO_ROOT / "fred_monthly.parquet",
        "FHFA HPI state": MACRO_ROOT / "hpi_state_quarterly.parquet",
    }
    ok = True
    print("\nMacro files:")
    for name, p in paths.items():
        if p.exists():
            df = pd.read_parquet(p)
            print(f"  ✓ {name:<16} {len(df):>8,} rows -> {p}")
            print(f"    cols: {', '.join(df.columns[:8])}")
        else:
            print(f"  ✗ {name:<16} missing        -> {p}")
            ok = False
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--check", action="store_true", help="verify files and preview schema")
    ap.add_argument("--fred-only", action="store_true", help="download only FRED series")
    ap.add_argument("--hpi-only", action="store_true", help="download only FHFA HPI")
    ap.add_argument("--skip-msa", action="store_true", help="kept for compatibility; MSA download not implemented")
    args = ap.parse_args()

    MACRO_ROOT.mkdir(parents=True, exist_ok=True)
    if args.check:
        raise SystemExit(check_files())
    if args.fred_only and args.hpi_only:
        raise SystemExit("--fred-only and --hpi-only are mutually exclusive")
    if args.skip_msa:
        log.info("--skip-msa provided (MSA download not implemented in this script)")

    fred_out = MACRO_ROOT / "fred_monthly.parquet"
    hpi_out = MACRO_ROOT / "hpi_state_quarterly.parquet"
    if not args.hpi_only:
        fred = fetch_fred(args.start)
        fred.to_parquet(fred_out, index=False)
        print(f"saved {fred_out} rows={len(fred):,}")
    if not args.fred_only:
        hpi = fetch_hpi_state()
        if len(hpi):
            hpi.to_parquet(hpi_out, index=False)
            print(f"saved {hpi_out} rows={len(hpi):,}")
        else:
            print(
                "HPI download unavailable from known endpoints. "
                "Saved FRED only. To enable HPI features, place FHFA CSV at "
                "data/macro/_hpi_state_raw.csv and rerun."
            )
    print("\nRun `python scripts/download_macro.py --check` to verify outputs.")


if __name__ == "__main__":
    main()
    sys.exit(0)

