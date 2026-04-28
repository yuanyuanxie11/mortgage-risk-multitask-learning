"""Streaming zip -> Parquet converter for Fannie Mae SF Loan Performance data.

Reads one member CSV out of `Performance_All.zip` via `unzip -p` (no scratch
file), applies the fixed 110-column schema, and writes a single Parquet part
under a Hive-partitioned
directory layout:

    data/raw_parquet/orig_year={yyyy}/orig_qtr={Qn}/part-0.parquet

Usage:
    python -m src.ingest 2018Q1
    python -m src.ingest 2018Q1 --compression snappy --row-group 100000

Design notes:
- We stream via `unzip -p <zip> <member>` piped into DuckDB's CSV reader at
  `/dev/fd/<N>`. The subprocess stdout fd is passed inheritably so DuckDB
  can open it as a regular forward-only file.
- DuckDB's temp directory is pinned to NFS (`data/duckdb_tmp/`) because the
  host's root filesystem has only ~60 GB free; spill for the ORDER BY on
  large vintages must not land there.
- The per-vintage partition value (`orig_year`, `orig_qtr`) is *not* stored
  in the file; Hive partitioning derives it from the directory name when
  queried with `hive_partitioning=1`.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb

from .schema import SCHEMA, duckdb_columns_clause

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH_DEFAULT = PROJECT_ROOT / "Performance_All.zip"
OUT_ROOT_DEFAULT = PROJECT_ROOT / "data" / "raw_parquet"
TMP_ROOT_DEFAULT = PROJECT_ROOT / "data" / "duckdb_tmp"

VINTAGE_RE = re.compile(r"^(\d{4})(Q[1-4])$")


def parse_vintage(vintage: str) -> tuple[int, str]:
    m = VINTAGE_RE.match(vintage)
    if not m:
        raise ValueError(f"vintage must look like 2018Q1, got {vintage!r}")
    return int(m.group(1)), m.group(2)


def target_path(out_root: Path, year: int, qtr: str) -> Path:
    return out_root / f"orig_year={year}" / f"orig_qtr={qtr}" / "part-0.parquet"


def convert_one(
    vintage: str,
    zip_path: Path = ZIP_PATH_DEFAULT,
    out_root: Path = OUT_ROOT_DEFAULT,
    tmp_root: Path = TMP_ROOT_DEFAULT,
    compression: str = "snappy",
    row_group_size: int = 100_000,
    threads: int = 4,
    memory_limit: str = "40GB",
    overwrite: bool = False,
) -> dict:
    """Convert a single vintage CSV member of the zip into a Parquet file.

    Returns a dict with row count, elapsed seconds, and output path.
    """
    year, qtr = parse_vintage(vintage)
    member = f"{vintage}.csv"
    dst = target_path(out_root, year, qtr)

    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists (use --overwrite)")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(".parquet.tmp")

    # spawn unzip streaming stdout
    unzip = subprocess.Popen(
        ["unzip", "-p", str(zip_path), member],
        stdout=subprocess.PIPE,
        bufsize=0,
    )
    assert unzip.stdout is not None
    fd = unzip.stdout.fileno()
    os.set_inheritable(fd, True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{tmp_root}'")

    columns_clause = duckdb_columns_clause()
    src = f"/dev/fd/{fd}"

    sql = f"""
        COPY (
            SELECT *
            FROM read_csv(
                '{src}',
                delim='|',
                header=false,
                columns={columns_clause},
                nullstr='',
                ignore_errors=false,
                parallel=true
            )
        )
        TO '{tmp_path}'
        (FORMAT PARQUET,
         COMPRESSION {compression.upper()},
         ROW_GROUP_SIZE {row_group_size});
    """

    t0 = time.time()
    try:
        con.execute(sql)
    finally:
        unzip.stdout.close()
        rc = unzip.wait()
        if rc != 0:
            # clean temp and surface error
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"unzip exited with code {rc} for member {member}")
    elapsed = time.time() - t0

    rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{tmp_path}')"
    ).fetchone()[0]
    size_mb = tmp_path.stat().st_size / 1024 / 1024

    # atomic move into place
    shutil.move(str(tmp_path), str(dst))

    return {
        "vintage": vintage,
        "rows": rows,
        "elapsed_sec": round(elapsed, 1),
        "out_path": str(dst),
        "size_mb": round(size_mb, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vintage", help="e.g. 2018Q1")
    ap.add_argument("--zip", default=str(ZIP_PATH_DEFAULT))
    ap.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--tmp-root", default=str(TMP_ROOT_DEFAULT))
    ap.add_argument("--compression", default="snappy", choices=["snappy", "zstd", "gzip"])
    ap.add_argument("--row-group", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory-limit", default="40GB")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    info = convert_one(
        vintage=args.vintage,
        zip_path=Path(args.zip),
        out_root=Path(args.out_root),
        tmp_root=Path(args.tmp_root),
        compression=args.compression,
        row_group_size=args.row_group,
        threads=args.threads,
        memory_limit=args.memory_limit,
        overwrite=args.overwrite,
    )
    print(
        f"[ingest] vintage={info['vintage']} rows={info['rows']:,} "
        f"size={info['size_mb']} MB elapsed={info['elapsed_sec']}s "
        f"-> {info['out_path']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
