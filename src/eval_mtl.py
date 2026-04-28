"""Evaluate trained MTL model on val/test/stress splits.

Usage:
  python -m src.eval_mtl
  python -m src.eval_mtl --split test
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

try:
    from .config import ALL_FEATURES, FEAT_ROOT, METRICS_DIR, MODELS_DIR
    from .data import SPLIT_BANDS
    from .eval import recall_at_k
    from .model_mtl import MortgageMTL
except ImportError:
    from config import ALL_FEATURES, FEAT_ROOT, METRICS_DIR, MODELS_DIR
    from data import SPLIT_BANDS
    from eval import recall_at_k
    from model_mtl import MortgageMTL


def load_raw_split(split: str) -> pd.DataFrame:
    lo, hi = SPLIT_BANDS[split]
    glob = str(FEAT_ROOT / "vintage=*" / "part-*.parquet")
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT {", ".join(ALL_FEATURES)}, y_delinq_90p, y_prepay
        FROM read_parquet('{glob}', hive_partitioning=1)
        WHERE eligible = 1
          AND CAST(SUBSTR(vintage, 1, 4) AS INTEGER) BETWEEN {lo} AND {hi}
        """
    ).df()
    con.close()
    if df.empty:
        raise RuntimeError(f"No rows in split={split}")
    return df


def transform(df: pd.DataFrame, prep: dict) -> np.ndarray:
    for c, m in prep["medians"].items():
        if c in df.columns:
            df[c] = df[c].fillna(m)
    for c in prep["cat_cols"]:
        df[c] = df[c].astype("object").fillna("__MISSING__")
    x = pd.get_dummies(df[ALL_FEATURES], columns=prep["cat_cols"], dummy_na=False)
    x = x.reindex(columns=prep["feature_columns"], fill_value=0.0)
    return prep["scaler"].transform(x.astype(np.float32))


def evaluate_split(model: MortgageMTL, prep: dict, split: str, device: torch.device) -> list[dict]:
    import torch
    df = load_raw_split(split)
    x = transform(df.copy(), prep)
    xt = torch.from_numpy(x).float().to(device)
    with torch.no_grad():
        ld, lp = model(xt)
    p_delinq = torch.sigmoid(ld).cpu().numpy()
    p_prepay = torch.sigmoid(lp).cpu().numpy()

    rows = []
    for task, y_col, pred in [
        ("delinq", "y_delinq_90p", p_delinq),
        ("prepay", "y_prepay", p_prepay),
    ]:
        y = df[y_col].to_numpy(dtype=np.int32)
        rows.append(
            {
                "model": "mtl",
                "task": task,
                "split": split,
                "n": len(y),
                "prevalence": float(y.mean()),
                "pr_auc": float(average_precision_score(y, pred)),
                "roc_auc": float(roc_auc_score(y, pred)),
                "brier": float(brier_score_loss(y, pred)),
                "recall@5pct": float(recall_at_k(y, pred, 0.05)),
                "recall@10pct": float(recall_at_k(y, pred, 0.10)),
                "recall@20pct": float(recall_at_k(y, pred, 0.20)),
            }
        )
    return rows


def main() -> None:
    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            "PyTorch is not usable in this environment. "
            "Install a CPU-compatible torch wheel or fix CUDA/library mismatch."
        ) from e
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["val", "test", "stress"], help="evaluate one split only")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    ckpt_path = MODELS_DIR / "mtl_best.pt"
    prep_path = MODELS_DIR / "mtl_preprocessor.pkl"
    if not ckpt_path.exists() or not prep_path.exists():
        raise FileNotFoundError("Missing mtl_best.pt or mtl_preprocessor.pkl. Run `python -m src.train_mtl` first.")

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cuda" if args.device == "cuda" else "cpu"))
    ckpt = torch.load(ckpt_path, map_location=device)
    with open(prep_path, "rb") as f:
        prep = pickle.load(f)

    model = MortgageMTL(input_dim=ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    splits = [args.split] if args.split else ["val", "test"]
    rows: list[dict] = []
    for s in splits:
        rows.extend(evaluate_split(model, prep, s, device))

    out = pd.DataFrame(rows)
    out_path = METRICS_DIR / "mtl_eval_table.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

