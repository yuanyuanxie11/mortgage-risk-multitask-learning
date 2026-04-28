"""Train PyTorch multi-task baseline for delinquency + prepayment.

Usage:
  python -m src.train_mtl
  python -m src.train_mtl --epochs 30 --batch-size 4096
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

try:
    from .config import ALL_FEATURES, CAT_FEATURES, FEAT_ROOT, METRICS_DIR, MODELS_DIR
    from .data import SPLIT_BANDS
    from .model_mtl import MortgageMTL
except ImportError:
    from config import ALL_FEATURES, CAT_FEATURES, FEAT_ROOT, METRICS_DIR, MODELS_DIR
    from data import SPLIT_BANDS
    from model_mtl import MortgageMTL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_split_raw(split: str) -> pd.DataFrame:
    lo, hi = SPLIT_BANDS[split]
    glob = str(FEAT_ROOT / "vintage=*" / "part-*.parquet")
    con = duckdb.connect()
    sql = f"""
        SELECT {", ".join(ALL_FEATURES)},
               y_delinq_90p, y_prepay
        FROM read_parquet('{glob}', hive_partitioning=1)
        WHERE eligible = 1
          AND CAST(SUBSTR(vintage, 1, 4) AS INTEGER) BETWEEN {lo} AND {hi}
    """
    df = con.execute(sql).df()
    con.close()
    if df.empty:
        raise RuntimeError(f"No rows for split={split}. Build features first.")
    return df


def _fit_preprocessor(df_train: pd.DataFrame) -> dict:
    num_cols = [c for c in ALL_FEATURES if c not in CAT_FEATURES]
    cat_cols = [c for c in CAT_FEATURES if c in df_train.columns]

    medians = {c: float(df_train[c].median()) for c in num_cols}
    for c in num_cols:
        df_train[c] = df_train[c].fillna(medians[c])
    for c in cat_cols:
        df_train[c] = df_train[c].astype("object").fillna("__MISSING__")

    x_train = pd.get_dummies(df_train[ALL_FEATURES], columns=cat_cols, dummy_na=False)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
    return {
        "medians": medians,
        "cat_cols": cat_cols,
        "feature_columns": list(x_train.columns),
        "scaler": scaler,
        "x_train_scaled": x_train_scaled,
    }


def _transform(df: pd.DataFrame, prep: dict) -> np.ndarray:
    num_cols = [c for c in ALL_FEATURES if c not in prep["cat_cols"]]
    for c in num_cols:
        df[c] = df[c].fillna(prep["medians"][c])
    for c in prep["cat_cols"]:
        df[c] = df[c].astype("object").fillna("__MISSING__")

    x = pd.get_dummies(df[ALL_FEATURES], columns=prep["cat_cols"], dummy_na=False)
    x = x.reindex(columns=prep["feature_columns"], fill_value=0.0)
    return prep["scaler"].transform(x.astype(np.float32))


def _to_loader(x: np.ndarray, y_d: np.ndarray, y_p: np.ndarray, batch_size: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y_d).float(),
        torch.from_numpy(y_p).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _evaluate(model: Any, loader: Any, device: Any) -> dict:
    import torch
    model.eval()
    d_logits, p_logits, d_true, p_true = [], [], [], []
    with torch.no_grad():
        for xb, yd, yp in loader:
            xb = xb.to(device)
            ld, lp = model(xb)
            d_logits.append(ld.cpu().numpy())
            p_logits.append(lp.cpu().numpy())
            d_true.append(yd.numpy())
            p_true.append(yp.numpy())
    d_prob = 1.0 / (1.0 + np.exp(-np.concatenate(d_logits)))
    p_prob = 1.0 / (1.0 + np.exp(-np.concatenate(p_logits)))
    d_true = np.concatenate(d_true)
    p_true = np.concatenate(p_true)
    return {
        "delinq_pr_auc": float(average_precision_score(d_true, d_prob)),
        "prepay_pr_auc": float(average_precision_score(p_true, p_prob)),
        "joint_pr_auc": float((average_precision_score(d_true, d_prob) + average_precision_score(p_true, p_prob)) / 2.0),
    }


def main() -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError(
            "PyTorch is not usable in this environment. "
            "Install a CPU-compatible torch wheel or fix CUDA/library mismatch."
        ) from e
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lambda-delinq", type=float, default=1.0)
    ap.add_argument("--lambda-prepay", type=float, default=1.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cuda" if args.device == "cuda" else "cpu"))
    log.info("device=%s", device)

    df_train = _load_split_raw("train")
    df_val = _load_split_raw("val")

    prep = _fit_preprocessor(df_train.copy())
    x_train = prep["x_train_scaled"]
    y_train_d = df_train["y_delinq_90p"].to_numpy(dtype=np.float32)
    y_train_p = df_train["y_prepay"].to_numpy(dtype=np.float32)

    x_val = _transform(df_val.copy(), prep)
    y_val_d = df_val["y_delinq_90p"].to_numpy(dtype=np.float32)
    y_val_p = df_val["y_prepay"].to_numpy(dtype=np.float32)

    train_loader = _to_loader(x_train, y_train_d, y_train_p, args.batch_size, True)
    val_loader = _to_loader(x_val, y_val_d, y_val_p, args.batch_size, False)

    model = MortgageMTL(input_dim=x_train.shape[1]).to(device)
    pos_w_d = torch.tensor([(len(y_train_d) - y_train_d.sum()) / max(y_train_d.sum(), 1.0)], dtype=torch.float32, device=device)
    pos_w_p = torch.tensor([(len(y_train_p) - y_train_p.sum()) / max(y_train_p.sum(), 1.0)], dtype=torch.float32, device=device)
    loss_d = nn.BCEWithLogitsLoss(pos_weight=pos_w_d)
    loss_p = nn.BCEWithLogitsLoss(pos_weight=pos_w_p)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = {"joint": -1.0, "epoch": -1}
    history = []
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yd, yp in train_loader:
            xb, yd, yp = xb.to(device), yd.to(device), yp.to(device)
            opt.zero_grad(set_to_none=True)
            ld, lp = model(xb)
            l = args.lambda_delinq * loss_d(ld, yd) + args.lambda_prepay * loss_p(lp, yp)
            l.backward()
            opt.step()
            running += float(l.item()) * len(xb)
            n += len(xb)

        train_loss = running / max(n, 1)
        val_metrics = _evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(row)
        log.info(
            "epoch=%d loss=%.5f val_joint=%.5f val_delinq=%.5f val_prepay=%.5f",
            epoch, train_loss, val_metrics["joint_pr_auc"], val_metrics["delinq_pr_auc"], val_metrics["prepay_pr_auc"],
        )

        if val_metrics["joint_pr_auc"] > best["joint"] + 1e-6:
            best = {"joint": val_metrics["joint_pr_auc"], "epoch": epoch}
            stale = 0
            ckpt = {
                "model_state": model.state_dict(),
                "input_dim": x_train.shape[1],
                "best_epoch": epoch,
                "best_joint_pr_auc": best["joint"],
                "feature_columns": prep["feature_columns"],
                "seed": args.seed,
            }
            torch.save(ckpt, MODELS_DIR / "mtl_best.pt")
            with open(MODELS_DIR / "mtl_preprocessor.pkl", "wb") as f:
                pickle.dump(
                    {
                        "medians": prep["medians"],
                        "cat_cols": prep["cat_cols"],
                        "feature_columns": prep["feature_columns"],
                        "scaler": prep["scaler"],
                    },
                    f,
                )
        else:
            stale += 1
            if stale >= args.patience:
                log.info("early stop at epoch=%d (best epoch=%d)", epoch, best["epoch"])
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(METRICS_DIR / "mtl_training_history.csv", index=False)
    with open(METRICS_DIR / "mtl_training_summary.json", "w") as f:
        json.dump(
            {
                "best_epoch": best["epoch"],
                "best_joint_pr_auc": best["joint"],
                "epochs_ran": int(hist_df["epoch"].max()) if len(hist_df) else 0,
                "seed": args.seed,
                "device": str(device),
            },
            f,
            indent=2,
        )
    print(f"Saved best model -> {MODELS_DIR / 'mtl_best.pt'}")
    print(f"Saved history    -> {METRICS_DIR / 'mtl_training_history.csv'}")


if __name__ == "__main__":
    main()

