"""Evaluate trained XGBoost baselines.

Produces:
  outputs/metrics/baseline_table.csv     ← the anchor benchmark table
  outputs/metrics/scores_{target}_{split}.parquet  ← raw scores for downstream use
  outputs/plots/pr_curve_{target}.png
  outputs/plots/roc_curve_{target}.png

Usage:
    python eval.py                        # all targets, val + test splits
    python eval.py --target delinq --split stress   # stress-test the delinq model
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)

try:
    from .config import TARGETS, MODELS_DIR, METRICS_DIR, PLOTS_DIR, RECALL_AT_K, N_CALIBRATION_BINS
    from .data import load_split
except ImportError:
    from config import TARGETS, MODELS_DIR, METRICS_DIR, PLOTS_DIR, RECALL_AT_K, N_CALIBRATION_BINS
    from data import load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    """Fraction of positives captured in the top-k% by score.
    This is the operational metric: if you can only intervene on k% of loans,
    what fraction of actual defaults do you catch?
    """
    n = len(y_score)
    n_top = max(1, int(np.ceil(k * n)))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    return float(y_true[top_idx].sum() / max(y_true.sum(), 1))


def calibration_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = N_CALIBRATION_BINS,
) -> pd.DataFrame:
    """Bin predicted probabilities and compare to observed rates.
    A well-calibrated model has mean_pred ≈ mean_actual in each bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_score, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_lo":      round(bins[i], 2),
            "bin_hi":      round(bins[i + 1], 2),
            "n":           int(mask.sum()),
            "mean_pred":   round(float(y_score[mask].mean()), 4),
            "mean_actual": round(float(y_true[mask].mean()), 4),
        })
    return pd.DataFrame(rows)


# ── Per-model evaluation ───────────────────────────────────────────────────────

def evaluate_one(
    target: str,
    split: str,
    save_scores: bool = True,
) -> dict:
    model_path = MODELS_DIR / f"xgb_{target}.ubj"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_xgb.py first.")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    X, y = load_split(split, target)
    y_np = y.values

    y_score = model.predict_proba(X)[:, 1]

    # ── Core metrics ──────────────────────────────────────────────────────────
    pr_auc = average_precision_score(y_np, y_score)
    roc_auc = roc_auc_score(y_np, y_score)
    brier   = brier_score_loss(y_np, y_score)

    recalls = {
        f"recall@{int(k*100)}pct": recall_at_k(y_np, y_score, k)
        for k in RECALL_AT_K
    }

    # Baseline recall@k for a random model
    random_recalls = {
        f"random_recall@{int(k*100)}pct": k
        for k in RECALL_AT_K
    }

    metrics = {
        "target": target,
        "split":  split,
        "n":      len(y_np),
        "prevalence": round(float(y_np.mean()), 4),
        "pr_auc":  round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier":   round(brier, 4),
        **{k: round(v, 4) for k, v in recalls.items()},
        **random_recalls,
    }

    log.info(
        "[%s | %s] PR-AUC=%.4f  ROC-AUC=%.4f  Brier=%.4f  "
        "recall@5%%=%.4f  recall@10%%=%.4f",
        target, split,
        pr_auc, roc_auc, brier,
        recalls.get("recall@5pct", 0),
        recalls.get("recall@10pct", 0),
    )

    # ── Calibration ───────────────────────────────────────────────────────────
    cal = calibration_table(y_np, y_score)
    cal_path = METRICS_DIR / f"calibration_{target}_{split}.csv"
    cal.to_csv(cal_path, index=False)

    # ── Save raw scores for downstream (SHAP, intervention layer) ─────────────
    if save_scores:
        scores_df = pd.DataFrame({"y_true": y_np, "y_score": y_score})
        scores_df.to_parquet(
            METRICS_DIR / f"scores_{target}_{split}.parquet", index=False
        )

    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pr_curve(target: str, splits: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    model = xgb.XGBClassifier()
    model.load_model(MODELS_DIR / f"xgb_{target}.ubj")

    for split in splits:
        X, y = load_split(split, target)
        y_score = model.predict_proba(X)[:, 1]
        prec, rec, _ = precision_recall_curve(y, y_score)
        ap = average_precision_score(y, y_score)
        ax.plot(rec, prec, label=f"{split}  PR-AUC={ap:.4f}")

    prevalence = y.mean()
    ax.axhline(prevalence, linestyle="--", color="gray", alpha=0.5,
               label=f"Random baseline ({prevalence:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {target}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / f"pr_curve_{target}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("PR curve → %s", path)


def plot_calibration(target: str, split: str) -> None:
    cal = pd.read_csv(METRICS_DIR / f"calibration_{target}_{split}.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(cal["bin_lo"], cal["mean_actual"], width=0.1, alpha=0.6,
           align="edge", label="Observed rate")
    ax.plot(cal["bin_lo"] + 0.05, cal["mean_pred"], "o--",
            color="steelblue", label="Mean predicted")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Predicted probability bin")
    ax.set_ylabel("Rate")
    ax.set_title(f"Calibration — {target} | {split}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / f"calibration_{target}_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Calibration plot → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS.keys()))
    ap.add_argument("--split",  choices=["val", "test", "stress"])
    args = ap.parse_args()

    targets = [args.target] if args.target else list(TARGETS.keys())
    splits  = [args.split]  if args.split  else ["val", "test"]

    all_metrics = []
    for target in targets:
        for split in splits:
            try:
                m = evaluate_one(target, split)
                all_metrics.append(m)
            except FileNotFoundError as e:
                log.warning(str(e))

        plot_pr_curve(target, splits)
        for split in splits:
            plot_calibration(target, split)

    # ── Baseline table ────────────────────────────────────────────────────────
    if not all_metrics:
        raise RuntimeError("No evaluations ran. Train models first via train_xgb.py.")
    table = pd.DataFrame(all_metrics)
    table_path = METRICS_DIR / "baseline_table.csv"
    table.to_csv(table_path, index=False)

    print("\n" + "=" * 70)
    print("BASELINE TABLE")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nSaved → {table_path}")


if __name__ == "__main__":
    main()
