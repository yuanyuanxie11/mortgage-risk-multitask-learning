"""Generate SHAP feature importance reports for both XGBoost baselines.

Produces:
  outputs/shap/shap_summary_{target}.png      ← beeswarm plot
  outputs/shap/shap_bar_{target}.png          ← mean |SHAP| bar chart
  outputs/shap/shap_values_{target}.parquet   ← raw SHAP values (for MTL comparison)
  outputs/shap/shap_top20_{target}.csv        ← top-20 features by mean |SHAP|

Run AFTER eval.py so score parquets exist (used to pick a representative sample).

Usage:
    python shap_report.py                     # both targets, val split
    python shap_report.py --target delinq --split test
    python shap_report.py --n-samples 10000   # larger sample (slower)
"""
from __future__ import annotations
import argparse
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

try:
    from .config import TARGETS, MODELS_DIR, SHAP_DIR, SHAP_SAMPLE_N
    from .data import load_split
except ImportError:
    from config import TARGETS, MODELS_DIR, SHAP_DIR, SHAP_SAMPLE_N
    from data import load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def shap_report_one(
    target: str,
    split: str = "val",
    n_samples: int = SHAP_SAMPLE_N,
) -> pd.DataFrame:
    """Compute and save SHAP values for one target/split combination.
    Returns a DataFrame of mean |SHAP| per feature (for cross-task comparison).
    """
    model_path = MODELS_DIR / f"xgb_{target}.ubj"
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    X, y = load_split(split, target)

    # Sample for speed — SHAP on full dataset takes very long.
    # For delinquency (rare positive class), random sampling is unstable.
    # Use stratified sampling to force enough positive examples.
    if len(X) > n_samples:
        rng = np.random.default_rng(42)
        if target == "delinq":
            pos_idx = np.where(y.values == 1)[0]
            neg_idx = np.where(y.values == 0)[0]
            n_pos = min(len(pos_idx), n_samples // 2)
            n_neg = min(len(neg_idx), n_samples - n_pos)
            if n_pos == 0 or n_neg == 0:
                idx = rng.choice(len(X), size=n_samples, replace=False)
            else:
                take_pos = rng.choice(pos_idx, size=n_pos, replace=False)
                take_neg = rng.choice(neg_idx, size=n_neg, replace=False)
                idx = np.concatenate([take_pos, take_neg])
                rng.shuffle(idx)
        else:
            idx = rng.choice(len(X), size=n_samples, replace=False)
        X_sample = X.iloc[idx].reset_index(drop=True)
        y_sample = y.iloc[idx].reset_index(drop=True)
    else:
        X_sample, y_sample = X, y

    log.info(
        "[%s | %s] Computing SHAP on %d samples (prevalence=%.4f) ...",
        target, split, len(X_sample), float(y_sample.mean())
    )

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)           # Explanation object

    # ── Save raw SHAP values ───────────────────────────────────────────────────
    shap_df = pd.DataFrame(
        shap_values.values,
        columns=X_sample.columns,
    )
    shap_df.to_parquet(SHAP_DIR / f"shap_values_{target}_{split}.parquet", index=False)

    # ── Mean |SHAP| ranking ───────────────────────────────────────────────────
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top20 = mean_abs.head(20).reset_index()
    top20.columns = ["feature", "mean_abs_shap"]
    top20.to_csv(SHAP_DIR / f"shap_top20_{target}_{split}.csv", index=False)

    # ── Beeswarm plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP beeswarm — {target} | {split}", pad=12)
    plt.tight_layout()
    beeswarm_path = SHAP_DIR / f"shap_summary_{target}_{split}.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Beeswarm → %s", beeswarm_path)

    # ── Bar chart (mean |SHAP|) ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    top_n = 20
    features = mean_abs.head(top_n).index[::-1]
    values   = mean_abs.head(top_n).values[::-1]
    ax.barh(features, values, color="steelblue", alpha=0.8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature importance (SHAP) — {target} | {split}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    bar_path = SHAP_DIR / f"shap_bar_{target}_{split}.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    log.info("Bar chart → %s", bar_path)

    return top20


def compare_tasks(split: str = "val") -> None:
    """Side-by-side SHAP comparison across delinq and prepay.
    This is the key plot for your interview: which features are shared
    (motivate the MTL shared encoder) vs task-specific.
    """
    frames = {}
    for target in TARGETS:
        csv = SHAP_DIR / f"shap_top20_{target}_{split}.csv"
        if not csv.exists():
            log.warning("Missing %s — run shap_report.py for %s first", csv, target)
            return
        frames[target] = pd.read_csv(csv).set_index("feature")["mean_abs_shap"]

    # Align on union of top-20 features
    all_features = sorted(
        set(frames["delinq"].index) | set(frames["prepay"].index)
    )
    df = pd.DataFrame(frames).reindex(all_features).fillna(0)
    df["total"] = df.sum(axis=1)
    df = df.sort_values("total", ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(9, 8))
    x = np.arange(len(df))
    w = 0.38
    ax.barh(x - w/2, df["delinq"], w, label="Delinquency 90+", color="#E24B4A", alpha=0.8)
    ax.barh(x + w/2, df["prepay"], w, label="Prepayment",       color="#378ADD", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP comparison: delinquency vs prepayment | {split}")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = SHAP_DIR / f"shap_comparison_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Task comparison → %s", path)
    print(f"\nKey MTL insight: features with high SHAP in BOTH tasks = candidates for shared encoder.")
    print(df[["delinq", "prepay"]].to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS.keys()))
    ap.add_argument("--split",  default="val", choices=["val", "test", "stress"])
    ap.add_argument("--n-samples", type=int, default=SHAP_SAMPLE_N)
    args = ap.parse_args()

    targets = [args.target] if args.target else list(TARGETS.keys())

    for target in targets:
        shap_report_one(target, args.split, args.n_samples)

    if len(targets) == 2:
        compare_tasks(args.split)


if __name__ == "__main__":
    main()
