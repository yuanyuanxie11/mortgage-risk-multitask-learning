"""Train XGBoost single-task baselines for delinquency and prepayment.

Trains one model per target, uses early stopping on val PR-AUC,
saves model + feature importance + training curve.

Usage:
    python train_xgb.py                   # both targets
    python train_xgb.py --target delinq   # one target only
    python train_xgb.py --tune            # Optuna HPO first (slow, ~2h)
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from .config import XGB_BASE_PARAMS, TARGETS, MODELS_DIR, METRICS_DIR
    from .data import load_split
except ImportError:
    from config import XGB_BASE_PARAMS, TARGETS, MODELS_DIR, METRICS_DIR
    from data import load_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Ratio of negatives to positives — standard XGBoost imbalance correction."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    if pos == 0:
        return 1.0
    spw = neg / pos
    log.info("  scale_pos_weight = %.1f  (neg=%d pos=%d)", spw, neg, pos)
    return float(spw)


def train_one(target: str) -> dict:
    """Train a single XGBoost model for the given target.
    Returns a metrics dict for the baseline table.
    """
    log.info("=" * 60)
    log.info("Training target: %s", target)
    log.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, y_train = load_split("train", target)
    X_val,   y_val   = load_split("val",   target)

    # ── Build model ───────────────────────────────────────────────────────────
    params = dict(XGB_BASE_PARAMS)
    params["scale_pos_weight"] = compute_scale_pos_weight(y_train)

    model = xgb.XGBClassifier(
        **params,
        early_stopping_rounds=50,
        callbacks=[xgb.callback.EvaluationMonitor(period=50)],
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0

    best_round = model.best_iteration
    log.info("Best round: %d  (elapsed %.0fs)", best_round, elapsed)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / f"xgb_{target}.ubj"
    model.save_model(model_path)
    log.info("Model saved → %s", model_path)

    # ── Feature importance table ───────────────────────────────────────────────
    booster = model.get_booster()
    gain_map = booster.get_score(importance_type="gain")
    weight_map = booster.get_score(importance_type="weight")
    features = list(model.feature_names_in_)
    fi = pd.DataFrame(
        {
            "feature": features,
            "gain": [gain_map.get(f, 0.0) for f in features],
            "weight": [weight_map.get(f, 0.0) for f in features],
        }
    ).sort_values(["gain", "weight"], ascending=False)
    fi_path = METRICS_DIR / f"feature_importance_{target}.csv"
    fi.to_csv(fi_path, index=False)

    # Training curve (evals_result)
    evals = model.evals_result()
    curve_path = METRICS_DIR / f"training_curve_{target}.json"
    with open(curve_path, "w") as f:
        json.dump(evals, f)

    return {
        "target": target,
        "best_round": best_round,
        "train_prevalence": float(y_train.mean()),
        "val_prevalence":   float(y_val.mean()),
        "n_train": len(y_train),
        "n_val":   len(y_val),
        "elapsed_sec": round(elapsed),
        "model_path": str(model_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS.keys()),
                    help="train one target only; default is both")
    ap.add_argument("--tune", action="store_true",
                    help="run Optuna HPO before training (slow)")
    args = ap.parse_args()

    targets = [args.target] if args.target else list(TARGETS.keys())

    if args.tune:
        log.info("HPO requested — run tune_xgb.py first, then re-run without --tune")
        return

    results = []
    for t in targets:
        info = train_one(t)
        results.append(info)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n  target       : {r['target']}")
        print(f"  n_train      : {r['n_train']:,}")
        print(f"  prevalence   : train={r['train_prevalence']:.4f}  val={r['val_prevalence']:.4f}")
        print(f"  best_round   : {r['best_round']}")
        print(f"  elapsed      : {r['elapsed_sec']}s")
        print(f"  saved to     : {r['model_path']}")

    summary_path = METRICS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Summary → %s", summary_path)


if __name__ == "__main__":
    main()
