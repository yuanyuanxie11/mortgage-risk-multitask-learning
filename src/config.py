"""Central config for the XGBoost baseline package."""

from pathlib import Path

# Running location: .../mortgage-risk-multitask-learning/src/config.py
SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
FEAT_ROOT = PROJECT_ROOT / "data" / "features"  # produced by src/features.py
OUT_ROOT = PROJECT_ROOT / "outputs"

MODELS_DIR   = OUT_ROOT / "models"
METRICS_DIR  = OUT_ROOT / "metrics"
PLOTS_DIR    = OUT_ROOT / "plots"
SHAP_DIR     = OUT_ROOT / "shap"

for _d in [MODELS_DIR, METRICS_DIR, PLOTS_DIR, SHAP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Feature columns (from src/schema.py + src/features.py) ───────────────────
STATIC_FEATURES = [
    "ORIG_RATE", "ORIG_UPB", "ORIG_TERM",
    "OLTV", "OCLTV", "DTI", "NUM_BO",
    "CSCORE_B", "CSCORE_C",
    "PURPOSE", "PROP", "OCC_STAT", "FIRST_FLAG", "CHANNEL",
    "STATE",
]

DYNAMIC_FEATURES = [
    "f12_n_obs",
    "f12_dlq_max",
    "f12_months_30p", "f12_months_60p", "f12_months_90p",
    "f12_ever_30", "f12_ever_60", "f12_ever_90",
    "f12_mod_flag",
    "f12_forbearance",
    "f12_last_upb",
    "f12_upb_paid_ratio",
    "f12_last_age_seen",
    # macro/collateral context
    "macro_mortgage30us",
    "macro_unrate",
    "macro_dgs10",
    "macro_term_spread_10y3m",
    "hpi_state_curr",
    "hpi_state_orig",
    "hpi_state_growth_1y",
    "hpi_state_ratio_curr_to_orig",
    "rate_incentive",
    "est_current_ltv_from_oltv",
]

ALL_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES

# Categorical columns (XGBoost handles natively with enable_categorical=True)
CAT_FEATURES = ["PURPOSE", "PROP", "OCC_STAT", "FIRST_FLAG", "CHANNEL", "STATE"]

# ── Targets ───────────────────────────────────────────────────────────────────
TARGETS = {
    "delinq": "y_delinq_90p",   # 90+ days past due in months 13-24
    "prepay": "y_prepay",       # prepayment (ZB_CODE=01) in months 13-24
}

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
# Conservative defaults tuned for large imbalanced tabular data.
XGB_BASE_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      ["aucpr", "logloss"],
    "tree_method":      "hist",          # fast on CPU; swap to "gpu_hist" if GPU
    "max_depth":        6,
    "learning_rate":    0.05,
    "n_estimators":     1000,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,              # large value fights imbalance
    "scale_pos_weight": None,            # set per-task in train_xgb.py
    "random_state":     42,
    "n_jobs":           -1,
    "enable_categorical": True,
}

# ── Evaluation settings ───────────────────────────────────────────────────────
RECALL_AT_K = [0.05, 0.10, 0.20]       # recall at top 5%, 10%, 20%
N_CALIBRATION_BINS = 10
SHAP_SAMPLE_N = 5_000                   # rows to use for SHAP (full is slow)
