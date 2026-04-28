# Mortgage Risk Decision System
### From large-scale loan data to decision-ready risk analytics

> We built this project to show full-stack ML engineering: robust data systems, rigorous evaluation, and transferable modeling skills (including PyTorch multi-task learning design).

## What this project demonstrates

This repository is a portfolio-quality implementation of a real-world ML workflow:

- **Data engineering at scale** on Fannie Mae longitudinal mortgage data.
- **Leakage-safe supervised learning** with time-aware windows and vintage splits.
- **Strong baseline discipline** using XGBoost + PR-AUC/recall@k/calibration.
- **Model interpretability** with SHAP to connect metrics to business behavior.
- **Decision-system thinking** beyond pure prediction (intervention analysis path).

The emphasis is not only "can we train a model?" but "can we build a reliable system that another team could operate and extend?"

## Why this is transferable

Although the domain is mortgage risk, the core skills transfer directly to many ML roles:

- designing and operating **large tabular/time-series pipelines**
- enforcing **train/validation/test temporal correctness**
- handling **class imbalance** and choosing the right metrics
- building **interpretable, decision-oriented** model outputs
- evolving from baseline models to **PyTorch multi-task architectures**

## System architecture

```text
Performance_All.zip (Fannie Mae)
        ->
Streaming typed ingestion (DuckDB, Parquet partitions)
        ->
Leakage-safe feature store (per loan, per vintage)
        ->
Single-task XGBoost baselines (delinq / prepay)
        ->
PyTorch multi-task model (planned next milestone)
        ->
Intervention analysis (propensity-adjusted comparison)
```

## Repository layout

```text
mortgage-risk-multitask-learning/
  src/
    schema.py         # 110-column schema
    ingest.py         # streaming zip -> Parquet
    features.py       # leakage-safe features + labels + macro join
    splits.py         # vintage split policy
    views.py          # shared DuckDB views
    data.py           # baseline data loader
    data_quality.py   # null/range/drift contracts
    train_xgb.py      # baseline training
    eval.py           # PR-AUC, recall@k, calibration
    shap_report.py    # feature importance analysis
    model_mtl.py      # shared-encoder MTL model definition
    train_mtl.py      # MTL training loop (PyTorch)
    eval_mtl.py       # MTL evaluation table
    run_baseline.sh   # one-command baseline pipeline
    run_mtl.sh        # one-command MTL train+eval
  scripts/
    01_convert_one_vintage.sh
    02_convert_all.sh
    03_validate_vintage.py
    download_macro.py
  data/
    raw_parquet/
    features/
    macro/
  outputs/
    models/
    metrics/
    plots/
    shap/
```

## Engineering highlights

- **No full unzip**: zip members are streamed directly into DuckDB via file descriptors.
- **Typed schema ingest**: avoids silent inference drift across vintages.
- **Partition-aware storage**: Parquet with Hive-style partitioning for fast scans.
- **Critical edge-case handling**: terminal rows with null `LOAN_AGE` are preserved via date-derived fallback logic.
- **Temporal leakage control**:
  - feature window: months `0-12`
  - label window: months `13-24`
  - split policy: by origination vintage, never random rows

## Data sources

- Fannie Mae single-family performance data (`Performance_All.zip`)
- FHFA HPI datasets: [https://www.fhfa.gov/data/hpi/datasets](https://www.fhfa.gov/data/hpi/datasets)
- FRED macro series:
  - `MORTGAGE30US`
  - `UNRATE`
  - `DGS10`
  - `GS3M`

## Quick start

### 1) Environment

```bash
conda env create -f environment.yml
conda activate mortgage-risk
# or: pip install -r requirements.txt
# or: ./setup_env.sh   (recommended on mixed Linux/TLJH environments)
```

If PyTorch import fails due to CUDA wheel mismatch, force CPU wheel:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

### 2) Download macro data

```bash
python scripts/download_macro.py
python scripts/download_macro.py --check
```

### 3) Build one-vintage pipeline (smoke test)

```bash
./scripts/01_convert_one_vintage.sh 2018Q1
python -m src.features 2018Q1
python scripts/03_validate_vintage.py 2018Q1 --skip-csv-rowcount
```

### 4) Run full baseline package

```bash
chmod +x src/run_baseline.sh
./src/run_baseline.sh
```

### 5) Train + evaluate MTL

```bash
chmod +x src/run_mtl.sh
./src/run_mtl.sh
```

## Step-by-step runbook (recommended)

This is the exact order we use for reliable runs.

### Step 0 — one-time setup

```bash
cd /nfs/home/ucb9673/mortgage-risk-multitask-learning
chmod +x setup_env.sh src/run_baseline.sh scripts/*.sh
./setup_env.sh --venv
source venv/bin/activate
```

### Step 1 — macro data

```bash
python scripts/download_macro.py
python scripts/download_macro.py --check
```

Expected:
- always: `data/macro/fred_monthly.parquet`
- if FHFA endpoint succeeds (or local fallback file provided): `data/macro/hpi_state_quarterly.parquet`

If FHFA auto-download fails, place a FHFA CSV at `data/macro/_hpi_state_raw.csv` and re-run.

### Step 2 — smoke test on one vintage

```bash
./scripts/01_convert_one_vintage.sh 2018Q1 --overwrite
python -m src.features 2018Q1 --overwrite
python scripts/03_validate_vintage.py 2018Q1 --skip-csv-rowcount
```

Quick checks:
- feature file exists: `data/features/vintage=2018Q1/part-0.parquet`
- logs show non-zero loans and sensible prevalence values.

### Step 3 — full raw conversion

```bash
./scripts/02_convert_all.sh --parallel 4
```

Output summary:
- `logs/convert_all_summary.tsv`
- one log per vintage in `logs/convert_*.log`

### Step 4 — build features for all vintages

```bash
for y in $(seq 2000 2025); do
  for q in Q1 Q2 Q3 Q4; do
    v="${y}${q}"
    python -m src.features "$v" --overwrite || true
  done
done
```

Tip: keep `|| true` while backfilling because late/future vintages may not exist.

### Step 5 — baseline package

```bash
./src/run_baseline.sh
```

Primary artifacts:
- `outputs/metrics/baseline_table.csv`
- `outputs/metrics/data_quality_report.csv`
- `outputs/plots/pr_curve_*.png`
- `outputs/plots/calibration_*.png`
- `outputs/shap/shap_comparison_*.png`

### Step 6 — interpretation and iteration

1. Review baseline table by target and split.
2. Review calibration and SHAP charts.
3. Adjust features/hyperparameters.
4. Re-run `./src/run_baseline.sh`.

### Step 7 — multi-task learning

```bash
./src/run_mtl.sh
```

Primary MTL artifacts:
- `outputs/models/mtl_best.pt`
- `outputs/models/mtl_preprocessor.pkl`
- `outputs/metrics/mtl_training_history.csv`
- `outputs/metrics/mtl_eval_table.csv`

## Evaluation philosophy

For delinquency, class imbalance is severe, so we center evaluation on:

- **PR-AUC** (primary)
- **recall@k** (operational capacity framing)
- **calibration + Brier score** (probability reliability)

`ROC-AUC` is retained as a secondary diagnostic, not the lead metric.

## Current status (honest snapshot)

- Streaming ingestion pipeline: **complete**
- Feature store + split framework: **complete**
- Macro integration:
  - FRED join path: **complete**
  - FHFA HPI auto-download: **best-effort with fallback/manual path**
- XGBoost baseline package: **implemented**
- Full-vintage backfill: **in progress**
- PyTorch multi-task training/evaluation: **implemented**
- Intervention impact analysis: **planned after benchmark lock**

## Roadmap

1. Complete full backfill across all vintages.
2. Lock baseline benchmark table for `delinq` and `prepay`.
3. Implement and evaluate PyTorch multi-task model against baseline.
4. Add propensity-adjusted intervention analysis and impact reporting.

## For recruiters and collaborators

This project is built to evidence:

- practical ML engineering under scale constraints,
- careful model evaluation under imbalance and temporal risk,
- explainability and model governance habits,
- and the ability to translate modeling outputs into product/decision workflows.

If you want one place to start, review:

1. `src/features.py` (temporal + leakage-safe construction)
2. `src/train_xgb.py` + `src/eval.py` (baseline rigor)
3. `outputs/metrics/baseline_table.csv` (when generated)

---

Built for research and portfolio purposes using public datasets.
