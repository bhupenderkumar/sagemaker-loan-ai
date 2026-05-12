# Loan Prediction — Amazon SageMaker + XGBoost

A complete ML pipeline that trains, deploys, and retrains a **loan-approval classifier** using Amazon SageMaker with XGBoost. Includes a Flask web UI for live demo.

## Architecture

```
CSV data (S3, 1 000 rows)
    ↓
Training Job  (XGBoost 1.7 on ml.m5.xlarge)
    ↓
Model Artifact  (S3: model.tar.gz)
    ↓
Real-time Endpoint  (ml.m5.large)
    ↓
Flask Web UI  →  POST /predict  →  SageMaker Inference
```

## Quick Start

```bash
# 1. Create & activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip3 install boto3 pandas scikit-learn flask

# 3. One-command pipeline (prepare → train → deploy)
./scripts/start.sh

# 4. Launch the demo web UI
python3 web/app.py          # opens http://localhost:5050
```

Or run each step individually:

```bash
python3 scripts/prepare_data.py   # preprocess & upload to S3
python3 scripts/train.py          # ~5 min training job
python3 scripts/deploy.py         # ~5-8 min endpoint deployment
python3 scripts/predict.py        # CLI predictions
```

## Retrain with New Data

```bash
# Generate a fresh 1 000-row synthetic dataset
python3 scripts/generate_data.py

# Retrain & hot-swap the live endpoint
python3 scripts/retrain.py --data data/loan_data.csv
```

## Cleanup (stop billing!)

```bash
./scripts/nuke.sh                 # deletes endpoint, models, S3, IAM role
./scripts/status.sh               # check if anything is still running
```

> **⚠️ Endpoints cost ~$0.13/hour.** Always run `nuke.sh` when done.

## Project Structure

| File / Folder              | Purpose                                          |
|----------------------------|--------------------------------------------------|
| `data/loan_data.csv`       | Synthetic training data (1 000 rows)             |
| `scripts/config.py`        | Shared AWS config (bucket, role, region)          |
| `scripts/generate_data.py` | Generate realistic synthetic loan data            |
| `scripts/prepare_data.py`  | Preprocess CSV → train/test split → upload to S3  |
| `scripts/train.py`         | Launch SageMaker XGBoost training job             |
| `scripts/deploy.py`        | Deploy trained model to a real-time endpoint      |
| `scripts/predict.py`       | CLI: send sample loan applications to endpoint    |
| `scripts/retrain.py`       | Upload new data + retrain + update endpoint       |
| `scripts/cleanup.py`       | Delete endpoint, model, config, S3, IAM role      |
| `scripts/start.sh`         | One-command full pipeline                         |
| `scripts/nuke.sh`          | One-command full cleanup                          |
| `scripts/status.sh`        | Check running resources & costs                   |
| `web/app.py`               | Flask web server (demo UI)                        |
| `web/templates/index.html` | Loan application form + prediction results UI     |

## Feature Columns

| Feature           | Type    | Description                           |
|-------------------|---------|---------------------------------------|
| age               | int     | Applicant age (20–60)                 |
| income            | float   | Annual income (₹)                    |
| loan_amount       | float   | Requested loan amount (₹)            |
| credit_score      | int     | Credit score (300–850)                |
| employment_years  | int     | Years of employment                   |
| debt_to_income    | float   | Debt-to-income ratio (0.0–1.0)       |
| area_type         | cat     | urban / suburban / rural              |
| education         | cat     | high_school / graduate / postgraduate |
| self_employed     | binary  | 0 or 1                                |
| property_value    | float   | Collateral property value (₹)        |

## AWS Details

| Resource        | Value                                   |
|-----------------|-----------------------------------------|
| Region          | ap-south-1                              |
| S3 Bucket       | loan-prediction-749794722618            |
| Endpoint        | loan-approval-endpoint                  |
| Algorithm       | XGBoost 1.7-1 (built-in container)      |
| Train instance  | ml.m5.xlarge                            |
| Deploy instance | ml.m5.large                             |

## Cost Estimate (Free Tier)

- **Training:** ml.m5.xlarge → ~$0.27/hour (training takes ~5 min ≈ $0.02)
- **Endpoint:** ml.m5.large → ~$0.13/hour (**DELETE when not in use!**)
- **S3:** pennies for this data size
