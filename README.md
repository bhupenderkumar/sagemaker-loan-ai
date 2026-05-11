# ──────────────────────────────────────────────────────────────
# Loan Prediction — SageMaker Project
# ──────────────────────────────────────────────────────────────
# This project builds, deploys, and retrains a loan-approval
# classifier using Amazon SageMaker + XGBoost.
#
# QUICK REFRESHER (AWS ML Concepts):
# ───────────────────────────────────
# 1. SageMaker Training Job  — Spins up an EC2 instance, runs
#    your algorithm on S3 data, saves the model artifact to S3.
#
# 2. Model                   — A pointer to the artifact + the
#    container image that knows how to load & serve it.
#
# 3. Endpoint Configuration  — Defines instance type, count,
#    and model variant (supports A/B testing).
#
# 4. Endpoint                — A live HTTPS URL that receives
#    JSON/CSV, runs inference, returns predictions in real time.
#
# 5. Retraining              — Upload new CSV → kick off a new
#    Training Job → update the Endpoint with the fresh model.
#
# ML PIPELINE FLOW:
# ─────────────────
#   CSV data (S3)
#       ↓
#   Training Job (XGBoost on ml.m5.large)
#       ↓
#   Model Artifact (S3: model.tar.gz)
#       ↓
#   Deploy → Endpoint (ml.m5.large)
#       ↓
#   Client sends { age, income, ... } → gets { approved: 1/0, probability: 0.87 }
#
# RETRAINING:
#   New CSV → S3 → retrain.py → new Training Job → update Endpoint
#
# ──────────────────────────────────────────────────────────────

## Setup

```bash
pip install boto3 sagemaker pandas scikit-learn
```

## Files

| File                       | Purpose                                        |
|----------------------------|-------------------------------------------------|
| data/loan_data.csv         | Sample training data (50 rows)                  |
| scripts/config.py          | Shared AWS config (bucket, role, region)         |
| scripts/prepare_data.py    | Preprocess CSV → train/test split → upload to S3 |
| scripts/train.py           | Launch SageMaker XGBoost Training Job            |
| scripts/deploy.py          | Deploy trained model to a real-time endpoint     |
| scripts/predict.py         | Send a loan application to the endpoint          |
| scripts/retrain.py         | Upload new data + retrain + update endpoint      |
| scripts/cleanup.py         | Delete endpoint, model, config (save $$$)        |

## Usage

```bash
# 1. Prepare & upload data
python scripts/prepare_data.py

# 2. Train the model
python scripts/train.py

# 3. Deploy to endpoint
python scripts/deploy.py

# 4. Make predictions
python scripts/predict.py

# 5. Retrain with new data
python scripts/retrain.py --data data/new_loan_data.csv

# 6. Cleanup (IMPORTANT — endpoints cost money!)
python scripts/cleanup.py
```

## Feature Columns

| Feature           | Type    | Description                          |
|-------------------|---------|--------------------------------------|
| age               | int     | Applicant age                        |
| income            | float   | Annual income                        |
| loan_amount       | float   | Requested loan amount                |
| credit_score      | int     | Credit score (300-850)               |
| employment_years  | int     | Years at current employer            |
| debt_to_income    | float   | Debt-to-income ratio                 |
| area_type         | cat     | urban / suburban / rural             |
| education         | cat     | high_school / graduate / postgraduate|
| self_employed     | binary  | 0 or 1                               |
| property_value    | float   | Property value                       |

## Cost Estimate (Free Tier)
- Training: ml.m5.large → ~$0.10/hour (training takes ~5 min)
- Endpoint: ml.m5.large → ~$0.13/hour (DELETE when not in use!)
- S3: pennies for this data size
