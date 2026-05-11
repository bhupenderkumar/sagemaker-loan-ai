"""
Step 1: Prepare data — preprocess, split, upload to S3.

REFRESHER — What this does:
  1. Reads the raw CSV
  2. Encodes categorical columns (area_type, education) → numbers
     (XGBoost needs numeric input only)
  3. Splits into 80% train / 20% test
  4. Puts the TARGET column FIRST (SageMaker XGBoost expects this)
  5. Uploads both CSVs to S3

WHY TARGET FIRST?
  SageMaker's built-in XGBoost reads CSV without headers.
  It assumes column 0 = label, columns 1..N = features.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import get_session, ensure_bucket, BUCKET_NAME, S3_PREFIX

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def prepare():
    session = get_session()
    ensure_bucket(session)

    # ── Load ──────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, "loan_data.csv"))
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # ── Encode categoricals ───────────────────────────────────
    area_map = {"rural": 0, "suburban": 1, "urban": 2}
    edu_map = {"high_school": 0, "graduate": 1, "postgraduate": 2}
    df["area_type"] = df["area_type"].map(area_map)
    df["education"] = df["education"].map(edu_map)

    # ── Target first (SageMaker XGBoost requirement) ──────────
    target = "approved"
    features = [c for c in df.columns if c != target]
    df = df[[target] + features]

    # ── Split ─────────────────────────────────────────────────
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                         stratify=df[target])
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    # ── Save locally ──────────────────────────────────────────
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    train_df.to_csv(train_path, index=False, header=False)
    test_df.to_csv(test_path, index=False, header=False)

    # ── Upload to S3 ──────────────────────────────────────────
    s3 = session.client("s3")
    s3.upload_file(train_path, BUCKET_NAME, f"{S3_PREFIX}/train/train.csv")
    s3.upload_file(test_path, BUCKET_NAME, f"{S3_PREFIX}/test/test.csv")
    print(f"✓ Uploaded to s3://{BUCKET_NAME}/{S3_PREFIX}/train/")
    print(f"✓ Uploaded to s3://{BUCKET_NAME}/{S3_PREFIX}/test/")

    # Also save the encoding maps for inference
    import json
    maps = {"area_type": area_map, "education": edu_map}
    maps_path = os.path.join(DATA_DIR, "encoding_maps.json")
    with open(maps_path, "w") as f:
        json.dump(maps, f, indent=2)
    print(f"✓ Saved encoding maps to {maps_path}")


if __name__ == "__main__":
    prepare()
