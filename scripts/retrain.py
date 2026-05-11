"""
Retrain — Upload new data, retrain the model, update the endpoint.

REFRESHER — Retraining Flow:
  1. Upload new CSV to S3 (replaces old training data)
  2. Kick off a NEW Training Job (new model artifact)
  3. Update the existing endpoint to use the new model
  4. Old model is replaced seamlessly (blue-green deployment)

Usage:
  python retrain.py                          # retrain with existing S3 data
  python retrain.py --data path/to/new.csv   # upload new data first
"""
import argparse
import os
import time
import pandas as pd
import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from config import (
    get_session, ensure_role, ensure_bucket, REGION, BUCKET_NAME, S3_PREFIX,
    S3_OUTPUT_PATH, TRAINING_JOB_PREFIX, MODEL_NAME, ENDPOINT_CONFIG_NAME,
    ENDPOINT_NAME, INSTANCE_TYPE_TRAIN, INSTANCE_TYPE_DEPLOY,
)

AREA_MAP = {"rural": 0, "suburban": 1, "urban": 2}
EDU_MAP = {"high_school": 0, "graduate": 1, "postgraduate": 2}


def upload_new_data(session, csv_path):
    """Preprocess and upload new CSV to S3."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    print(f"New data: {len(df)} rows")

    df["area_type"] = df["area_type"].map(AREA_MAP)
    df["education"] = df["education"].map(EDU_MAP)

    target = "approved"
    features = [c for c in df.columns if c != target]
    df = df[[target] + features]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                         stratify=df[target])

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_df.to_csv(train_path, index=False, header=False)
    test_df.to_csv(test_path, index=False, header=False)

    s3 = session.client("s3")
    s3.upload_file(train_path, BUCKET_NAME, f"{S3_PREFIX}/train/train.csv")
    s3.upload_file(test_path, BUCKET_NAME, f"{S3_PREFIX}/test/test.csv")
    print(f"✓ Uploaded new data to S3")


def retrain(data_path=None):
    session = get_session()
    role_arn = ensure_role(session)
    sm_session = sagemaker.Session(boto_session=session)
    sm_client = session.client("sagemaker")

    if data_path:
        ensure_bucket(session)
        upload_new_data(session, data_path)

    container = image_uris.retrieve("xgboost", REGION, version="1.7-1")

    # ── New Training Job ──────────────────────────────────────
    xgb = sagemaker.estimator.Estimator(
        image_uri=container,
        role=role_arn,
        instance_count=1,
        instance_type=INSTANCE_TYPE_TRAIN,
        output_path=S3_OUTPUT_PATH,
        sagemaker_session=sm_session,
        base_job_name=f"{TRAINING_JOB_PREFIX}-retrain",
    )
    xgb.set_hyperparameters(
        max_depth=5, eta=0.2, gamma=4, min_child_weight=6,
        subsample=0.8, colsample_bytree=0.8, num_round=100,
        objective="binary:logistic", eval_metric="auc",
    )

    train_input = TrainingInput(s3_data=f"s3://{BUCKET_NAME}/{S3_PREFIX}/train/", content_type="text/csv")
    test_input = TrainingInput(s3_data=f"s3://{BUCKET_NAME}/{S3_PREFIX}/test/", content_type="text/csv")

    print("Starting retraining job...")
    xgb.fit({"train": train_input, "validation": test_input})
    print(f"✓ Retrained: {xgb.latest_training_job.name}")

    # ── Update Endpoint ───────────────────────────────────────
    new_model_name = f"{MODEL_NAME}-{int(time.time())}"
    new_epc_name = f"{ENDPOINT_CONFIG_NAME}-{int(time.time())}"

    sm_client.create_model(
        ModelName=new_model_name,
        PrimaryContainer={"Image": container, "ModelDataUrl": xgb.model_data},
        ExecutionRoleArn=role_arn,
    )

    sm_client.create_endpoint_config(
        EndpointConfigName=new_epc_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": new_model_name,
            "InstanceType": INSTANCE_TYPE_DEPLOY,
            "InitialInstanceCount": 1,
        }],
    )

    sm_client.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=new_epc_name,
    )
    print(f"✓ Endpoint updating with new model...")

    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME)
    print(f"✓ Endpoint updated and LIVE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to new CSV data file")
    args = parser.parse_args()
    retrain(args.data)
