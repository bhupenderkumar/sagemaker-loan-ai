"""
Step 2: Train — Launch a SageMaker XGBoost Training Job.

REFRESHER — How SageMaker Training Works:
  1. You specify an ALGORITHM CONTAINER IMAGE (AWS provides built-in ones
     for XGBoost, Linear Learner, etc.)
  2. SageMaker spins up the instance you pick (ml.m5.large)
  3. It downloads your S3 data into the container
  4. Runs XGBoost with the hyperparameters you set
  5. Saves the model artifact (model.tar.gz) to your S3 output path
  6. Shuts down the instance (you only pay for training time)

HYPERPARAMETERS explained:
  - max_depth: Tree depth. Deeper = more complex, risk overfitting. 5 is safe.
  - eta: Learning rate. Lower = slower but more accurate. 0.2 is balanced.
  - num_round: Number of boosting rounds (trees). 100 is good for small data.
  - objective: binary:logistic = outputs probability 0-1 for binary classification.
  - eval_metric: auc = Area Under ROC Curve (how well it separates 0s from 1s).
"""
import time
import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from config import (
    get_session, ensure_role, REGION, BUCKET_NAME, S3_PREFIX,
    S3_OUTPUT_PATH, TRAINING_JOB_PREFIX, INSTANCE_TYPE_TRAIN,
)


def train():
    session = get_session()
    role_arn = ensure_role(session)

    sm_session = sagemaker.Session(boto_session=session)

    # ── Get the XGBoost container image URI ───────────────────
    # AWS maintains pre-built containers for popular algorithms
    container = image_uris.retrieve("xgboost", REGION, version="1.7-1")
    print(f"XGBoost container: {container}")

    # ── Create the Estimator ──────────────────────────────────
    # An Estimator = wrapper that defines: container + instance + hyperparams
    xgb = sagemaker.estimator.Estimator(
        image_uri=container,
        role=role_arn,
        instance_count=1,
        instance_type=INSTANCE_TYPE_TRAIN,
        output_path=S3_OUTPUT_PATH,
        sagemaker_session=sm_session,
        base_job_name=TRAINING_JOB_PREFIX,
    )

    # ── Set hyperparameters ───────────────────────────────────
    xgb.set_hyperparameters(
        max_depth=5,           # tree depth
        eta=0.2,               # learning rate
        gamma=4,               # min loss reduction to split
        min_child_weight=6,    # min samples in leaf
        subsample=0.8,         # % of rows per tree
        colsample_bytree=0.8,  # % of features per tree
        num_round=100,         # number of boosting rounds
        objective="binary:logistic",
        eval_metric="auc",
    )

    # ── Point to S3 data ──────────────────────────────────────
    train_input = TrainingInput(
        s3_data=f"s3://{BUCKET_NAME}/{S3_PREFIX}/train/",
        content_type="text/csv",
    )
    test_input = TrainingInput(
        s3_data=f"s3://{BUCKET_NAME}/{S3_PREFIX}/test/",
        content_type="text/csv",
    )

    # ── Launch Training ───────────────────────────────────────
    print("Starting training job...")
    xgb.fit({"train": train_input, "validation": test_input})

    # The job name is auto-generated
    job_name = xgb.latest_training_job.name
    print(f"\n✓ Training complete!")
    print(f"  Job name:     {job_name}")
    print(f"  Model artifact: {xgb.model_data}")
    print(f"\nNext step: python deploy.py")

    # Save job name for deploy script
    with open("/tmp/loan_training_job.txt", "w") as f:
        f.write(job_name)

    return xgb


if __name__ == "__main__":
    train()
