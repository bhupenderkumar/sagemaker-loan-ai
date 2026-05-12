"""
Step 2: Train — Launch a SageMaker XGBoost Training Job using boto3.

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
from config import (
    get_session, ensure_role, REGION, BUCKET_NAME, S3_PREFIX,
    S3_OUTPUT_PATH, TRAINING_JOB_PREFIX, INSTANCE_TYPE_TRAIN,
)

# XGBoost container image for ap-south-1 (v1.7-1)
XGBOOST_IMAGE = f"720646828776.dkr.ecr.{REGION}.amazonaws.com/sagemaker-xgboost:1.7-1"


def train():
    session = get_session()
    role_arn = ensure_role(session)
    sm = session.client("sagemaker")

    job_name = f"{TRAINING_JOB_PREFIX}-{int(time.time())}"
    print(f"XGBoost container: {XGBOOST_IMAGE}")
    print(f"Starting training job: {job_name}")

    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": XGBOOST_IMAGE,
            "TrainingInputMode": "File",
        },
        RoleArn=role_arn,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{BUCKET_NAME}/{S3_PREFIX}/train/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{BUCKET_NAME}/{S3_PREFIX}/test/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            },
        ],
        OutputDataConfig={"S3OutputPath": S3_OUTPUT_PATH},
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE_TRAIN,
            "InstanceCount": 1,
            "VolumeSizeInGB": 5,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 600},
        HyperParameters={
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "min_child_weight": "6",
            "subsample": "0.8",
            "colsample_bytree": "0.8",
            "num_round": "100",
            "objective": "binary:logistic",
            "eval_metric": "auc",
        },
    )

    # Wait for job to complete
    print("Training in progress...")
    while True:
        resp = sm.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        if status == "Completed":
            model_artifact = resp["ModelArtifacts"]["S3ModelArtifacts"]
            print(f"\n✓ Training complete!")
            print(f"  Job name:       {job_name}")
            print(f"  Model artifact: {model_artifact}")
            print(f"\nNext step: python deploy.py")
            break
        elif status == "Failed":
            reason = resp.get("FailureReason", "Unknown")
            print(f"\n✗ Training FAILED: {reason}")
            break
        else:
            print(f"  Status: {status}...")
            time.sleep(30)


if __name__ == "__main__":
    train()
