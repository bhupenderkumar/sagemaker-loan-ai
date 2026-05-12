"""
Shared configuration for the Loan Prediction SageMaker project.
"""
import boto3

# ── AWS Settings ──────────────────────────────────────────────
AWS_PROFILE = "personal"
REGION = "ap-south-1"
ACCOUNT_ID = "749794722618"

# ── S3 ────────────────────────────────────────────────────────
BUCKET_NAME = f"loan-prediction-{ACCOUNT_ID}"
S3_PREFIX = "loan-prediction"
S3_TRAIN_PATH = f"s3://{BUCKET_NAME}/{S3_PREFIX}/train/train.csv"
S3_TEST_PATH = f"s3://{BUCKET_NAME}/{S3_PREFIX}/test/test.csv"
S3_OUTPUT_PATH = f"s3://{BUCKET_NAME}/{S3_PREFIX}/output"

# ── SageMaker ─────────────────────────────────────────────────
TRAINING_JOB_PREFIX = "loan-xgboost"
MODEL_NAME = "loan-approval-model"
ENDPOINT_CONFIG_NAME = "loan-approval-epc"
ENDPOINT_NAME = "loan-approval-endpoint"
INSTANCE_TYPE_TRAIN = "ml.m5.xlarge"
INSTANCE_TYPE_DEPLOY = "ml.m5.large"

# ── IAM Role ──────────────────────────────────────────────────
# This role must have SageMaker + S3 access.
# We'll create it if it doesn't exist (see create_role below).
SAGEMAKER_ROLE_NAME = "SageMakerLoanPredictionRole"
SAGEMAKER_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/{SAGEMAKER_ROLE_NAME}"

# ── Helper: get boto3 session ─────────────────────────────────
def get_session():
    return boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)


def ensure_bucket(session):
    """Create the S3 bucket if it doesn't exist."""
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"✓ Bucket exists: {BUCKET_NAME}")
    except Exception:
        print(f"Creating bucket: {BUCKET_NAME}")
        s3.create_bucket(
            Bucket=BUCKET_NAME,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
        print(f"✓ Bucket created: {BUCKET_NAME}")


def ensure_role(session):
    """Create the SageMaker execution role if it doesn't exist."""
    import json

    iam = session.client("iam")
    trust_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    })

    try:
        resp = iam.get_role(RoleName=SAGEMAKER_ROLE_NAME)
        print(f"✓ Role exists: {SAGEMAKER_ROLE_NAME}")
        return resp["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        print(f"Creating role: {SAGEMAKER_ROLE_NAME}")
        resp = iam.create_role(
            RoleName=SAGEMAKER_ROLE_NAME,
            AssumeRolePolicyDocument=trust_policy,
            Description="SageMaker execution role for loan prediction",
        )
        # Attach required policies
        for policy_arn in [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        ]:
            iam.attach_role_policy(RoleName=SAGEMAKER_ROLE_NAME, PolicyArn=policy_arn)

        print(f"✓ Role created: {SAGEMAKER_ROLE_NAME}")
        return resp["Role"]["Arn"]
