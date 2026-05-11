"""
CLEANUP — Delete ALL billable resources. Run this when done!

Deletes:
  1. SageMaker Endpoint       (the $0.13/hr cost)
  2. SageMaker Endpoint Config
  3. SageMaker Model(s)
  4. S3 bucket contents + bucket
  5. IAM Role (optional)

⚠️  This is DESTRUCTIVE. All model artifacts will be gone.
"""
import sys
from config import (
    get_session, BUCKET_NAME, MODEL_NAME, ENDPOINT_CONFIG_NAME,
    ENDPOINT_NAME, SAGEMAKER_ROLE_NAME, TRAINING_JOB_PREFIX,
)


def cleanup(keep_role=False, keep_s3=False):
    session = get_session()
    sm = session.client("sagemaker")
    s3 = session.client("s3")
    iam = session.client("iam")

    print("=" * 55)
    print("  CLEANUP — Deleting all billable resources")
    print("=" * 55)

    # ── 1. Delete Endpoint ────────────────────────────────────
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"✓ Deleted endpoint: {ENDPOINT_NAME}")
    except sm.exceptions.ClientError:
        print(f"  (no endpoint: {ENDPOINT_NAME})")

    # ── 2. Delete ALL Endpoint Configs ────────────────────────
    try:
        configs = sm.list_endpoint_configs(NameContains="loan-approval")
        for c in configs.get("EndpointConfigs", []):
            sm.delete_endpoint_config(EndpointConfigName=c["EndpointConfigName"])
            print(f"✓ Deleted endpoint config: {c['EndpointConfigName']}")
    except Exception as e:
        print(f"  (endpoint configs: {e})")

    # ── 3. Delete ALL Models ──────────────────────────────────
    try:
        models = sm.list_models(NameContains="loan-approval")
        for m in models.get("Models", []):
            sm.delete_model(ModelName=m["ModelName"])
            print(f"✓ Deleted model: {m['ModelName']}")
    except Exception as e:
        print(f"  (models: {e})")

    # ── 4. Delete S3 bucket ───────────────────────────────────
    if not keep_s3:
        try:
            # Must empty bucket before deleting
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=BUCKET_NAME):
                for obj in page.get("Contents", []):
                    s3.delete_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                    print(f"  Deleted s3://{BUCKET_NAME}/{obj['Key']}")
            s3.delete_bucket(Bucket=BUCKET_NAME)
            print(f"✓ Deleted bucket: {BUCKET_NAME}")
        except Exception as e:
            print(f"  (bucket: {e})")
    else:
        print(f"  Keeping S3 bucket: {BUCKET_NAME}")

    # ── 5. Delete IAM Role ────────────────────────────────────
    if not keep_role:
        try:
            # Detach policies before deleting role
            policies = iam.list_attached_role_policies(RoleName=SAGEMAKER_ROLE_NAME)
            for p in policies.get("AttachedPolicies", []):
                iam.detach_role_policy(RoleName=SAGEMAKER_ROLE_NAME, PolicyArn=p["PolicyArn"])
            iam.delete_role(RoleName=SAGEMAKER_ROLE_NAME)
            print(f"✓ Deleted IAM role: {SAGEMAKER_ROLE_NAME}")
        except Exception as e:
            print(f"  (role: {e})")
    else:
        print(f"  Keeping IAM role: {SAGEMAKER_ROLE_NAME}")

    print("\n" + "=" * 55)
    print("  ✓ All billable resources cleaned up!")
    print("  Your AWS bill should be $0 going forward.")
    print("=" * 55)


if __name__ == "__main__":
    keep_role = "--keep-role" in sys.argv
    keep_s3 = "--keep-s3" in sys.argv
    cleanup(keep_role=keep_role, keep_s3=keep_s3)
