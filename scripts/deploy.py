"""
Step 3: Deploy — Create a SageMaker real-time endpoint.

REFRESHER — Deployment Flow:
  1. Model:    Points to the artifact (model.tar.gz in S3) + container image
  2. Endpoint Config: Defines instance type, count, variant name
  3. Endpoint: The live HTTPS URL. Takes ~5 min to spin up.

  Client → HTTPS POST → Endpoint → Container loads model → Prediction → Response

COSTS:
  An endpoint runs 24/7 on the instance you pick.
  ml.m5.large ≈ $0.13/hour ≈ $3.12/day ≈ $94/month
  ⚠️  ALWAYS run cleanup.py when done testing!
"""
import sagemaker
from sagemaker import image_uris
from config import (
    get_session, ensure_role, REGION, BUCKET_NAME, S3_PREFIX,
    MODEL_NAME, ENDPOINT_CONFIG_NAME, ENDPOINT_NAME,
    INSTANCE_TYPE_DEPLOY, S3_OUTPUT_PATH,
)


def deploy():
    session = get_session()
    role_arn = ensure_role(session)
    sm_session = sagemaker.Session(boto_session=session)
    sm_client = session.client("sagemaker")

    # ── Find the latest training job ──────────────────────────
    jobs = sm_client.list_training_jobs(
        NameContains="loan-xgboost",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    if not jobs["TrainingJobSummaries"]:
        print("✗ No training jobs found. Run train.py first!")
        return

    job_name = jobs["TrainingJobSummaries"][0]["TrainingJobName"]
    job_info = sm_client.describe_training_job(TrainingJobName=job_name)
    model_data = job_info["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"Using model from job: {job_name}")
    print(f"Model artifact: {model_data}")

    container = image_uris.retrieve("xgboost", REGION, version="1.7-1")

    # ── Delete old resources if they exist ────────────────────
    for name, delete_fn in [
        (ENDPOINT_NAME, lambda: sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)),
        (ENDPOINT_CONFIG_NAME, lambda: sm_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)),
        (MODEL_NAME, lambda: sm_client.delete_model(ModelName=MODEL_NAME)),
    ]:
        try:
            delete_fn()
            print(f"  Deleted old: {name}")
        except Exception:
            pass

    # ── Create Model ──────────────────────────────────────────
    sm_client.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            "Image": container,
            "ModelDataUrl": model_data,
        },
        ExecutionRoleArn=role_arn,
    )
    print(f"✓ Model created: {MODEL_NAME}")

    # ── Create Endpoint Config ────────────────────────────────
    sm_client.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": MODEL_NAME,
            "InstanceType": INSTANCE_TYPE_DEPLOY,
            "InitialInstanceCount": 1,
        }],
    )
    print(f"✓ Endpoint config created: {ENDPOINT_CONFIG_NAME}")

    # ── Create Endpoint ───────────────────────────────────────
    sm_client.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
    )
    print(f"✓ Endpoint creating: {ENDPOINT_NAME}")
    print("  This takes 5-8 minutes...")

    # Wait for endpoint to be InService
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME)
    print(f"\n✓ Endpoint is LIVE: {ENDPOINT_NAME}")
    print(f"\nNext step: python predict.py")


if __name__ == "__main__":
    deploy()
