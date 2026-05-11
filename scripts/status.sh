#!/bin/bash
# Quick status check — see if anything is running and costing money
set -e

echo "════════════════════════════════════════════════════════"
echo "  AWS COST CHECK — Loan Prediction Resources"
echo "════════════════════════════════════════════════════════"

PROFILE="personal"
REGION="ap-south-1"

echo ""
echo "── SageMaker Endpoints (💰 these cost money!) ──"
aws sagemaker list-endpoints --profile $PROFILE --region $REGION \
    --query 'Endpoints[*].[EndpointName,EndpointStatus]' --output table 2>/dev/null || echo "  None"

echo ""
echo "── SageMaker Models ──"
aws sagemaker list-models --profile $PROFILE --region $REGION \
    --query 'Models[?contains(ModelName,`loan`)].[ModelName]' --output table 2>/dev/null || echo "  None"

echo ""
echo "── S3 Buckets ──"
aws s3 ls --profile $PROFILE --region $REGION 2>/dev/null | grep loan || echo "  No loan buckets"

echo ""
echo "── Training Jobs (recent) ──"
aws sagemaker list-training-jobs --profile $PROFILE --region $REGION \
    --name-contains loan --max-results 3 \
    --query 'TrainingJobSummaries[*].[TrainingJobName,TrainingJobStatus]' --output table 2>/dev/null || echo "  None"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  If you see any ACTIVE endpoints, run: ./nuke.sh"
echo "════════════════════════════════════════════════════════"
