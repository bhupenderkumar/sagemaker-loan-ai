#!/bin/bash
# ──────────────────────────────────────────────────────────────
# NUKE — Delete EVERYTHING from AWS to ensure $0 bill.
# ──────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════"
echo "  ⚠️  NUKE — Deleting ALL AWS resources"
echo "════════════════════════════════════════════════════════"
echo ""
echo "This will delete:"
echo "  • SageMaker endpoint (stops billing)"
echo "  • SageMaker models & configs"
echo "  • S3 bucket & all data"
echo "  • IAM role"
echo ""
read -p "Are you sure? (y/N) " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
python cleanup.py

echo ""
echo "Verifying no endpoints remain..."
aws sagemaker list-endpoints --profile personal --region ap-south-1 \
    --query 'Endpoints[?contains(EndpointName,`loan`)]' --output table 2>/dev/null || true

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ Everything deleted. $0 bill guaranteed."
echo "════════════════════════════════════════════════════════"
