#!/bin/bash
# ──────────────────────────────────────────────────────────────
# START — Prepare data, train model, deploy endpoint
# Run this once. Takes ~10-15 minutes total.
# ──────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════"
echo "  LOAN PREDICTION — Full Pipeline"
echo "════════════════════════════════════════════════════════"

echo ""
echo "Step 1/4: Installing dependencies..."
pip install boto3 sagemaker pandas scikit-learn --quiet

echo ""
echo "Step 2/4: Preparing & uploading data..."
python prepare_data.py

echo ""
echo "Step 3/4: Training model (takes ~5 min)..."
python train.py

echo ""
echo "Step 4/4: Deploying endpoint (takes ~5-8 min)..."
python deploy.py

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ DONE! Endpoint is live."
echo ""
echo "  Test it:   python predict.py"
echo ""
echo "  ⚠️  IMPORTANT: When done testing, run:"
echo "       ./nuke.sh"
echo "  to delete everything and avoid charges!"
echo "════════════════════════════════════════════════════════"
