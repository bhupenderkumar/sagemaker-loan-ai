#!/bin/bash
# ──────────────────────────────────────────────────────────────
# START — Prepare data, train model, deploy endpoint
# Run this once. Takes ~10-15 minutes total.
# ──────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"
source "$ROOT_DIR/.venv/bin/activate" 2>/dev/null || true
unset REQUESTS_CA_BUNDLE SSL_CERT_FILE 2>/dev/null || true

echo "════════════════════════════════════════════════════════"
echo "  LOAN PREDICTION — Full Pipeline"
echo "════════════════════════════════════════════════════════"

echo ""
echo "Step 1/4: Installing dependencies..."
pip3 install boto3 sagemaker pandas scikit-learn --quiet

echo ""
echo "Step 2/4: Preparing & uploading data..."
python3 prepare_data.py

echo ""
echo "Step 3/4: Training model (takes ~5 min)..."
python3 train.py

echo ""
echo "Step 4/4: Deploying endpoint (takes ~5-8 min)..."
python3 deploy.py

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ DONE! Endpoint is live."
echo ""
echo "  Test it:   python3 predict.py"
echo ""
echo "  ⚠️  IMPORTANT: When done testing, run:"
echo "       ./nuke.sh"
echo "  to delete everything and avoid charges!"
echo "════════════════════════════════════════════════════════"
