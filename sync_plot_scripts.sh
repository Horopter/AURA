#!/bin/bash
# Quick script to sync plot generation scripts to cluster

SOURCE_DIR="${SOURCE_DIR:-/Users/santoshdesai/Downloads/fvc/}"
DEST_HOST="${DEST_HOST:-santoshd@greatlakes.arc-ts.umich.edu}"
DEST_PATH="${DEST_PATH:-/scratch/si670f25_class_root/si670f25_class/santoshd/fvc/}"

echo "Syncing plot generation scripts to cluster..."
echo ""

rsync -avh --progress \
  --relative \
  "$SOURCE_DIR/./generate_plots_from_trained_models.py" \
  "$SOURCE_DIR/./generate_plots.sh" \
  "$SOURCE_DIR/./check_metrics.sh" \
  "$DEST_HOST:$DEST_PATH"

echo ""
echo "✓ Done! Scripts synced to cluster."
echo ""
echo "On cluster, you can now run:"
echo "  ./generate_plots.sh --model-type logistic_regression"

