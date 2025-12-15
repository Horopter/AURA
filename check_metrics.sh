#!/bin/bash
# Script to check where metrics are located and verify they exist

MODEL_TYPE="${1:-logistic_regression}"
BASE_DIR="${BASE_DIR:-data/stage5}"

echo "=========================================="
echo "Checking Metrics for: $MODEL_TYPE"
echo "=========================================="
echo ""

MODEL_DIR="$BASE_DIR/$MODEL_TYPE"

if [ ! -d "$MODEL_DIR" ]; then
    echo "✗ Model directory not found: $MODEL_DIR"
    exit 1
fi

echo "Model directory: $MODEL_DIR"
echo ""

# Check each fold
for fold in {1..5}; do
    FOLD_DIR="$MODEL_DIR/fold_$fold"
    METRICS_FILE="$FOLD_DIR/metrics.jsonl"
    
    echo "Fold $fold:"
    if [ -f "$METRICS_FILE" ]; then
        SIZE=$(ls -lh "$METRICS_FILE" | awk '{print $5}')
        LINES=$(wc -l < "$METRICS_FILE" 2>/dev/null || echo "0")
        echo "  ✓ metrics.jsonl exists ($SIZE, $LINES lines)"
        echo "  Location: $METRICS_FILE"
        
        # Show first few lines
        echo "  Sample entries:"
        head -3 "$METRICS_FILE" | while read -r line; do
            echo "    $line"
        done
    else
        echo "  ✗ metrics.jsonl NOT FOUND"
        echo "  Expected: $METRICS_FILE"
        echo "  Reason: Model was trained before metrics logging was added"
        echo "  Solution: Retrain with --delete-existing flag"
    fi
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="

# Count existing metrics files
EXISTING=$(find "$MODEL_DIR" -name "metrics.jsonl" 2>/dev/null | wc -l)
echo "Metrics files found: $EXISTING / 5"
echo ""

if [ "$EXISTING" -eq 0 ]; then
    echo "⚠ No metrics files found!"
    echo ""
    echo "To generate metrics, retrain with:"
    echo "  python src/scripts/run_stage5_training.py \\"
    echo "    --model-types $MODEL_TYPE \\"
    echo "    --delete-existing"
    echo ""
    echo "Or submit SLURM job with delete flag:"
    echo "  FVC_DELETE_EXISTING=1 sbatch scripts/slurm_jobs/slurm_stage5a.sh"
fi

