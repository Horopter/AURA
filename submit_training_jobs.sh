#!/bin/bash
# Submit all Stage 5 training jobs to generate metrics
# This submits jobs for 5a-5i, 5alpha, and 5beta

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=========================================="
echo "Submitting Stage 5 Training Jobs"
echo "=========================================="
echo "This will submit jobs for:"
echo "  - 5a (logistic_regression)"
echo "  - 5b (svm)"
echo "  - 5alpha (sklearn_logreg)"
echo "  - 5beta (gradient_boosting)"
echo "  - 5c (naive_cnn)"
echo "  - 5d (pretrained_inception)"
echo "  - 5e (variable_ar_cnn)"
echo "  - 5f (xgboost_pretrained_inception)"
echo "  - 5g (xgboost_i3d)"
echo "  - 5h (xgboost_r2plus1d)"
echo "  - 5i (xgboost_vit_gru)"
echo ""
echo "All jobs will log metrics to metrics.jsonl files"
echo ""

# Array of job scripts to submit
JOBS=(
    "scripts/slurm_jobs/slurm_stage5a.sh"
    "scripts/slurm_jobs/slurm_stage5b.sh"
    "scripts/slurm_jobs/slurm_stage5alpha.sh"
    "scripts/slurm_jobs/slurm_stage5beta.sh"
    "scripts/slurm_jobs/slurm_stage5c.sh"
    "scripts/slurm_jobs/slurm_stage5d.sh"
    "scripts/slurm_jobs/slurm_stage5e.sh"
    "scripts/slurm_jobs/slurm_stage5f.sh"
    "scripts/slurm_jobs/slurm_stage5g.sh"
    "scripts/slurm_jobs/slurm_stage5h.sh"
    "scripts/slurm_jobs/slurm_stage5i.sh"
)

# Submit all jobs
JOB_IDS=()
for job_script in "${JOBS[@]}"; do
    if [ ! -f "$job_script" ]; then
        echo "⚠ WARNING: Job script not found: $job_script"
        continue
    fi
    
    echo "Submitting: $job_script"
    JOB_ID=$(sbatch "$job_script" 2>&1 | grep -oP 'Submitted batch job \K[0-9]+' || echo "")
    
    if [ -n "$JOB_ID" ]; then
        JOB_IDS+=("$JOB_ID")
        echo "  ✓ Job ID: $JOB_ID"
    else
        echo "  ✗ Failed to submit job"
    fi
    echo ""
done

echo "=========================================="
echo "Job Submission Summary"
echo "=========================================="
echo "Submitted ${#JOB_IDS[@]} jobs:"
for job_id in "${JOB_IDS[@]}"; do
    echo "  Job ID: $job_id"
done

echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To check specific job:"
echo "  squeue -j <JOB_ID>"
echo ""
echo "To view job output:"
echo "  tail -f logs/stage5/stage5a-<JOB_ID>.out"
echo ""
echo "To check if metrics were generated:"
echo "  ls -lh data/stage5/*/fold_*/metrics.jsonl"

