#!/bin/bash
# Minimal rsync script to sync only files needed for Stage 5 training
# This syncs code, SLURM scripts, and data files required for training

# Update these paths if needed
SOURCE_DIR="${SOURCE_DIR:-/Users/santoshdesai/Downloads/fvc/}"
DEST_HOST="${DEST_HOST:-santoshd@greatlakes.arc-ts.umich.edu}"
DEST_PATH="${DEST_PATH:-/scratch/si670f25_class_root/si670f25_class/santoshd/fvc/}"

# Create temporary directory for SSH control socket
SSH_CONTROL_DIR="$HOME/.ssh/controlmasters"
mkdir -p "$SSH_CONTROL_DIR"

# Clean up any stale sockets from previous runs (older than 5 minutes)
find "$SSH_CONTROL_DIR" -name "greatlakes_control_*" -type s -mmin +5 -delete 2>/dev/null || true

# Use a unique socket name with PID and timestamp to avoid conflicts
SSH_CONTROL_SOCKET="$SSH_CONTROL_DIR/greatlakes_control_$$_$(date +%s)"

# Cleanup function to close SSH connection
cleanup() {
    if [ -S "$SSH_CONTROL_SOCKET" ] || [ -e "$SSH_CONTROL_SOCKET" ]; then
        ssh -S "$SSH_CONTROL_SOCKET" -O exit "$DEST_HOST" 2>/dev/null || true
        rm -f "$SSH_CONTROL_SOCKET" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Ensure socket doesn't exist before we start
rm -f "$SSH_CONTROL_SOCKET" 2>/dev/null || true

echo "Syncing files needed for Stage 5 training to Great Lakes cluster..."
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_HOST:$DEST_PATH"
echo ""
echo "This will sync:"
echo "  ✓ Code: lib/, src/"
echo "  ✓ SLURM scripts: scripts/slurm_jobs/"
echo "  ✓ Config: requirements.txt"
echo ""
echo "This will NOT sync:"
echo "  ✗ data/ (excluded completely - should exist on cluster)"
echo "  ✗ archive/ (excluded completely)"
echo "  ✗ venv/ (should exist on cluster)"
echo "  ✗ logs/ (will be generated)"
echo "  ✗ mlruns/ (MLflow runs)"
echo "  ✗ __pycache__/ (Python cache)"
echo "  ✗ videos/ (original videos)"
echo ""

# Use SSH connection sharing for all operations
SSH_OPTS=(-o ControlMaster=yes -o ControlPath="$SSH_CONTROL_SOCKET" -o ControlPersist=60)

# Sync code directories (maintain structure by using --relative or syncing from parent)
echo "Syncing code (lib/, src/)..."
rsync -avh --progress --exclude='__pycache__' --exclude='*.pyc' -e "ssh ${SSH_OPTS[*]}" \
  --relative \
  "$SOURCE_DIR/./lib" \
  "$SOURCE_DIR/./src" \
  "$DEST_HOST:$DEST_PATH"

# Sync SLURM scripts
echo ""
echo "Syncing SLURM scripts..."
rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
  --relative \
  "$SOURCE_DIR/./scripts" \
  "$DEST_HOST:$DEST_PATH"

# Sync requirements.txt
echo ""
echo "Syncing requirements.txt..."
rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
  --relative \
  "$SOURCE_DIR/./requirements.txt" \
  "$DEST_HOST:$DEST_PATH"

# Sync plot generation scripts
echo ""
echo "Syncing plot generation scripts..."
rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
  --relative \
  "$SOURCE_DIR/./generate_plots_from_trained_models.py" \
  "$SOURCE_DIR/./generate_plots.sh" \
  "$SOURCE_DIR/./check_metrics.sh" \
  "$DEST_HOST:$DEST_PATH"

# Note: data/ and archive/ folders are excluded completely
# Training data should already exist on the cluster from previous stages
echo ""
echo "⚠ Note: data/ and archive/ folders are excluded from sync"
echo "  Training data should already exist on the cluster from previous stages"

# Clear Python cache on remote
echo ""
echo "Clearing Python cache on cluster..."
ssh -T "${SSH_OPTS[@]}" "$DEST_HOST" << 'ENDSSH'
    DEST_PATH="/scratch/si670f25_class_root/si670f25_class/santoshd/fvc"
    
    # Clear Python cache
    find "$DEST_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$DEST_PATH" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "$DEST_PATH" -type f -name "*.pyo" -delete 2>/dev/null || true
    find "$DEST_PATH" -type f -name "*.pyd" -delete 2>/dev/null || true
    echo "✓ Python cache cleared"
ENDSSH

echo ""
echo "✓ Sync complete!"
echo ""
echo "Next steps on cluster:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Submit training jobs: sbatch scripts/slurm_jobs/slurm_stage5a.sh"
echo "  3. Check job status: squeue -u santoshd"

