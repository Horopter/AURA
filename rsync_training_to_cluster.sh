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
echo "  ✓ Training data: data/scaled_videos/, data/features_stage2/, data/features_stage4/"
echo "  ✓ Config: requirements.txt"
echo ""
echo "This will NOT sync:"
echo "  ✗ venv/ (should exist on cluster)"
echo "  ✗ logs/ (will be generated)"
echo "  ✗ data/stage5/ (will be regenerated)"
echo "  ✗ mlruns/ (MLflow runs)"
echo "  ✗ __pycache__/ (Python cache)"
echo "  ✗ videos/ (original videos, only scaled videos needed)"
echo ""

# Use SSH connection sharing for all operations
SSH_OPTS=(-o ControlMaster=yes -o ControlPath="$SSH_CONTROL_SOCKET" -o ControlPersist=60)

# Sync code directories
echo "Syncing code (lib/, src/)..."
rsync -avh --progress --exclude='__pycache__' --exclude='*.pyc' -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_DIR/lib" \
  "$SOURCE_DIR/src" \
  "$DEST_HOST:$DEST_PATH"

# Sync SLURM scripts
echo ""
echo "Syncing SLURM scripts..."
rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_DIR/scripts" \
  "$DEST_HOST:$DEST_PATH"

# Sync requirements.txt
echo ""
echo "Syncing requirements.txt..."
rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_DIR/requirements.txt" \
  "$DEST_HOST:$DEST_PATH"

# Sync training data (metadata files and feature files)
echo ""
echo "Syncing training data..."
echo "  - Scaled videos metadata..."
rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
  --include='scaled_metadata.*' \
  --include='*/' \
  --exclude='*' \
  "$SOURCE_DIR/data/scaled_videos/" \
  "$DEST_HOST:$DEST_PATH/data/scaled_videos/"

echo "  - Stage 2 features..."
rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_DIR/data/features_stage2/" \
  "$DEST_HOST:$DEST_PATH/data/features_stage2/"

echo "  - Stage 4 features..."
rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
  "$SOURCE_DIR/data/features_stage4/" \
  "$DEST_PATH/data/features_stage4/"

# For video-based models, we need the actual scaled video files
# This is optional - only sync if videos directory exists and is not too large
if [ -d "$SOURCE_DIR/data/scaled_videos" ]; then
    echo ""
    echo "Checking if scaled video files need to be synced..."
    VIDEO_COUNT=$(find "$SOURCE_DIR/data/scaled_videos" -name "*.mp4" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "$VIDEO_COUNT" -gt 0 ]; then
        echo "  Found $VIDEO_COUNT video files"
        echo "  ⚠ WARNING: Video files are large. Syncing them may take a long time."
        echo "  If videos already exist on cluster, you can skip this step."
        read -p "  Sync video files? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "  Syncing scaled video files (this may take a while)..."
            rsync -avh --progress -e "ssh ${SSH_OPTS[*]}" \
              --include='*.mp4' \
              --include='*/' \
              --exclude='*' \
              "$SOURCE_DIR/data/scaled_videos/" \
              "$DEST_HOST:$DEST_PATH/data/scaled_videos/"
        else
            echo "  Skipping video file sync (assuming they already exist on cluster)"
        fi
    else
        echo "  No video files found in data/scaled_videos/"
    fi
fi

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

