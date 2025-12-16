#!/bin/bash
# Optimized rsync script to sync FVC project code to Great Lakes cluster
# SAFETY: Only syncs lib/, src/, and scripts/ directories, never deletes files
# OPTIMIZATIONS: Single rsync call, parallel exclusions, efficient cache clearing

set -euo pipefail  # Exit on error, undefined vars, pipe failures

SOURCE_DIR="/Users/santoshdesai/Downloads/fvc"
DEST_HOST="santoshd@greatlakes.arc-ts.umich.edu"
DEST_PATH="/scratch/si670f25_class_root/si670f25_class/santoshd/fvc"

# Create temporary directory for SSH control socket
SSH_CONTROL_DIR="$HOME/.ssh/controlmasters"
mkdir -p "$SSH_CONTROL_DIR"

# Clean up any stale sockets from previous runs (older than 1 minute)
# Also try to exit any active connections gracefully
# Match both old and new socket naming patterns
find "$SSH_CONTROL_DIR" \( -name "greatlakes_control_*" -o -name "gl_*" \) -type s 2>/dev/null | while read -r socket; do
    # Check if socket is actually in use and try to exit gracefully
    if ssh -S "$socket" -O check "$DEST_HOST" 2>/dev/null; then
        # Socket is in use, try to exit gracefully
        ssh -S "$socket" -O exit "$DEST_HOST" 2>/dev/null || true
        sleep 0.1  # Brief pause to allow cleanup
    fi
    # Remove socket file if it still exists (or if it's old)
    if [ -e "$socket" ]; then
        # Check if it's old (older than 1 minute) or just remove it
        if find "$socket" -mmin +1 2>/dev/null | grep -q .; then
            rm -f "$socket" 2>/dev/null || true
        fi
    fi
done

# Use a short unique socket name (Unix sockets have ~108 char limit)
# Format: gl_<PID>_<timestamp>_<random> (kept short to avoid path length issues)
TIMESTAMP=$(date +%s)
RANDOM_SUFFIX=$(od -An -N1 -tu1 /dev/urandom 2>/dev/null | tr -d ' ' || echo $RANDOM)
# Keep total path under 80 chars to be safe (base path ~50 + socket name ~30)
SSH_CONTROL_SOCKET="$SSH_CONTROL_DIR/gl_$$_${TIMESTAMP}_${RANDOM_SUFFIX}"

# Final check: ensure socket doesn't exist before we start
if [ -e "$SSH_CONTROL_SOCKET" ] || [ -S "$SSH_CONTROL_SOCKET" ]; then
    # Try to exit if it's in use
    ssh -S "$SSH_CONTROL_SOCKET" -O exit "$DEST_HOST" 2>/dev/null || true
    # Force remove it
    rm -f "$SSH_CONTROL_SOCKET" 2>/dev/null || true
    # If it still exists, wait a moment and try again
    if [ -e "$SSH_CONTROL_SOCKET" ]; then
        sleep 0.2
        rm -f "$SSH_CONTROL_SOCKET" 2>/dev/null || true
    fi
fi

# Initialize multiplexing flag (will be set later)
USE_MULTIPLEXING=true

# Cleanup function to close SSH connection
cleanup() {
    # Only cleanup if multiplexing was actually used and socket exists
    if [ "$USE_MULTIPLEXING" = true ] && { [ -S "$SSH_CONTROL_SOCKET" ] || [ -e "$SSH_CONTROL_SOCKET" ]; }; then
        ssh -S "$SSH_CONTROL_SOCKET" -O exit "$DEST_HOST" 2>/dev/null || true
        rm -f "$SSH_CONTROL_SOCKET" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Socket cleanup is handled above

# Parse command line arguments
DRY_RUN=false
QUIET=false
for arg in "$@"; do
    case $arg in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --quiet|-q)
            QUIET=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run|-n] [--quiet|-q]"
            echo ""
            echo "Options:"
            echo "  --dry-run, -n    Show what would be synced without actually syncing"
            echo "  --quiet, -q      Suppress progress output"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

if [ "$QUIET" = false ]; then
    echo "Syncing FVC project code to Great Lakes cluster..."
    echo "Source: $SOURCE_DIR"
    echo "Destination: $DEST_HOST:$DEST_PATH"
    echo ""
    echo "✓ SAFETY: This script:"
    echo "   - Only syncs lib/, src/, and scripts/ directories"
    echo "   - Does NOT use --delete flag (won't remove any files)"
    echo "   - Will NOT touch: data/, archive/, models/, logs/, venv/, etc."
    echo "   - Only adds/updates files, never deletes"
    echo "   - Uses SSH connection sharing (single password prompt)"
    if [ "$DRY_RUN" = true ]; then
        echo "   - DRY RUN MODE: No files will be transferred"
    fi
    echo ""
fi

# Use SSH connection sharing for all operations
if [ "$USE_MULTIPLEXING" = true ]; then
    if [ "$QUIET" = false ]; then
        echo "Using SSH connection multiplexing for faster transfers..."
    fi
    SSH_OPTS=(
        -o ControlMaster=yes
        -o ControlPath="$SSH_CONTROL_SOCKET"
        -o ControlPersist=60
        -o Compression=yes  # Enable compression for faster transfer
        -o ServerAliveInterval=60
        -o ServerAliveCountMax=3
    )
else
    # Fallback: no multiplexing, but still use compression
    SSH_OPTS=(
        -o Compression=yes
        -o ServerAliveInterval=60
        -o ServerAliveCountMax=3
    )
fi

# Build rsync options
RSYNC_OPTS=(
    -avh
    --relative  # Preserve relative path structure
    --no-perms  # Don't sync permissions (faster, avoids permission issues)
    --no-owner  # Don't sync ownership (faster, avoids permission issues)
    --no-group  # Don't sync group (faster, avoids permission issues)
    --exclude='__pycache__'  # Exclude Python cache during sync
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='*.pyd'
    --exclude='.DS_Store'  # Exclude macOS metadata
    --exclude='*.swp'  # Exclude vim swap files
    --exclude='*.swo'
    --exclude='.git/'  # Exclude git directory if present
    -e "ssh ${SSH_OPTS[*]}"
)

# Add progress and dry-run flags conditionally
if [ "$QUIET" = false ]; then
    RSYNC_OPTS+=(--progress)
fi

if [ "$DRY_RUN" = true ]; then
    RSYNC_OPTS+=(--dry-run)
fi

# Sync each directory separately - ONLY CODE FILES
# SSH connection is reused automatically via ControlMaster
if [ "$QUIET" = false ]; then
    echo "Syncing CODE ONLY from lib/, src/, and scripts/ directories..."
    echo "  (Excluding: data/, archive/, cache, compiled files, etc.)"
fi

# Build rsync options - explicitly include only code files, exclude everything else
RSYNC_DIR_OPTS=(
    -avh
    --no-perms
    --no-owner
    --no-group
    # Explicitly exclude data and archive at any level
    --exclude='data/'
    --exclude='archive/'
    --exclude='**/data/'
    --exclude='**/archive/'
    # Exclude cache and compiled files
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='*.pyd'
    --exclude='*.pyi'
    # Exclude build artifacts
    --exclude='*.egg-info/'
    --exclude='dist/'
    --exclude='build/'
    --exclude='.pytest_cache/'
    # Exclude IDE and system files
    --exclude='.DS_Store'
    --exclude='*.swp'
    --exclude='*.swo'
    --exclude='.git/'
    --exclude='.idea/'
    --exclude='.vscode/'
    # Exclude logs and temporary files
    --exclude='*.log'
    --exclude='*.tmp'
    --exclude='*.bak'
    # Only include code files explicitly
    --include='*.py'
    --include='*.sh'
    --include='*.yaml'
    --include='*.yml'
    --include='*.json'
    --include='*.txt'
    --include='*.md'
    --include='*.cfg'
    --include='*.ini'
    --include='*.toml'
    --include='*.toml'
    --include='*.lock'
    # Include directories (but contents filtered by above rules)
    --include='*/'
    # Exclude everything else
    --exclude='*'
    -e "ssh ${SSH_OPTS[*]}"
)

# Add progress and dry-run flags conditionally
if [ "$QUIET" = false ]; then
    RSYNC_DIR_OPTS+=(--progress)
fi

if [ "$DRY_RUN" = true ]; then
    RSYNC_DIR_OPTS+=(--dry-run)
fi

# Sync lib/ directory (CODE ONLY)
if [ "$QUIET" = false ]; then
    echo "  → lib/ (code files only)"
fi
rsync "${RSYNC_DIR_OPTS[@]}" \
    "$SOURCE_DIR/lib/" \
    "$DEST_HOST:$DEST_PATH/lib/"

# Sync src/ directory (CODE ONLY)
if [ "$QUIET" = false ]; then
    echo "  → src/ (code files only)"
fi
rsync "${RSYNC_DIR_OPTS[@]}" \
    "$SOURCE_DIR/src/" \
    "$DEST_HOST:$DEST_PATH/src/"

# Sync scripts/ directory (CODE ONLY)
if [ "$QUIET" = false ]; then
    echo "  → scripts/ (code files only)"
fi
rsync "${RSYNC_DIR_OPTS[@]}" \
    "$SOURCE_DIR/scripts/" \
    "$DEST_HOST:$DEST_PATH/scripts/"

# Clear Python cache explicitly in synced directories (reuses SSH connection)
if [ "$DRY_RUN" = false ]; then
    if [ "$QUIET" = false ]; then
        echo ""
        echo "Clearing Python cache in lib/, src/, and scripts/..."
    fi
    
    # Clear Python cache in each synced directory explicitly
    ssh -T "${SSH_OPTS[@]}" "$DEST_HOST" << ENDSSH
        DEST_PATH="$DEST_PATH"
        
        # Clear Python cache in lib/
        find "\$DEST_PATH/lib" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "\$DEST_PATH/lib" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete 2>/dev/null || true
        
        # Clear Python cache in src/
        find "\$DEST_PATH/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "\$DEST_PATH/src" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete 2>/dev/null || true
        
        # Clear Python cache in scripts/
        find "\$DEST_PATH/scripts" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "\$DEST_PATH/scripts" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete 2>/dev/null || true
        
        if [ "$QUIET" = false ]; then
            echo "✓ Python cache cleared in lib/, src/, and scripts/"
        fi
ENDSSH
fi

if [ "$QUIET" = false ]; then
    echo ""
    if [ "$DRY_RUN" = true ]; then
        echo "Dry run complete! Use without --dry-run to actually sync."
    else
        echo "✓ Sync complete!"
    fi
fi
