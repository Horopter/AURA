#!/bin/bash
# Wrapper script to run cleanup on remote cluster
# This script SSHes to the cluster and runs the cleanup script

DEST_HOST="${DEST_HOST:-santoshd@greatlakes.arc-ts.umich.edu}"
REMOTE_PATH="${REMOTE_PATH:-/scratch/si670f25_class_root/si670f25_class/santoshd/fvc}"
DRY_RUN="${DRY_RUN:-true}"

echo "=========================================="
echo "Remote Cleanup Wrapper"
echo "=========================================="
echo "Host: $DEST_HOST"
echo "Path: $REMOTE_PATH"
echo "Dry run: $DRY_RUN"
echo ""
echo "This will clean up sync artifacts on the remote cluster"
echo "while preserving data/ and archive/ folders."
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "⚠ DRY RUN MODE - No files will be deleted"
    echo ""
    read -p "Continue with dry run? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! $REPLY == "" ]]; then
        echo "Cancelled."
        exit 0
    fi
else
    echo "⚠ WARNING: This will DELETE files on the remote cluster!"
    echo "  data/ and archive/ folders will be preserved"
    read -p "Are you sure? (yes/N): " -r
    if [[ ! $REPLY == "yes" ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

echo ""
echo "Running cleanup on remote cluster..."
echo ""

# Read the cleanup script and execute it remotely
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cat "$SCRIPT_DIR/cleanup_remote_sync.sh" | ssh "$DEST_HOST" "REMOTE_PATH=\"$REMOTE_PATH\" DRY_RUN=\"$DRY_RUN\" bash"

echo ""
echo "Done!"

