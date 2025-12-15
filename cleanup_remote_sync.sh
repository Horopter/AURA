#!/bin/bash
# Safe cleanup script for remote cluster after improper sync
# This script removes common sync artifacts while preserving data/ and archive/ folders
#
# Usage on cluster:
#   bash cleanup_remote_sync.sh
#   OR
#   ssh santoshd@greatlakes.arc-ts.umich.edu 'bash -s' < cleanup_remote_sync.sh

set -euo pipefail

# Configuration
REMOTE_PATH="${REMOTE_PATH:-/scratch/si670f25_class_root/si670f25_class/santoshd/fvc}"
DRY_RUN="${DRY_RUN:-false}"

echo "=========================================="
echo "Remote Sync Cleanup Script"
echo "=========================================="
echo "Target path: $REMOTE_PATH"
echo "Dry run: $DRY_RUN"
echo ""
echo "This script will:"
echo "  ✓ Remove Python cache (__pycache__, *.pyc)"
echo "  ✓ Remove incorrectly synced files"
echo "  ✓ Clean up temporary files"
echo "  ✓ Preserve data/ folder completely"
echo "  ✓ Preserve archive/ folder completely"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "⚠ DRY RUN MODE - No files will be deleted"
    echo ""
fi

# Function to safely remove files/directories
safe_remove() {
    local target="$1"
    local description="$2"
    
    if [ ! -e "$target" ]; then
        return 0  # Doesn't exist, skip
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY RUN] Would remove: $target ($description)"
    else
        echo "Removing: $target ($description)"
        rm -rf "$target" 2>/dev/null || true
    fi
}

# Change to target directory
cd "$REMOTE_PATH" || {
    echo "✗ ERROR: Cannot access $REMOTE_PATH"
    echo "  Make sure the path is correct and you have permissions"
    exit 1
}

echo "Current directory: $(pwd)"
echo ""

# 1. Remove Python cache files
echo "Step 1: Removing Python cache files..."
find . -type d -name "__pycache__" -not -path "./data/*" -not -path "./archive/*" | while read -r dir; do
    safe_remove "$dir" "Python cache directory"
done

find . -type f -name "*.pyc" -not -path "./data/*" -not -path "./archive/*" | while read -r file; do
    safe_remove "$file" "Python bytecode"
done

find . -type f -name "*.pyo" -not -path "./data/*" -not -path "./archive/*" | while read -r file; do
    safe_remove "$file" "Python optimized bytecode"
done

# 2. Remove incorrectly synced directories (if they exist at wrong locations)
echo ""
echo "Step 2: Checking for incorrectly placed directories..."

# Check if data/ or archive/ were synced into wrong locations (e.g., lib/data/, src/data/)
for dir in lib src scripts; do
    if [ -d "$dir/data" ]; then
        echo "⚠ Found $dir/data/ - this shouldn't be here"
        safe_remove "$dir/data" "Incorrectly synced data directory"
    fi
    if [ -d "$dir/archive" ]; then
        echo "⚠ Found $dir/archive/ - this shouldn't be here"
        safe_remove "$dir/archive" "Incorrectly synced archive directory"
    fi
done

# 3. Remove duplicate files that might have been synced incorrectly
echo ""
echo "Step 3: Checking for duplicate or incorrectly synced files..."

# Remove .DS_Store files (macOS)
find . -name ".DS_Store" -not -path "./data/*" -not -path "./archive/*" | while read -r file; do
    safe_remove "$file" "macOS metadata file"
done

# Remove Thumbs.db files (Windows)
find . -name "Thumbs.db" -not -path "./data/*" -not -path "./archive/*" | while read -r file; do
    safe_remove "$file" "Windows thumbnail file"
done

# 4. Remove temporary files
echo ""
echo "Step 4: Removing temporary files..."
find . -type f -name "*.tmp" -not -path "./data/*" -not -path "./archive/*" | while read -r file; do
    safe_remove "$file" "Temporary file"
done

find . -type f -name "*.temp" -not -path "./data/*" -not -path "./archive/*" | while read -r file; do
    safe_remove "$file" "Temporary file"
done

find . -type f -name "*.bak" -not -path "./data/*" -not -path "./archive/*" | while read -r file; do
    safe_remove "$file" "Backup file"
done

# 5. Remove incorrectly synced venv if it exists at root
if [ -d "venv" ] && [ ! -L "venv" ]; then
    echo ""
    echo "⚠ Found venv/ directory at root"
    echo "  This might be from an improper sync"
    read -p "  Remove venv/ directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        safe_remove "venv" "Virtual environment (should be managed separately)"
    fi
fi

# 6. Clean up any .git directories that might have been synced
if [ -d ".git" ]; then
    echo ""
    echo "⚠ Found .git/ directory"
    echo "  Git repository should not be synced to cluster"
    read -p "  Remove .git/ directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        safe_remove ".git" "Git repository directory"
    fi
fi

# 7. Verify data/ and archive/ are preserved
echo ""
echo "Step 5: Verifying data/ and archive/ are preserved..."
if [ -d "data" ]; then
    DATA_SIZE=$(du -sh data 2>/dev/null | cut -f1)
    echo "  ✓ data/ folder exists ($DATA_SIZE)"
else
    echo "  ⚠ data/ folder not found (this is OK if it doesn't exist yet)"
fi

if [ -d "archive" ]; then
    ARCHIVE_SIZE=$(du -sh archive 2>/dev/null | cut -f1)
    echo "  ✓ archive/ folder exists ($ARCHIVE_SIZE)"
else
    echo "  ⚠ archive/ folder not found (this is OK if it doesn't exist yet)"
fi

echo ""
echo "=========================================="
if [ "$DRY_RUN" = "true" ]; then
    echo "DRY RUN COMPLETE - No files were deleted"
    echo "Run with DRY_RUN=false to actually clean up"
else
    echo "Cleanup complete!"
fi
echo "=========================================="
echo ""
echo "Remaining directory structure:"
ls -lah | grep -E "^d" | awk '{print $9}' | grep -v "^\.$" | grep -v "^\.\.$" | while read -r dir; do
    if [ -d "$dir" ]; then
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  $dir/ ($SIZE)"
    fi
done

