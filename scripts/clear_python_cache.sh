#!/bin/bash
# Clear Python cache on server to ensure fresh code is loaded
# Run this on the server after syncing code

PROJECT_DIR="${1:-/scratch/si670f25_class_root/si670f25_class/santoshd/fvc}"

echo "Clearing Python cache in $PROJECT_DIR..."

# Find and remove all __pycache__ directories
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Find and remove all .pyc, .pyo, .pyd files
find "$PROJECT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_DIR" -type f -name "*.pyo" -delete 2>/dev/null || true
find "$PROJECT_DIR" -type f -name "*.pyd" -delete 2>/dev/null || true

echo "Python cache cleared!"
