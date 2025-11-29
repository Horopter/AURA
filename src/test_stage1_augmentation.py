#!/usr/bin/env python3
"""
Test Stage 1 Augmentation on 10 videos from FVC1 (5 real, 5 fake)

Creates a test subset and runs augmentation to verify the pipeline works.
"""

from __future__ import annotations

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
from lib.data import load_metadata, filter_existing_videos
from lib.augmentation import stage1_augment_videos

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_test_subset(project_root: Path, num_per_class: int = 5) -> Path:
    """
    Create a test subset CSV with num_per_class videos from each class (real/fake) from FVC1.
    
    Args:
        project_root: Project root directory
        num_per_class: Number of videos per class (default: 5)
    
    Returns:
        Path to the test CSV file
    """
    logger.info("=" * 80)
    logger.info("Creating test subset: %d real + %d fake videos from FVC1", num_per_class, num_per_class)
    logger.info("=" * 80)
    
    # Load full metadata
    full_csv = project_root / "data" / "video_index_input.csv"
    if not full_csv.exists():
        logger.error("Full metadata CSV not found: %s", full_csv)
        logger.error("Please run: python src/setup_fvc_dataset.py")
        sys.exit(1)
    
    logger.info("Loading full metadata from: %s", full_csv)
    df = load_metadata(str(full_csv))
    logger.info("Total videos in metadata: %d", df.height)
    
    # Filter to only FVC1 videos (check if video_path contains FVC1)
    logger.info("Filtering to FVC1 videos...")
    fvc1_df = df.filter(pl.col("video_path").str.contains("FVC1"))
    logger.info("FVC1 videos found: %d", fvc1_df.height)
    
    if fvc1_df.height == 0:
        logger.error("No FVC1 videos found in metadata!")
        logger.error("Please ensure FVC1 dataset is set up correctly")
        sys.exit(1)
    
    # Filter existing videos
    logger.info("Filtering to existing videos...")
    fvc1_df = filter_existing_videos(fvc1_df, str(project_root))
    logger.info("Existing FVC1 videos: %d", fvc1_df.height)
    
    if fvc1_df.height == 0:
        logger.error("No existing FVC1 videos found!")
        sys.exit(1)
    
    # Check label column
    if "label" not in fvc1_df.columns:
        logger.error("'label' column not found in metadata!")
        logger.error("Available columns: %s", fvc1_df.columns)
        sys.exit(1)
    
    # Get unique labels
    unique_labels = fvc1_df["label"].unique().to_list()
    logger.info("Unique labels found: %s", unique_labels)
    
    # Select num_per_class from each label
    subsets = []
    for label in unique_labels:
        label_df = fvc1_df.filter(pl.col("label") == label)
        logger.info("Label '%s': %d videos available", label, label_df.height)
        
        if label_df.height == 0:
            logger.warning("No videos found for label '%s', skipping", label)
            continue
        
        # Sample up to num_per_class videos
        sample_size = min(num_per_class, label_df.height)
        sampled = label_df.sample(n=sample_size, seed=42)
        subsets.append(sampled)
        logger.info("Selected %d videos for label '%s'", sample_size, label)
    
    if not subsets:
        logger.error("No videos selected for any label!")
        sys.exit(1)
    
    # Combine subsets
    test_df = pl.concat(subsets)
    logger.info("Test subset created: %d videos total", test_df.height)
    
    # Show label distribution
    label_counts = test_df["label"].value_counts().sort("label")
    logger.info("Label distribution:")
    for row in label_counts.iter_rows(named=True):
        logger.info("  %s: %d", row["label"], row["count"])
    
    # Create temporary test CSV
    test_csv = project_root / "data" / "video_index_input_test.csv"
    test_df.write_csv(test_csv)
    logger.info("Test CSV saved to: %s", test_csv)
    
    return test_csv


def main():
    """Run Stage 1 augmentation test on 10 videos."""
    project_root = Path(__file__).parent.parent
    
    logger.info("=" * 80)
    logger.info("STAGE 1 AUGMENTATION TEST")
    logger.info("=" * 80)
    logger.info("Project root: %s", project_root)
    logger.info("Test: 10 videos (5 real, 5 fake) from FVC1")
    logger.info("=" * 80)
    
    # Create test subset
    test_csv = create_test_subset(project_root, num_per_class=5)
    
    # Temporarily replace the main CSV with test CSV
    main_csv = project_root / "data" / "video_index_input.csv"
    backup_csv = project_root / "data" / "video_index_input_backup.csv"
    
    # Backup original if it exists
    if main_csv.exists():
        logger.info("Backing up original CSV to: %s", backup_csv)
        import shutil
        shutil.copy2(main_csv, backup_csv)
    
    # Copy test CSV to main location
    logger.info("Using test CSV for augmentation...")
    import shutil
    shutil.copy2(test_csv, main_csv)
    
    try:
        # Run Stage 1 augmentation with reduced augmentations for testing
        num_augmentations = 3  # Reduced for testing
        logger.info("=" * 80)
        logger.info("Running Stage 1 augmentation with %d augmentations per video", num_augmentations)
        logger.info("=" * 80)
        
        result_df = stage1_augment_videos(
            project_root=str(project_root),
            num_augmentations=num_augmentations,
            output_dir="data/augmented_videos_test"
        )
        
        logger.info("=" * 80)
        logger.info("STAGE 1 TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        if result_df is not None and hasattr(result_df, 'height'):
            logger.info("Total videos processed: %d", result_df.height)
            
            # Show statistics
            if "is_original" in result_df.columns:
                original_count = result_df.filter(pl.col("is_original") == True).height
                augmented_count = result_df.filter(pl.col("is_original") == False).height
                logger.info("Original videos: %d", original_count)
                logger.info("Augmented videos: %d", augmented_count)
                logger.info("Total videos (original + augmented): %d", result_df.height)
            
            # Show label distribution
            if "label" in result_df.columns:
                label_counts = result_df["label"].value_counts().sort("label")
                logger.info("Label distribution in output:")
                for row in label_counts.iter_rows(named=True):
                    logger.info("  %s: %d", row["label"], row["count"])
        
        logger.info("=" * 80)
        logger.info("Test output saved to: %s", project_root / "data" / "augmented_videos_test")
        logger.info("=" * 80)
        
    finally:
        # Restore original CSV
        if backup_csv.exists():
            logger.info("Restoring original CSV...")
            shutil.copy2(backup_csv, main_csv)
            backup_csv.unlink()
            logger.info("Original CSV restored")
        else:
            # If no backup, just remove the test CSV copy
            if main_csv.exists() and main_csv.read_text().startswith(test_csv.read_text()[:100]):
                logger.warning("No backup found, keeping test CSV in place")
        
        logger.info("Test CSV remains at: %s", test_csv)
        logger.info("You can manually delete it when done testing")


if __name__ == "__main__":
    main()

