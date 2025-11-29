"""
Stage 2: Extract Handcrafted Features

Extract M handcrafted features from all 11N videos.
Input: 11N videos (from Stage 1)
Output: M features per video
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List
import polars as pl
import numpy as np

from .handcrafted_features import extract_all_features
from .video_paths import resolve_video_path
from .mlops_utils import aggressive_gc, log_memory_stats

logger = logging.getLogger(__name__)


def stage2_extract_features(
    project_root: str,
    augmented_metadata_path: str,
    num_frames: int = 8,
    output_dir: str = "data/features_stage2"
) -> pl.DataFrame:
    """
    Stage 2: Extract handcrafted features from all augmented videos.
    
    Args:
        project_root: Project root directory
        augmented_metadata_path: Path to Stage 1 metadata CSV
        num_frames: Number of frames to sample for feature extraction
        output_dir: Directory to save feature files
    
    Returns:
        DataFrame with video paths and extracted features
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load augmented video metadata
    logger.info("Stage 2: Loading augmented video metadata...")
    metadata_df = pl.read_csv(augmented_metadata_path)
    
    logger.info(f"Stage 2: Found {metadata_df.height} videos to process")
    logger.info(f"Stage 2: Extracting features from {num_frames} frames per video")
    logger.info(f"Stage 2: Output directory: {output_dir}")
    
    feature_rows = []
    
    # Process each video one at a time
    for idx in range(metadata_df.height):
        row = metadata_df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        aug_idx = row.get("augmentation_idx", -1)
        is_original = row.get("is_original", False)
        
        try:
            video_path = resolve_video_path(video_rel, project_root)
            
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Stage 2: Processing video {idx + 1}/{metadata_df.height}: {Path(video_path).name}")
            logger.info(f"{'='*80}")
            
            log_memory_stats(f"Stage 2: before video {idx + 1}")
            
            # Extract features
            features = extract_all_features(
                video_path,
                num_frames=num_frames,
                project_root=str(project_root)
            )
            
            if features is None or len(features) == 0:
                logger.warning(f"No features extracted from {video_path}")
                continue
            
            # Save features to file
            video_id = Path(video_path).stem
            video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
            feature_file = output_dir / f"{video_id}_features.npy"
            np.save(str(feature_file), features)
            
            # Create metadata row
            feature_rows.append({
                "video_path": video_rel,
                "label": label,
                "augmentation_idx": aug_idx,
                "is_original": is_original,
                "feature_file": str(feature_file.relative_to(project_root)),
                "num_features": len(features),
            })
            
            logger.info(f"✓ Extracted {len(features)} features from {video_path}")
            
            # Clear memory
            aggressive_gc(clear_cuda=False)
            
            log_memory_stats(f"Stage 2: after video {idx + 1}")
            
        except Exception as e:
            logger.error(f"Failed to extract features from {video_rel}: {e}", exc_info=True)
            continue
    
    # Save feature metadata
    if feature_rows:
        feature_df = pl.DataFrame(feature_rows)
        metadata_path = output_dir / "features_metadata.csv"
        feature_df.write_csv(str(metadata_path))
        logger.info(f"\n✓ Stage 2 complete: Saved feature metadata to {metadata_path}")
        logger.info(f"✓ Stage 2: Extracted features from {len(feature_rows)} videos")
        
        # Determine M (number of features)
        if feature_rows:
            M = feature_rows[0]["num_features"]
            logger.info(f"✓ Stage 2: M = {M} features per video")
        
        return feature_df
    else:
        logger.error("Stage 2: No features extracted!")
        return pl.DataFrame()

