"""
Stage 4: Extract Additional Handcrafted Features from Downscaled Videos

Extract P additional handcrafted features from downscaled videos.
These should be new features that can be detected on downscaled videos.
Input: 11N downscaled videos (from Stage 3)
Output: P additional features per video
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict
import polars as pl
import numpy as np
import cv2

from .handcrafted_features import (
    extract_noise_residual,
    extract_dct_statistics,
    extract_blur_sharpness,
    extract_boundary_inconsistency
)
from ..video_paths import resolve_video_path
from ..video_modeling import _read_video_wrapper, uniform_sample_indices
from ..utils.mlops_utils import aggressive_gc, log_memory_stats

logger = logging.getLogger(__name__)


def extract_downscaled_specific_features(frame: np.ndarray) -> Dict[str, float]:
    """
    Extract features specific to downscaled videos.
    
    These are features that may be more detectable or different in downscaled videos:
    - Compression artifacts (more visible after downscaling)
    - Block artifacts (from downscaling process)
    - Frequency domain features (DCT on downscaled frames)
    - Edge preservation metrics
    - Texture uniformity
    
    Args:
        frame: Downscaled frame (H, W, C) in uint8 format
    
    Returns:
        Dictionary of feature values
    """
    features = {}
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    gray_uint8 = gray.astype(np.uint8)
    
    # 1. Block artifact strength (from downscaling/compression)
    # Check for 8x8 block patterns (common in JPEG/MPEG compression)
    block_size = 8
    h, w = gray.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    
    if h_blocks > 0 and w_blocks > 0:
        block_variances = []
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_variances.append(np.var(block))
        
        features['block_variance_mean'] = np.mean(block_variances) if block_variances else 0.0
        features['block_variance_std'] = np.std(block_variances) if block_variances else 0.0
        features['block_variance_max'] = np.max(block_variances) if block_variances else 0.0
        features['block_variance_min'] = np.min(block_variances) if block_variances else 0.0
    else:
        features['block_variance_mean'] = 0.0
        features['block_variance_std'] = 0.0
        features['block_variance_max'] = 0.0
        features['block_variance_min'] = 0.0
    
    # 2. Edge preservation after downscaling
    # Compare edge strength in original vs smoothed version
    edges = cv2.Canny(gray_uint8, 50, 150)
    blurred = cv2.GaussianBlur(gray_uint8, (5, 5), 1.0)
    edges_blurred = cv2.Canny(blurred, 50, 150)
    
    edge_preservation = np.sum(edges > 0) / (np.sum(edges_blurred > 0) + 1e-6)
    features['edge_preservation'] = edge_preservation
    
    # 3. Texture uniformity (downscaled videos may have more uniform textures)
    # Use gradient variance as texture measure (no skimage dependency)
    grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    features['texture_uniformity'] = np.var(grad_mag)
    
    # 4. Frequency domain features (on downscaled frame)
    dct_stats = extract_dct_statistics(frame, block_size=8)
    features['dct_downscaled_dc_mean'] = dct_stats.get('dct_dc_mean', 0.0)
    features['dct_downscaled_ac_energy'] = dct_stats.get('dct_ac_energy', 0.0)
    features['dct_downscaled_low_freq_ratio'] = dct_stats.get('dct_low_freq_ratio', 0.0)
    
    # 5. Compression artifact visibility
    # Check for ringing artifacts (common in downscaled/compressed videos)
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    ringing_strength = np.std(laplacian)
    features['ringing_artifact_strength'] = ringing_strength
    
    # 6. Color consistency (downscaling may affect color distribution)
    if len(frame.shape) == 3:
        color_std = np.std(frame, axis=(0, 1))
        features['color_std_r'] = float(color_std[0])
        features['color_std_g'] = float(color_std[1])
        features['color_std_b'] = float(color_std[2])
    else:
        features['color_std_r'] = 0.0
        features['color_std_g'] = 0.0
        features['color_std_b'] = 0.0
    
    return features


def extract_all_downscaled_features(
    video_path: str,
    num_frames: int = 8,
    project_root: str = None
) -> np.ndarray:
    """
    Extract all downscaled-specific features from a video.
    
    Args:
        video_path: Path to downscaled video file
        num_frames: Number of frames to sample
        project_root: Project root for path resolution
    
    Returns:
        Feature vector as numpy array
    """
    from .video_paths import resolve_video_path
    
    # Resolve video path
    if project_root:
        video_path = resolve_video_path(video_path, project_root)
    
    # Read video
    try:
        video = _read_video_wrapper(video_path)
        if video.shape[0] == 0:
            logger.warning(f"Video has no frames: {video_path}")
            return np.zeros(20)  # Return zero vector if video is empty
    except Exception as e:
        logger.warning(f"Failed to read video {video_path}: {e}")
        return np.zeros(20)
    
    total_frames = video.shape[0]
    
    # Sample frames uniformly
    indices = uniform_sample_indices(total_frames, num_frames)
    
    # Extract features from each frame
    all_features = []
    
    for idx in indices:
        frame = video[idx].numpy()  # (H, W, C) in uint8
        
        # Extract downscaled-specific features
        frame_features = extract_downscaled_specific_features(frame)
        all_features.append(frame_features)
    
    # Aggregate features across frames (mean and std)
    if not all_features:
        return np.zeros(20)
    
    # Convert to array and compute statistics
    feature_names = sorted(all_features[0].keys())
    feature_matrix = np.array([[f[name] for name in feature_names] for f in all_features])
    
    # Compute mean and std across frames
    mean_features = np.mean(feature_matrix, axis=0)
    std_features = np.std(feature_matrix, axis=0)
    
    # Concatenate mean and std
    combined_features = np.concatenate([mean_features, std_features])
    
    return combined_features


def stage4_extract_downscaled_features(
    project_root: str,
    downscaled_metadata_path: str,
    num_frames: int = 8,
    output_dir: str = "data/features_stage4"
) -> pl.DataFrame:
    """
    Stage 4: Extract additional handcrafted features from downscaled videos.
    
    Args:
        project_root: Project root directory
        downscaled_metadata_path: Path to Stage 3 metadata CSV
        num_frames: Number of frames to sample for feature extraction
        output_dir: Directory to save feature files
    
    Returns:
        DataFrame with video paths and extracted features
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load downscaled video metadata
    logger.info("Stage 4: Loading downscaled video metadata...")
    metadata_df = pl.read_csv(downscaled_metadata_path)
    
    logger.info(f"Stage 4: Found {metadata_df.height} downscaled videos to process")
    logger.info(f"Stage 4: Extracting features from {num_frames} frames per video")
    logger.info(f"Stage 4: Output directory: {output_dir}")
    
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
            logger.info(f"Stage 4: Processing video {idx + 1}/{metadata_df.height}: {Path(video_path).name}")
            logger.info(f"{'='*80}")
            
            log_memory_stats(f"Stage 4: before video {idx + 1}")
            
            # Extract features
            features = extract_all_downscaled_features(
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
            feature_file = output_dir / f"{video_id}_features_downscaled.npy"
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
            
            logger.info(f"✓ Extracted {len(features)} downscaled features from {video_path}")
            
            # Clear memory
            del features
            aggressive_gc(clear_cuda=False)
            
            log_memory_stats(f"Stage 4: after video {idx + 1}")
            
        except Exception as e:
            logger.error(f"Failed to extract features from {video_rel}: {e}", exc_info=True)
            continue
    
    # Save feature metadata
    if feature_rows:
        feature_df = pl.DataFrame(feature_rows)
        metadata_path = output_dir / "features_downscaled_metadata.csv"
        feature_df.write_csv(str(metadata_path))
        logger.info(f"\n✓ Stage 4 complete: Saved feature metadata to {metadata_path}")
        logger.info(f"✓ Stage 4: Extracted features from {len(feature_rows)} videos")
        
        # Determine P (number of features)
        if feature_rows:
            P = feature_rows[0]["num_features"]
            logger.info(f"✓ Stage 4: P = {P} features per video")
        
        return feature_df
    else:
        logger.error("Stage 4: No features extracted!")
        return pl.DataFrame()

