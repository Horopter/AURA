"""
Path resolution utilities.

Provides:
- Video path resolution
- Path validation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import polars as pl


def resolve_video_path(video_rel: str, project_root: str) -> str:
    """
    Resolve a relative video path to an absolute path.
    
    Tries multiple path resolution strategies in order:
    1. Add 'videos/' prefix (most common: CSV has FVC1/... but files are at videos/FVC1/...)
    2. Direct relative path from project_root
    3. Remove 'videos/' prefix if present
    
    Args:
        video_rel: Relative video path from CSV (e.g., "FVC1/youtube/.../video.mp4")
        project_root: Project root directory
        
    Returns:
        Absolute path to the video file (first existing path found, or most likely path)
    """
    if not video_rel:
        raise ValueError("video_rel cannot be empty")
    
    video_rel = str(video_rel).strip()
    project_root = Path(project_root).resolve()
    
    # Strategy 1: Add 'videos/' prefix (most common case)
    candidate1 = project_root / "videos" / video_rel
    if candidate1.exists():
        return str(candidate1)
    
    # Strategy 2: Direct relative path from project_root
    candidate2 = project_root / video_rel
    if candidate2.exists():
        return str(candidate2)
    
    # Strategy 3: Remove 'videos/' prefix if present
    if video_rel.startswith("videos/"):
        video_rel_no_prefix = video_rel[7:]  # Remove "videos/"
        candidate3 = project_root / video_rel_no_prefix
        if candidate3.exists():
            return str(candidate3)
    
    # If none exist, return the most likely path (strategy 1)
    return str(candidate1)


def get_video_path_candidates(video_rel: str, project_root: str) -> List[str]:
    """
    Get all possible candidate paths for a video.
    
    Returns:
        List of candidate absolute paths in order of likelihood
    """
    if not video_rel:
        return []
    
    video_rel = str(video_rel).strip()
    project_root = Path(project_root).resolve()
    
    candidates = []
    
    # Strategy 1: Add 'videos/' prefix
    candidates.append(str(project_root / "videos" / video_rel))
    
    # Strategy 2: Direct relative path
    candidates.append(str(project_root / video_rel))
    
    # Strategy 3: Remove 'videos/' prefix if present
    if video_rel.startswith("videos/"):
        video_rel_no_prefix = video_rel[7:]
        candidates.append(str(project_root / video_rel_no_prefix))
    
    return candidates


def check_video_path_exists(video_rel: str, project_root: str) -> bool:
    """
    Check if a video file exists at any of the candidate paths.
    
    Args:
        video_rel: Relative video path
        project_root: Project root directory
        
    Returns:
        True if video exists at any candidate path, False otherwise
    """
    candidates = get_video_path_candidates(video_rel, project_root)
    return any(os.path.exists(c) for c in candidates)


def find_metadata_file(base_path: Path, filename_base: str) -> Optional[Path]:
    """
    Find metadata file with any supported extension (.arrow, .parquet, .csv).
    
    Tries in order: .arrow, .parquet, .csv (preferring Arrow format).
    
    Args:
        base_path: Base directory path
        filename_base: Filename without extension (e.g., "augmented_metadata")
    
    Returns:
        Path to existing metadata file, or None if not found
    """
    base_path = Path(base_path)
    for ext in ['.arrow', '.parquet', '.csv']:
        candidate = base_path / f"{filename_base}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_metadata_flexible(path: str) -> Optional[pl.DataFrame]:
    """
    Load metadata from CSV, Arrow, or Parquet format.
    
    Tries formats in order: .arrow, .parquet, .csv (preferring Arrow format).
    
    Args:
        path: Path to metadata file (with or without extension)
    
    Returns:
        Polars DataFrame, or None if file doesn't exist
    """
    path_obj = Path(path)
    
    # If path doesn't exist, try with different extensions
    if not path_obj.exists():
        for ext in ['.arrow', '.parquet', '.csv']:
            candidate = path_obj.with_suffix(ext)
            if candidate.exists():
                path_obj = candidate
                break
        else:
            return None
    
    # Load based on extension
    if path_obj.suffix == '.arrow':
        return pl.read_ipc(path_obj)
    elif path_obj.suffix == '.parquet':
        return pl.read_parquet(path_obj)
    else:
        return pl.read_csv(path_obj)


__all__ = [
    "resolve_video_path",
    "get_video_path_candidates",
    "check_video_path_exists",
    "find_metadata_file",
    "load_metadata_flexible",
]
