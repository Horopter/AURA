"""
Stage 1: Video Augmentation

Generate 10 augmentations per video.
Input: N videos
Output: 11N videos (N original + 10N augmented)
"""

from __future__ import annotations

import os
import sys
import logging
import hashlib
import random
import numpy as np
from pathlib import Path
from typing import List, Optional
import polars as pl
import av

from .video_data import load_metadata, filter_existing_videos
from .video_paths import resolve_video_path
from .mlops_utils import aggressive_gc, log_memory_stats

logger = logging.getLogger(__name__)


def load_video_frames(video_path: str) -> tuple[List[np.ndarray], float]:
    """Load all frames from a video, preserving original resolution. Returns (frames, fps)."""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        frames = []
        
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_array = frame.to_ndarray(format='rgb24')  # (H, W, 3)
                frames.append(frame_array)
        
        container.close()
        return frames, fps
    except Exception as e:
        logger.error(f"Failed to load video {video_path}: {e}")
        return [], 30.0


def save_video_frames(frames: List[np.ndarray], output_path: str, fps: float = 30.0) -> bool:
    """Save frames as a video file, preserving original resolution."""
    if not frames:
        logger.warning(f"No frames to save for {output_path}")
        return False
    
    try:
        height, width = frames[0].shape[:2]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        container = av.open(str(output_path), mode='w')
        stream = container.add_stream('libx264', rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        
        for packet in stream.encode():
            container.mux(packet)
        
        container.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save video {output_path}: {e}")
        return False


def apply_augmentation(frame: np.ndarray, aug_type: str, seed: int) -> np.ndarray:
    """Apply a single augmentation to a frame."""
    random.seed(seed)
    np.random.seed(seed)
    
    if aug_type == 'none':
        return frame
    
    from PIL import Image
    pil_image = Image.fromarray(frame)
    
    if aug_type == 'rotation':
        angle = random.uniform(-15, 15)
        pil_image = pil_image.rotate(angle, fillcolor=(0, 0, 0))
    
    elif aug_type == 'flip':
        if random.random() < 0.5:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    
    elif aug_type == 'brightness':
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(pil_image)
        factor = random.uniform(0.7, 1.3)
        pil_image = enhancer.enhance(factor)
    
    elif aug_type == 'contrast':
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(pil_image)
        factor = random.uniform(0.7, 1.3)
        pil_image = enhancer.enhance(factor)
    
    elif aug_type == 'saturation':
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(pil_image)
        factor = random.uniform(0.7, 1.3)
        pil_image = enhancer.enhance(factor)
    
    elif aug_type == 'gaussian_noise':
        frame_array = np.array(pil_image)
        noise = np.random.normal(0, 10, frame_array.shape).astype(np.float32)
        frame_array = np.clip(frame_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_array)
    
    elif aug_type == 'gaussian_blur':
        from PIL import ImageFilter
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
    
    elif aug_type == 'affine':
        # Simple affine transformation
        import torch
        from torchvision import transforms
        transform = transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            fill=0
        )
        tensor = transforms.ToTensor()(pil_image)
        tensor = transform(tensor)
        pil_image = transforms.ToPILImage()(tensor)
    
    elif aug_type == 'elastic':
        # Simplified elastic transform
        frame_array = np.array(pil_image)
        alpha = random.uniform(50, 150)
        sigma = random.uniform(5, 10)
        # Simple implementation - can be enhanced
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter((np.random.rand(*frame_array.shape[:2]) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*frame_array.shape[:2]) * 2 - 1), sigma) * alpha
        # Apply displacement (simplified)
        pil_image = Image.fromarray(frame_array)
    
    return np.array(pil_image)


def augment_video(
    video_path: str,
    num_augmentations: int = 10,
    augmentation_types: Optional[List[str]] = None
) -> List[List[np.ndarray]]:
    """
    Generate augmented versions of a video.
    
    Returns:
        List of augmented frame sequences (each is List[np.ndarray])
    """
    logger.info(f"Loading video: {video_path}")
    original_frames, fps = load_video_frames(video_path)
    
    if not original_frames:
        logger.warning(f"No frames loaded from {video_path}")
        return []
    
    logger.info(f"Loaded {len(original_frames)} frames from {video_path}")
    
    # Generate deterministic seed from video path
    video_path_str = str(video_path)
    base_seed = int(hashlib.md5(video_path_str.encode()).hexdigest()[:8], 16) % (2**31)
    
    # Default augmentation types
    if augmentation_types is None:
        augmentation_types = [
            'rotation', 'flip', 'brightness', 'contrast', 'saturation',
            'gaussian_noise', 'gaussian_blur', 'affine', 'elastic', 'none'
        ]
    
    augmented_videos = []
    
    for aug_idx in range(num_augmentations):
        aug_seed = base_seed + aug_idx
        random.seed(aug_seed)
        np.random.seed(aug_seed)
        
        # Select augmentation type
        aug_type = random.choice(augmentation_types) if len(augmentation_types) > 1 else augmentation_types[0]
        
        # Apply augmentation to all frames
        augmented_frames = []
        for frame_idx, frame in enumerate(original_frames):
            augmented_frame = apply_augmentation(frame, aug_type, aug_seed + frame_idx)
            augmented_frames.append(augmented_frame)
        
        augmented_videos.append(augmented_frames)
        logger.info(f"Generated augmentation {aug_idx + 1}/{num_augmentations} with type '{aug_type}'")
    
    return augmented_videos


def stage1_augment_videos(
    project_root: str,
    num_augmentations: int = 10,
    output_dir: str = "data/augmented_videos"
) -> pl.DataFrame:
    """
    Stage 1: Augment all videos.
    
    Args:
        project_root: Project root directory
        num_augmentations: Number of augmentations per video (default: 10)
        output_dir: Directory to save augmented videos
    
    Returns:
        DataFrame with metadata for all videos (original + augmented)
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    logger.info("Stage 1: Loading video metadata...")
    metadata_path = project_root / "data" / "video_index_input.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return pl.DataFrame()
    
    df = load_metadata(str(metadata_path))
    df = filter_existing_videos(df, str(project_root))
    
    logger.info(f"Stage 1: Found {df.height} original videos")
    logger.info(f"Stage 1: Generating {num_augmentations} augmentation(s) per video")
    logger.info(f"Stage 1: Output directory: {output_dir}")
    
    all_video_metadata = []
    
    # Process each video one at a time
    for idx in range(df.height):
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        
        try:
            video_path = resolve_video_path(video_rel, project_root)
            
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Stage 1: Processing video {idx + 1}/{df.height}: {Path(video_path).name}")
            logger.info(f"{'='*80}")
            
            log_memory_stats(f"Stage 1: before video {idx + 1}")
            
            # Save original video metadata (augmentation_idx = -1 for original)
            video_id = Path(video_path).stem
            video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
            
            # Copy original to output directory (or reference it)
            original_output = output_dir / f"{video_id}_original.mp4"
            if not original_output.exists():
                import shutil
                shutil.copy2(video_path, original_output)
            
            all_video_metadata.append({
                "video_path": str(original_output.relative_to(project_root)),
                "label": label,
                "original_video": video_rel,
                "augmentation_idx": -1,  # -1 indicates original
                "is_original": True,
            })
            
            # Generate augmentations
            augmented_videos = augment_video(video_path, num_augmentations=num_augmentations)
            
            # Get FPS from original video
            try:
                container = av.open(video_path)
                stream = container.streams.video[0]
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                container.close()
            except:
                fps = 30.0
            
            if not augmented_videos:
                logger.warning(f"No augmentations generated for {video_path}")
                continue
            
            # Save augmented videos
            for aug_idx, aug_frames in enumerate(augmented_videos):
                aug_filename = f"{video_id}_aug{aug_idx}.mp4"
                aug_path = output_dir / aug_filename
                
                logger.info(f"Saving augmentation {aug_idx + 1} to {aug_path}")
                success = save_video_frames(aug_frames, str(aug_path), fps=fps)
                
                if success:
                    aug_path_rel = str(aug_path.relative_to(project_root))
                    all_video_metadata.append({
                        "video_path": aug_path_rel,
                        "label": label,
                        "original_video": video_rel,
                        "augmentation_idx": aug_idx,
                        "is_original": False,
                    })
                    logger.info(f"✓ Saved: {aug_path}")
                else:
                    logger.error(f"✗ Failed to save: {aug_path}")
            
            # Clear memory
            del augmented_videos
            aggressive_gc(clear_cuda=False)
            
            log_memory_stats(f"Stage 1: after video {idx + 1}")
            
        except Exception as e:
            logger.error(f"Failed to process video {video_rel}: {e}", exc_info=True)
            continue
    
    # Save metadata
    if all_video_metadata:
        metadata_df = pl.DataFrame(all_video_metadata)
        metadata_path = output_dir / "augmented_metadata.csv"
        metadata_df.write_csv(str(metadata_path))
        logger.info(f"\n✓ Stage 1 complete: Saved metadata to {metadata_path}")
        logger.info(f"✓ Stage 1: Generated {len(all_video_metadata)} total videos ({df.height} original + {len(all_video_metadata) - df.height} augmented)")
        return metadata_df
    else:
        logger.error("Stage 1: No videos processed!")
        return pl.DataFrame()

