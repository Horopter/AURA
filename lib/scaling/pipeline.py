"""
Video scaling pipeline.

Scales videos to target resolutions using letterbox resizing or autoencoder
methods while preserving aspect ratios. Can both downscale and upscale videos
to ensure max(width, height) = target_size.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import av

from lib.data import load_metadata
from lib.utils.paths import resolve_video_path
from lib.utils.memory import aggressive_gc, log_memory_stats
from lib.scaling.methods import (
    scale_video_frames,
    letterbox_resize,
    load_hf_autoencoder
)
from lib.augmentation.io import load_frames, save_frames, concatenate_videos

logger = logging.getLogger(__name__)


def scale_video(
    video_path: str,
    output_path: str,
    target_size: int = 256,
    max_frames: Optional[int] = 250,
    chunk_size: int = 250,
    method: str = "letterbox",
    autoencoder: Optional[object] = None
) -> bool:
    """
    Scale a single video to target max dimension using chunked processing to avoid OOM.
    
    Can both downscale (if max dimension > target_size) or upscale (if max dimension < target_size)
    to ensure max(width, height) = target_size.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        target_size: Target max dimension (max(width, height) will be target_size)
        max_frames: Maximum frames to process per chunk (default: 250 for memory safety)
        chunk_size: Number of frames to process per chunk (default: 250)
        method: Scaling method ("letterbox" or "autoencoder")
        autoencoder: Optional autoencoder model for autoencoder method
    
    Returns:
        True if successful, False otherwise
    """
    import tempfile
    import shutil
    
    if max_frames is None:
        max_frames = 250
    if chunk_size is None:
        chunk_size = 250
    
    container = None
    temp_dir = None
    try:
        # Use cached metadata to avoid duplicate frame counting
        from lib.utils.video_cache import get_video_metadata
        
        metadata = get_video_metadata(video_path, use_cache=True)
        total_frames = metadata['total_frames']
        fps = metadata['fps']
        
        container = None
        aggressive_gc(clear_cuda=False)
        
        logger.debug(f"Video has {total_frames} frames, processing in chunks of {chunk_size}")
        
        # Create temporary directory for intermediate chunks
        temp_dir = Path(tempfile.mkdtemp(prefix="scale_chunks_"))
        intermediate_files = []
        
        # Process video in chunks
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, total_frames)
            
            if start_frame >= total_frames:
                break
            
            logger.debug(f"Processing chunk {chunk_idx + 1}/{num_chunks} (frames {start_frame}-{end_frame-1})")
            
            # Load chunk
            chunk_frames, chunk_fps = load_frames(video_path, max_frames=chunk_size, start_frame=start_frame)
            
            if not chunk_frames:
                logger.warning(f"No frames loaded for chunk {chunk_idx + 1}, skipping")
                continue
            
            # Scale frames in chunk
            if method == "autoencoder" and autoencoder is not None:
                # Use autoencoder for scaling (preserves aspect ratio)
                scaled_chunk = scale_video_frames(
                    chunk_frames,
                    method="autoencoder",
                    target_size=target_size,
                    autoencoder=autoencoder,
                    preserve_aspect_ratio=True
                )
                # Aggressive GC after autoencoder processing
                aggressive_gc(clear_cuda=False)
            else:
                # Use letterbox resize
                scaled_chunk = []
                for frame_idx, frame in enumerate(chunk_frames):
                    scaled_frame = letterbox_resize(frame, target_size)
                    scaled_chunk.append(scaled_frame)
                    
                    # Aggressive GC every 50 frames
                    if (frame_idx + 1) % 50 == 0:
                        aggressive_gc(clear_cuda=False)
            
            # Save chunk to intermediate file
            intermediate_path = temp_dir / f"chunk_{chunk_idx}.mp4"
            if save_frames(scaled_chunk, str(intermediate_path), fps=chunk_fps):
                intermediate_files.append(str(intermediate_path))
                logger.debug(f"Saved chunk {chunk_idx + 1} with {len(scaled_chunk)} frames")
            else:
                logger.error(f"Failed to save chunk {chunk_idx + 1}")
            
            # Clear chunk memory immediately
            del chunk_frames, scaled_chunk
            aggressive_gc(clear_cuda=False)
        
        # Concatenate all chunks into final video
        if intermediate_files:
            logger.debug(f"Concatenating {len(intermediate_files)} chunks into final video...")
            success = concatenate_videos(intermediate_files, output_path, fps=fps)
            
            # Clean up intermediate files
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.debug(f"Could not delete temp directory: {e}")
            
            aggressive_gc(clear_cuda=False)
            return success
        else:
            logger.warning(f"No chunks processed from {video_path}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to scale video {video_path}: {e}")
        return False
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        aggressive_gc(clear_cuda=False)


def stage3_scale_videos(
    project_root: str,
    augmented_metadata_path: str,
    output_dir: str = "data/scaled_videos",
    target_size: int = 256,
    max_frames: Optional[int] = 250,
    chunk_size: int = 250,
    method: str = "letterbox",
    autoencoder_model: Optional[str] = None
) -> pl.DataFrame:
    """
    Stage 3: Scale all videos to target max dimension.
    
    Can both downscale (if max dimension > target_size) or upscale (if max dimension < target_size)
    to ensure max(width, height) = target_size.
    
    Args:
        project_root: Project root directory
        augmented_metadata_path: Path to augmented metadata CSV
        output_dir: Directory to save scaled videos
        target_size: Target max dimension (max(width, height) = target_size, default: 256)
        max_frames: Maximum frames to process per video
        method: Scaling method ("letterbox" or "autoencoder")
        autoencoder_model: Hugging Face model name for autoencoder (e.g., "stabilityai/sd-vae-ft-mse")
                          If None and method="autoencoder", uses default model
    
    Returns:
        DataFrame with scaled video metadata (includes original_width and original_height)
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load augmented metadata (support both CSV and Arrow/Parquet)
    logger.info("Stage 3: Loading augmented metadata...")
    if not Path(augmented_metadata_path).exists():
        logger.error(f"Augmented metadata not found: {augmented_metadata_path}")
        return pl.DataFrame()
    
    # Try Arrow/Parquet first, fallback to CSV
    metadata_path_obj = Path(augmented_metadata_path)
    if metadata_path_obj.suffix in ['.arrow', '.parquet']:
        df = pl.read_ipc(metadata_path_obj) if metadata_path_obj.suffix == '.arrow' else pl.read_parquet(metadata_path_obj)
    else:
        df = pl.read_csv(augmented_metadata_path)
    logger.info(f"Stage 3: Processing {df.height} videos")
    logger.info(f"Stage 3: Target max dimension: {target_size} pixels")
    
    # Map "resolution" to "letterbox" for backward compatibility
    if method == "resolution":
        method = "letterbox"
    
    logger.info(f"Stage 3: Method: {method}")
    logger.info(f"Stage 3: Chunk size: {chunk_size} frames (optimized for memory)")
    
    # Load autoencoder if needed
    autoencoder = None
    if method == "autoencoder":
        try:
            model_name = autoencoder_model or "stabilityai/sd-vae-ft-mse"
            logger.info(f"Stage 3: Loading Hugging Face autoencoder: {model_name}")
            autoencoder = load_hf_autoencoder(model_name)
            logger.info("✓ Autoencoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load autoencoder: {e}")
            logger.warning("Falling back to letterbox method")
            method = "letterbox"
            autoencoder = None
    
    # Use incremental Arrow/Parquet writing (more efficient than CSV)
    # We'll collect rows and write at the end, or use streaming if needed
    metadata_path = output_dir / "scaled_metadata.arrow"
    metadata_rows = []
    
    total_videos_processed = 0
    
    # Process each video
    for idx in range(df.height):
        row = df.row(idx, named=True)
        video_rel = row["video_path"]
        label = row["label"]
        original_video = row.get("original_video", video_rel)
        aug_idx = row.get("augmentation_idx", -1)
        is_original = row.get("is_original", False)
        
        try:
            video_path = resolve_video_path(video_rel, project_root)
            
            if not Path(video_path).exists():
                logger.warning(f"Video not found: {video_path}")
                continue
            
            if idx % 10 == 0:
                log_memory_stats(f"Stage 3: processing video {idx + 1}/{df.height}")
            
            # Create output path
            video_id = Path(video_path).stem
            if is_original:
                output_filename = f"{video_id}_scaled_original.mp4"
            else:
                output_filename = f"{video_id}_scaled_aug{aug_idx}.mp4"
            
            output_path = output_dir / output_filename
            
            # Get original video dimensions
            original_width = None
            original_height = None
            try:
                container = av.open(video_path)
                stream = container.streams.video[0]
                original_width = stream.width
                original_height = stream.height
                container.close()
            except Exception as e:
                logger.warning(f"Could not get dimensions for {video_path}: {e}")
            
            # Skip if already exists
            if output_path.exists():
                logger.debug(f"Scaled video already exists: {output_path}")
                output_rel = str(output_path.relative_to(project_root))
                metadata_row = {
                    "video_path": output_rel,
                    "label": label,
                    "original_video": original_video,
                    "augmentation_idx": aug_idx,
                    "is_original": is_original
                }
                if original_width is not None and original_height is not None:
                    metadata_row["original_width"] = original_width
                    metadata_row["original_height"] = original_height
                metadata_rows.append(metadata_row)
                total_videos_processed += 1
                continue
            
            # Scale video (downscale or upscale to target_size)
            logger.info(f"Scaling {Path(video_path).name} to {output_path.name}")
            success = scale_video(
                video_path,
                str(output_path),
                target_size=target_size,
                max_frames=max_frames,
                chunk_size=chunk_size,
                method=method,
                autoencoder=autoencoder
            )
            
            if success:
                output_rel = str(output_path.relative_to(project_root))
                metadata_row = {
                    "video_path": output_rel,
                    "label": label,
                    "original_video": original_video,
                    "augmentation_idx": aug_idx,
                    "is_original": is_original
                }
                if original_width is not None and original_height is not None:
                    metadata_row["original_width"] = original_width
                    metadata_row["original_height"] = original_height
                metadata_rows.append(metadata_row)
                total_videos_processed += 1
                logger.info(f"✓ Scaled: {output_path.name}")
            else:
                logger.error(f"✗ Failed to scale: {video_path}")
            
            # Aggressive GC after each video
            aggressive_gc(clear_cuda=False)
            
        except Exception as e:
            logger.error(f"Error processing {video_rel}: {e}", exc_info=True)
            aggressive_gc(clear_cuda=False)
            continue
    
    # Save final metadata as Arrow IPC
    if metadata_rows and total_videos_processed > 0:
        try:
            metadata_df = pl.DataFrame(metadata_rows)
            # Try Arrow IPC first, fallback to Parquet
            try:
                metadata_df.write_ipc(str(metadata_path))
                logger.debug(f"Saved metadata as Arrow IPC: {metadata_path}")
            except Exception as e:
                logger.warning(f"Arrow IPC write failed, using Parquet: {e}")
                metadata_path = output_dir / "scaled_metadata.parquet"
                metadata_df.write_parquet(str(metadata_path))
            
            logger.info(f"\n✓ Stage 3 complete: Saved metadata to {metadata_path}")
            logger.info(f"✓ Stage 3: Scaled {total_videos_processed} videos")
            return metadata_df
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return pl.DataFrame()
    else:
        logger.error("Stage 3: No videos processed!")
        return pl.DataFrame()

