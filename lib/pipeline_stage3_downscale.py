"""
Stage 3: Downscale Videos

Downscale all 11N videos to manageable sizes.
Methods: Resolution reduction or pretrained autoencoder
Input: 11N videos (from Stage 1)
Output: 11N downscaled videos
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
import polars as pl
import numpy as np
import av
import torch
import torch.nn as nn

from .video_paths import resolve_video_path
from .mlops_utils import aggressive_gc, log_memory_stats

logger = logging.getLogger(__name__)


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for video downscaling."""
    
    def __init__(self, input_size: Tuple[int, int], latent_size: Tuple[int, int]):
        super().__init__()
        h_in, w_in = input_size
        h_out, w_out = latent_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # /4
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # /8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(latent_size)  # To exact size
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


def downscale_by_resolution(
    video_path: str,
    target_size: Tuple[int, int] = (224, 224),
    method: str = "letterbox"
) -> Tuple[List[np.ndarray], float]:
    """
    Downscale video by reducing resolution.
    
    Args:
        video_path: Path to input video
        target_size: Target (height, width)
        method: "letterbox" (preserve aspect) or "crop" (center crop)
    
    Returns:
        (downscaled_frames, fps)
    """
    from PIL import Image
    
    # Load original video
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    
    original_frames = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            frame_array = frame.to_ndarray(format='rgb24')
            original_frames.append(frame_array)
    container.close()
    
    if not original_frames:
        return [], fps
    
    # Downscale frames
    downscaled_frames = []
    target_h, target_w = target_size
    
    for frame_array in original_frames:
        pil_image = Image.fromarray(frame_array)
        orig_h, orig_w = frame_array.shape[:2]
        
        if method == "letterbox":
            # Preserve aspect ratio with letterboxing
            scale = min(target_h / orig_h, target_w / orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Create letterboxed image
            result = Image.new('RGB', (target_w, target_h), (0, 0, 0))
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            result.paste(pil_image, (paste_x, paste_y))
            pil_image = result
        
        elif method == "crop":
            # Center crop and resize
            pil_image = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        downscaled_frames.append(np.array(pil_image))
    
    return downscaled_frames, fps


def downscale_by_autoencoder(
    video_path: str,
    autoencoder: Optional[nn.Module] = None,
    target_size: Tuple[int, int] = (224, 224),
    device: str = "cpu"
) -> Tuple[List[np.ndarray], float]:
    """
    Downscale video using autoencoder.
    
    Args:
        video_path: Path to input video
        autoencoder: Pretrained autoencoder model (if None, uses resolution downscaling)
        target_size: Target size for autoencoder
        device: Device to run autoencoder on
    
    Returns:
        (downscaled_frames, fps)
    """
    if autoencoder is None:
        # Fallback to resolution downscaling
        return downscale_by_resolution(video_path, target_size)
    
    # Load original video
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    
    original_frames = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            frame_array = frame.to_ndarray(format='rgb24')
            original_frames.append(frame_array)
    container.close()
    
    if not original_frames:
        return [], fps
    
    # Downscale using autoencoder
    autoencoder.eval()
    downscaled_frames = []
    
    with torch.no_grad():
        for frame_array in original_frames:
            # Convert to tensor and normalize
            frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(device)
            
            # Encode (downscale)
            encoded = autoencoder.encode(frame_tensor)
            
            # Decode back
            decoded = autoencoder(frame_tensor)
            
            # Convert back to numpy
            decoded_np = (decoded.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            downscaled_frames.append(decoded_np)
    
    return downscaled_frames, fps


def save_video_frames(frames: List[np.ndarray], output_path: str, fps: float = 30.0) -> bool:
    """Save frames as a video file."""
    if not frames:
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


def stage3_downscale_videos(
    project_root: str,
    augmented_metadata_path: str,
    output_dir: str = "data/downscaled_videos",
    method: str = "resolution",  # "resolution" or "autoencoder"
    target_size: Tuple[int, int] = (224, 224),
    autoencoder_path: Optional[str] = None,
    device: str = "cpu"
) -> pl.DataFrame:
    """
    Stage 3: Downscale all videos.
    
    Args:
        project_root: Project root directory
        augmented_metadata_path: Path to Stage 1 metadata CSV
        output_dir: Directory to save downscaled videos
        method: "resolution" or "autoencoder"
        target_size: Target (height, width) for downscaling
        autoencoder_path: Path to pretrained autoencoder (if using autoencoder method)
        device: Device to run autoencoder on
    
    Returns:
        DataFrame with metadata for downscaled videos
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load augmented video metadata
    logger.info("Stage 3: Loading augmented video metadata...")
    metadata_df = pl.read_csv(augmented_metadata_path)
    
    logger.info(f"Stage 3: Found {metadata_df.height} videos to downscale")
    logger.info(f"Stage 3: Method: {method}, Target size: {target_size}")
    logger.info(f"Stage 3: Output directory: {output_dir}")
    
    # Load autoencoder if needed
    autoencoder = None
    if method == "autoencoder" and autoencoder_path:
        try:
            checkpoint = torch.load(autoencoder_path, map_location=device)
            # Assume input size from first video
            # In practice, you'd want to save/load model config
            autoencoder = SimpleAutoencoder(
                input_size=(1080, 1920),  # Default, should be configurable
                latent_size=target_size
            )
            autoencoder.load_state_dict(checkpoint['model_state_dict'])
            autoencoder.to(device)
            logger.info(f"Loaded autoencoder from {autoencoder_path}")
        except Exception as e:
            logger.warning(f"Failed to load autoencoder: {e}. Falling back to resolution downscaling.")
            method = "resolution"
    
    downscaled_metadata = []
    
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
            logger.info(f"Stage 3: Processing video {idx + 1}/{metadata_df.height}: {Path(video_path).name}")
            logger.info(f"{'='*80}")
            
            log_memory_stats(f"Stage 3: before video {idx + 1}")
            
            # Downscale video
            if method == "autoencoder" and autoencoder is not None:
                downscaled_frames, fps = downscale_by_autoencoder(
                    video_path, autoencoder, target_size, device
                )
            else:
                downscaled_frames, fps = downscale_by_resolution(
                    video_path, target_size, method="letterbox"
                )
            
            if not downscaled_frames:
                logger.warning(f"No frames after downscaling: {video_path}")
                continue
            
            # Save downscaled video
            video_id = Path(video_path).stem
            video_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in video_id)
            downscaled_filename = f"{video_id}_downscaled.mp4"
            downscaled_path = output_dir / downscaled_filename
            
            logger.info(f"Saving downscaled video to {downscaled_path}")
            success = save_video_frames(downscaled_frames, str(downscaled_path), fps=fps)
            
            if success:
                downscaled_path_rel = str(downscaled_path.relative_to(project_root))
                downscaled_metadata.append({
                    "video_path": downscaled_path_rel,
                    "label": label,
                    "augmentation_idx": aug_idx,
                    "is_original": is_original,
                    "original_video": video_rel,
                    "downscale_method": method,
                    "target_size": f"{target_size[0]}x{target_size[1]}",
                })
                logger.info(f"✓ Saved: {downscaled_path}")
            else:
                logger.error(f"✗ Failed to save: {downscaled_path}")
            
            # Clear memory
            del downscaled_frames
            aggressive_gc(clear_cuda=False)
            
            log_memory_stats(f"Stage 3: after video {idx + 1}")
            
        except Exception as e:
            logger.error(f"Failed to downscale video {video_rel}: {e}", exc_info=True)
            continue
    
    # Save metadata
    if downscaled_metadata:
        metadata_df = pl.DataFrame(downscaled_metadata)
        metadata_path = output_dir / "downscaled_metadata.csv"
        metadata_df.write_csv(str(metadata_path))
        logger.info(f"\n✓ Stage 3 complete: Saved metadata to {metadata_path}")
        logger.info(f"✓ Stage 3: Downscaled {len(downscaled_metadata)} videos")
        return metadata_df
    else:
        logger.error("Stage 3: No videos downscaled!")
        return pl.DataFrame()

