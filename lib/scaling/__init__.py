"""
Video scaling module.

Provides:
- Resolution-based scaling (letterbox resize)
- Autoencoder-based scaling using pretrained Hugging Face models (optional)
- Stage 3: Scale all videos to target max dimension (can downscale or upscale)
"""

from .methods import (
    letterbox_resize,
    scale_video_frames,
    load_hf_autoencoder,
)
from .pipeline import (
    scale_video,
    stage3_scale_videos,
)


__all__ = [
    # Methods
    "letterbox_resize",
    "scale_video_frames",
    "load_hf_autoencoder",
    # Stage 3
    "scale_video",
    "stage3_scale_videos",
]

