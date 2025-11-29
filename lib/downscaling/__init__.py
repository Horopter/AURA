"""
Video downscaling module.

Provides:
- Resolution-based downscaling
- Autoencoder-based downscaling
- Stage 3: Downscale all videos
"""

from .pipeline_stage3_downscale import stage3_downscale_videos

__all__ = ["stage3_downscale_videos"]

