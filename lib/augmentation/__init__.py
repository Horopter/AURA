"""
Video augmentation module.

Provides:
- Spatial augmentations (rotation, affine, noise, blur, etc.)
- Temporal augmentations (frame dropping, duplication, reversal)
- Pre-generation pipeline for augmented clips
- Stage 1 augmentation pipeline
"""

from .video_augmentations import (
    build_comprehensive_frame_transforms,
    apply_temporal_augmentations,
)
from .video_augmentation_pipeline import (
    pregenerate_augmented_dataset,
    load_precomputed_clip,
)
from .pipeline_stage1_augmentation import stage1_augment_videos

__all__ = [
    "build_comprehensive_frame_transforms",
    "apply_temporal_augmentations",
    "pregenerate_augmented_dataset",
    "load_precomputed_clip",
    "stage1_augment_videos",
]

