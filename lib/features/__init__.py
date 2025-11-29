"""
Feature extraction module.

Provides:
- Handcrafted feature extraction (noise, DCT, blur, boundary, codec)
- Stage 2: Extract features from original videos
- Stage 4: Extract features from downscaled videos
"""

from .handcrafted_features import (
    extract_noise_residual,
    extract_dct_statistics,
    extract_blur_sharpness,
    extract_boundary_inconsistency,
    extract_codec_cues,
    extract_all_features,
    HandcraftedFeatureExtractor,
)
from .pipeline_stage2_features import stage2_extract_features
from .pipeline_stage4_features_downscaled import stage4_extract_downscaled_features

__all__ = [
    "extract_noise_residual",
    "extract_dct_statistics",
    "extract_blur_sharpness",
    "extract_boundary_inconsistency",
    "extract_codec_cues",
    "extract_all_features",
    "HandcraftedFeatureExtractor",
    "stage2_extract_features",
    "stage4_extract_downscaled_features",
]

