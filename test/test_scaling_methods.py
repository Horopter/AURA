"""
Comprehensive unit tests for scaling/methods module.
Tests video scaling methods with dummy frames.
"""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from lib.scaling.methods import (
    letterbox_resize,
    load_hf_autoencoder,
    _autoencoder_scale,
    scale_video_frames,
)


class TestLetterboxResize:
    """Tests for letterbox_resize function."""
    
    def test_letterbox_resize_basic(self):
        """Test letterbox_resize resizes frame."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        resized = letterbox_resize(frame, target_size=256)
        
        assert resized.shape[:2] == (256, 256)
        assert resized.shape[2] == 3


class TestLoadHfAutoencoder:
    """Tests for load_hf_autoencoder function."""
    
    @patch('lib.scaling.methods.AutoencoderKL')
    def test_load_hf_autoencoder_basic(self, mock_autoencoder):
        """Test load_hf_autoencoder loads model."""
        try:
            model = load_hf_autoencoder(model_id="stabilityai/sd-vae-ft-mse")
            # Should not raise
        except ImportError:
            pytest.skip("HuggingFace diffusers not available")


class TestScaleVideoFrames:
    """Tests for scale_video_frames function."""
    
    def test_scale_video_frames_letterbox(self):
        """Test scale_video_frames with letterbox method."""
        frames = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
        
        scaled = scale_video_frames(frames, method="letterbox", target_size=256)
        
        assert len(scaled) == len(frames)
        assert all(f.shape[:2] == (256, 256) for f in scaled)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
