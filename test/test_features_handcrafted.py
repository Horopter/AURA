"""
Comprehensive unit tests for features/handcrafted module.
Tests handcrafted feature extraction with dummy frames.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from lib.features.handcrafted import (
    _check_ffprobe_available,
    extract_noise_residual,
    extract_dct_statistics,
    extract_blur_sharpness,
    extract_boundary_inconsistency,
    extract_codec_cues,
    extract_all_features,
    HandcraftedFeatureExtractor,
)


class TestCheckFfprobeAvailable:
    """Tests for _check_ffprobe_available function."""
    
    @patch('lib.features.handcrafted.subprocess.run')
    def test_check_ffprobe_available_true(self, mock_subprocess):
        """Test _check_ffprobe_available returns True when ffprobe available."""
        mock_subprocess.return_value.returncode = 0
        assert _check_ffprobe_available() is True
    
    @patch('lib.features.handcrafted.subprocess.run')
    def test_check_ffprobe_available_false(self, mock_subprocess):
        """Test _check_ffprobe_available returns False when ffprobe not available."""
        mock_subprocess.side_effect = FileNotFoundError()
        assert _check_ffprobe_available() is False


class TestExtractNoiseResidual:
    """Tests for extract_noise_residual function."""
    
    def test_extract_noise_residual_basic(self):
        """Test extract_noise_residual with dummy frame."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        features = extract_noise_residual(frame)
        
        assert isinstance(features, dict)
        assert len(features) > 0


class TestExtractDctStatistics:
    """Tests for extract_dct_statistics function."""
    
    def test_extract_dct_statistics_basic(self):
        """Test extract_dct_statistics with dummy frame."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        features = extract_dct_statistics(frame, block_size=8)
        
        assert isinstance(features, dict)
        assert len(features) > 0


class TestExtractBlurSharpness:
    """Tests for extract_blur_sharpness function."""
    
    def test_extract_blur_sharpness_basic(self):
        """Test extract_blur_sharpness with dummy frame."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        features = extract_blur_sharpness(frame)
        
        assert isinstance(features, dict)
        assert "laplacian_var" in features or "gradient_mean" in features


class TestExtractBoundaryInconsistency:
    """Tests for extract_boundary_inconsistency function."""
    
    def test_extract_boundary_inconsistency_basic(self):
        """Test extract_boundary_inconsistency with dummy frame."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        inconsistency = extract_boundary_inconsistency(frame, block_size=8)
        
        assert isinstance(inconsistency, float)
        assert inconsistency >= 0


class TestExtractCodecCues:
    """Tests for extract_codec_cues function."""
    
    @patch('lib.features.handcrafted._check_ffprobe_available')
    @patch('lib.features.handcrafted.subprocess.run')
    def test_extract_codec_cues_basic(self, mock_subprocess, mock_check, temp_dir):
        """Test extract_codec_cues with dummy video."""
        mock_check.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = b'{"format": {"bit_rate": "1000000"}}'
        mock_subprocess.return_value = mock_result
        
        video_path = str(Path(temp_dir) / "test.mp4")
        Path(video_path).write_text("dummy")
        
        features = extract_codec_cues(video_path)
        
        assert isinstance(features, dict)


class TestExtractAllFeatures:
    """Tests for extract_all_features function."""
    
    def test_extract_all_features_basic(self):
        """Test extract_all_features with dummy frame."""
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        features = extract_all_features(frame)
        
        assert isinstance(features, dict)
        assert len(features) > 0


class TestHandcraftedFeatureExtractor:
    """Tests for HandcraftedFeatureExtractor class."""
    
    def test_initialization(self):
        """Test HandcraftedFeatureExtractor initialization."""
        extractor = HandcraftedFeatureExtractor()
        assert extractor is not None
    
    def test_extract_from_frame(self):
        """Test extract method (extract_from_frame is now extract)."""
        extractor = HandcraftedFeatureExtractor()
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        features = extractor.extract(frame)
        
        assert isinstance(features, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
