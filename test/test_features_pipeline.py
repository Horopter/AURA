"""
Comprehensive unit tests for features/pipeline module.
Tests feature extraction pipeline functions.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from lib.features.pipeline import (
    extract_features_from_video,
    stage2_extract_features,
)


class TestExtractFeaturesFromVideo:
    """Tests for extract_features_from_video function."""
    
    @patch('lib.features.pipeline.HandcraftedFeatureExtractor')
    @patch('lib.features.pipeline._read_video_wrapper')
    def test_extract_features_from_video_basic(self, mock_read, mock_extractor, temp_dir):
        """Test extract_features_from_video extracts features."""
        import numpy as np
        
        # Mock video frames
        mock_frames = np.random.randint(0, 256, (10, 224, 224, 3), dtype=np.uint8)
        mock_read.return_value = mock_frames
        
        # Mock feature extractor
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract_from_frame.return_value = {"feature1": 0.5}
        mock_extractor.return_value = mock_extractor_instance
        
        video_path = str(Path(temp_dir) / "test.mp4")
        Path(video_path).write_text("dummy")
        
        features = extract_features_from_video(video_path, num_frames=10)
        
        assert isinstance(features, dict)
        assert len(features) > 0


class TestStage2ExtractFeatures:
    """Tests for stage2_extract_features function."""
    
    @patch('lib.features.pipeline.extract_features_from_video')
    def test_stage2_extract_features_basic(self, mock_extract, temp_dir):
        """Test stage2_extract_features processes videos."""
        import polars as pl
        
        mock_extract.return_value = {"feature1": 0.5}
        
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0]
        })
        
        # Create output directory
        output_dir = Path(temp_dir) / "features"
        output_dir.mkdir()
        
        result = stage2_extract_features(
            df,
            project_root=temp_dir,
            output_dir=str(output_dir),
            num_frames=10
        )
        
        assert isinstance(result, pl.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
