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
    @patch('lib.features.pipeline.get_video_metadata')
    @patch('lib.features.pipeline.av.open')
    def test_extract_features_from_video_basic(self, mock_av_open, mock_metadata, mock_extractor, temp_dir):
        """Test extract_features_from_video extracts features."""
        import numpy as np
        import av
        
        # Mock video metadata
        mock_metadata.return_value = {'total_frames': 10}
        
        # Mock video container
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.frames = 10
        mock_container.streams.video = [mock_stream]
        mock_frame = Mock()
        mock_frame.to_ndarray.return_value = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        mock_packet = Mock()
        mock_packet.decode.return_value = [mock_frame]
        mock_container.demux.return_value = [mock_packet]
        mock_av_open.return_value = mock_container
        
        # Mock feature extractor
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract.return_value = {"feature1": 0.5, "feature2": 0.3}
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
        
        # Create metadata file
        metadata_path = Path(temp_dir) / "metadata.csv"
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0]
        })
        df.write_csv(metadata_path)
        
        # Create output directory
        output_dir = Path(temp_dir) / "features"
        output_dir.mkdir()
        
        result = stage2_extract_features(
            project_root=str(temp_dir),
            augmented_metadata_path=str(metadata_path),
            output_dir=str(output_dir),
            num_frames=10
        )
        
        assert isinstance(result, pl.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
