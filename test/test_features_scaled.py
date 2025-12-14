"""
Comprehensive unit tests for features/scaled module.
Tests scaled feature extraction functions.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from lib.features.scaled import (
    extract_scaled_features,
    stage4_extract_scaled_features,
)


class TestExtractScaledFeatures:
    """Tests for extract_scaled_features function."""
    
    @patch('lib.features.scaled.av.open')
    def test_extract_scaled_features_basic(self, mock_av_open, temp_dir):
        """Test extract_scaled_features extracts features from scaled video."""
        import numpy as np
        import av
        
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
        
        video_path = str(Path(temp_dir) / "scaled_video.mp4")
        Path(video_path).write_text("dummy")
        
        features = extract_scaled_features(video_path, num_frames=10)
        
        assert isinstance(features, dict)
        assert len(features) > 0


class TestStage4ExtractScaledFeatures:
    """Tests for stage4_extract_scaled_features function."""
    
    @patch('lib.features.scaled.extract_scaled_features')
    def test_stage4_extract_scaled_features_basic(self, mock_extract, temp_dir):
        """Test stage4_extract_scaled_features processes scaled videos."""
        import polars as pl
        
        mock_extract.return_value = {"feature1": 0.5}
        
        # Create metadata file
        metadata_path = Path(temp_dir) / "scaled_metadata.csv"
        df = pl.DataFrame({
            "video_path": ["scaled_video1.mp4"],
            "label": [0]
        })
        df.write_csv(metadata_path)
        
        # Create output directory
        output_dir = Path(temp_dir) / "scaled_features"
        output_dir.mkdir()
        
        result = stage4_extract_scaled_features(
            project_root=str(temp_dir),
            scaled_metadata_path=str(metadata_path),
            output_dir=str(output_dir),
            num_frames=10
        )
        
        assert isinstance(result, pl.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
