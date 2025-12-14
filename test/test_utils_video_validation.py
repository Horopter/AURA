"""
Comprehensive unit tests for utils/video_validation module.
Tests video validation functions.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from lib.utils.video_validation import (
    validate_video_file,
    validate_videos_batch,
    filter_valid_videos,
)


class TestValidateVideoFile:
    """Tests for validate_video_file function."""
    
    @patch('lib.utils.video_validation.resolve_video_path')
    @patch('lib.utils.video_validation.paths_validate_video_file')
    def test_validate_video_file_not_exists(
        self,
        mock_validate,
        mock_resolve,
        temp_dir
    ):
        """Test validate_video_file with non-existing file."""
        mock_resolve.return_value = str(Path(temp_dir) / "nonexistent.mp4")
        mock_validate.return_value = (False, "File not found")
        
        is_valid, error = validate_video_file("video.mp4", temp_dir, check_frames=False)
        
        assert is_valid is False
        assert error is not None


class TestValidateVideosBatch:
    """Tests for validate_videos_batch function."""
    
    @patch('lib.utils.video_validation.validate_video_file')
    def test_validate_videos_batch_basic(self, mock_validate, temp_dir):
        """Test validate_videos_batch with mocked validation."""
        mock_validate.return_value = (True, None)
        
        video_paths = ["video1.mp4", "video2.mp4"]
        results = validate_videos_batch(video_paths, temp_dir, check_frames=False)
        
        assert isinstance(results, dict)
        assert len(results) == len(video_paths)


class TestFilterValidVideos:
    """Tests for filter_valid_videos function."""
    
    @patch('lib.utils.video_validation.validate_videos_batch')
    def test_filter_valid_videos_basic(self, mock_validate, temp_dir):
        """Test filter_valid_videos with mocked validation."""
        import polars as pl
        
        mock_validate.return_value = {
            "video1.mp4": (True, None),
            "video2.mp4": (False, "Error")
        }
        
        df = pl.DataFrame({
            "video_path": ["video1.mp4", "video2.mp4"],
            "label": [0, 1]
        })
        
        filtered = filter_valid_videos(df, temp_dir, check_frames=False)
        
        assert filtered.height == 1  # Only valid video


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
