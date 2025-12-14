"""
Comprehensive unit tests for data/loading module.
Tests data loading functions with dummy data.
"""
import pytest
import polars as pl
from pathlib import Path
from unittest.mock import Mock, patch
from lib.data.loading import (
    filter_existing_videos,
    load_metadata,
)


class TestFilterExistingVideos:
    """Tests for filter_existing_videos function."""
    
    def test_filter_existing_videos_basic(self, temp_dir):
        """Test filter_existing_videos filters non-existing videos."""
        df = pl.DataFrame({
            "video_path": ["video1.mp4", "nonexistent.mp4"],
            "label": [0, 1]
        })
        
        # Create one video file
        video1 = Path(temp_dir) / "video1.mp4"
        video1.write_text("dummy")
        
        filtered = filter_existing_videos(df, project_root=temp_dir, check_frames=False)
        
        assert filtered.height == 1
        assert filtered["video_path"][0] == "video1.mp4"


class TestLoadMetadata:
    """Tests for load_metadata function."""
    
    def test_load_metadata_parquet(self, temp_dir):
        """Test load_metadata loads parquet file."""
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0]
        })
        metadata_path = Path(temp_dir) / "metadata.parquet"
        df.write_parquet(metadata_path)
        
        loaded = load_metadata(str(metadata_path))
        
        assert isinstance(loaded, pl.DataFrame)
        assert loaded.height == 1




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
