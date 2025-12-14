"""
Comprehensive unit tests for models/webdataset_loader module.
Tests webdataset loader functions.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from lib.models.webdataset_loader import (
    create_webdataset_loader,
    create_webdataset_from_videos,
)


class TestCreateWebdatasetLoader:
    """Tests for create_webdataset_loader function."""
    
    @patch('lib.models.webdataset_loader.webdataset')
    def test_create_webdataset_loader_basic(self, mock_webdataset, temp_dir):
        """Test create_webdataset_loader creates loader."""
        try:
            loader = create_webdataset_loader(
                shard_pattern=str(Path(temp_dir) / "shards" / "*.tar"),
                batch_size=4
            )
            # Should not raise
        except ImportError:
            pytest.skip("webdataset not available")


class TestCreateWebdatasetFromVideos:
    """Tests for create_webdataset_from_videos function."""
    
    @patch('lib.models.webdataset_loader.webdataset')
    def test_create_webdataset_from_videos_basic(self, mock_webdataset, temp_dir):
        """Test create_webdataset_from_videos creates dataset."""
        import polars as pl
        
        df = pl.DataFrame({
            "video_path": ["video1.mp4"],
            "label": [0]
        })
        
        output_dir = Path(temp_dir) / "webdataset"
        output_dir.mkdir()
        
        try:
            create_webdataset_from_videos(
                df,
                project_root=temp_dir,
                output_dir=str(output_dir),
                shard_size=100
            )
            # Should not raise
        except ImportError:
            pytest.skip("webdataset not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
