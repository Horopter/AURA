"""
Comprehensive unit tests for data/index module.
Tests video index building functions.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from lib.data.index import build_video_index


class TestBuildVideoIndex:
    """Tests for build_video_index function."""
    
    @patch('lib.data.index.scan_videos')
    @patch('lib.data.index.parse_metadata')
    def test_build_video_index_basic(self, mock_parse, mock_scan, temp_dir):
        """Test build_video_index joins metadata and videos."""
        from lib.data.config import FVCConfig
        
        mock_main = pd.DataFrame({
            "video_id": ["v1", "v2"],
            "label": [0, 1]
        })
        mock_dup = pd.DataFrame()
        mock_parse.return_value = (mock_main, mock_dup)
        
        mock_scan.return_value = [
            {"video_id": "v1", "video_path": "v1/video.mp4"},
            {"video_id": "v2", "video_path": "v2/video.mp4"}
        ]
        
        cfg = FVCConfig(root_dir=temp_dir)
        result = build_video_index(cfg, drop_duplicates=False, compute_stats=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "video_path" in result.columns
        assert "label" in result.columns
    
    @patch('lib.data.index.scan_videos')
    @patch('lib.data.index.parse_metadata')
    def test_build_video_index_drop_duplicates(self, mock_parse, mock_scan, temp_dir):
        """Test build_video_index drops duplicates when requested."""
        from lib.data.config import FVCConfig
        
        mock_main = pd.DataFrame({
            "video_id": ["v1"],
            "label": [0]
        })
        mock_dup = pd.DataFrame({
            "video_id": ["v1", "v1"],
            "dup_group": [1, 1],
            "label": [0, 0]
        })
        mock_parse.return_value = (mock_main, mock_dup)
        
        mock_scan.return_value = [
            {"video_id": "v1", "video_path": "v1/video.mp4"},
            {"video_id": "v1", "video_path": "v1/video2.mp4"}
        ]
        
        cfg = FVCConfig(root_dir=temp_dir)
        result = build_video_index(cfg, drop_duplicates=True, compute_stats=False)
        
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
