"""
Comprehensive unit tests for data/metadata module.
Tests metadata parsing functions.
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from lib.data.metadata import (
    _find_column,
    _extract_video_id_from_url,
    _normalize_label,
    load_main_metadata,
    load_duplicates,
    parse_metadata,
)


class TestFindColumn:
    """Tests for _find_column function."""
    
    def test_find_column_exists(self):
        """Test _find_column finds existing column."""
        columns = ["video_path", "label", "url"]
        candidates = ["url", "video_url"]
        
        result = _find_column(columns, candidates)
        assert result == "url"
    
    def test_find_column_not_exists(self):
        """Test _find_column raises when column not found."""
        columns = ["video_path", "label"]
        candidates = ["url", "video_url"]
        
        with pytest.raises(ValueError):
            _find_column(columns, candidates)


class TestExtractVideoIdFromUrl:
    """Tests for _extract_video_id_from_url function."""
    
    def test_extract_youtube_url(self):
        """Test _extract_video_id_from_url with YouTube URL."""
        url = "https://www.youtube.com/watch?v=abc123"
        result = _extract_video_id_from_url(url)
        
        assert result is not None
        video_id, platform = result
        assert video_id == "abc123"
        assert platform == "youtube"
    
    def test_extract_twitter_url(self):
        """Test _extract_video_id_from_url with Twitter URL."""
        url = "https://twitter.com/user/status/123456789"
        result = _extract_video_id_from_url(url)
        
        assert result is not None
        video_id, platform = result
        assert video_id == "123456789"
        assert platform == "twitter"
    
    def test_extract_invalid_url(self):
        """Test _extract_video_id_from_url with invalid URL."""
        url = "https://example.com/video"
        result = _extract_video_id_from_url(url)
        
        assert result is None


class TestNormalizeLabel:
    """Tests for _normalize_label function."""
    
    def test_normalize_label_real(self):
        """Test _normalize_label with 'real' label."""
        assert _normalize_label("real") == 0
        assert _normalize_label("Real") == 0
        assert _normalize_label("authentic") == 0
    
    def test_normalize_label_fake(self):
        """Test _normalize_label with 'fake' label."""
        assert _normalize_label("fake") == 1
        assert _normalize_label("Fake") == 1
        assert _normalize_label("manipulated") == 1
    
    def test_normalize_label_numeric(self):
        """Test _normalize_label with numeric label."""
        assert _normalize_label(0) == 0
        assert _normalize_label(1) == 1
        assert _normalize_label(0.0) == 0


class TestLoadMainMetadata:
    """Tests for load_main_metadata function."""
    
    @patch('lib.data.metadata.pd.read_csv')
    def test_load_main_metadata_basic(self, mock_read_csv, temp_dir):
        """Test load_main_metadata loads CSV."""
        from lib.data.config import FVCConfig
        
        mock_df = pd.DataFrame({
            "video_id": ["v1", "v2"],
            "label": [0, 1]
        })
        mock_read_csv.return_value = mock_df
        
        cfg = FVCConfig(root_dir=temp_dir)
        # Create metadata file
        metadata_dir = Path(cfg.metadata_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "FVC.csv").write_text("video_id,label\nv1,0\nv2,1")
        
        df = load_main_metadata(cfg)
        
        assert isinstance(df, pd.DataFrame)


class TestParseMetadata:
    """Tests for parse_metadata function."""
    
    @patch('lib.data.metadata.load_main_metadata')
    @patch('lib.data.metadata.load_duplicates')
    def test_parse_metadata_basic(self, mock_load_dup, mock_load_main, temp_dir):
        """Test parse_metadata returns main and dup DataFrames."""
        from lib.data.config import FVCConfig
        
        mock_main = pd.DataFrame({"video_id": ["v1"], "label": [0]})
        mock_dup = pd.DataFrame({"video_id": ["v2"], "label": [1]})
        
        mock_load_main.return_value = mock_main
        mock_load_dup.return_value = mock_dup
        
        cfg = FVCConfig(root_dir=temp_dir)
        main_df, dup_df = parse_metadata(cfg)
        
        assert isinstance(main_df, pd.DataFrame)
        assert isinstance(dup_df, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
