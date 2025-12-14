"""
Comprehensive unit tests for utils/video_cache module.
Tests video metadata caching functions.
"""
import pytest
from pathlib import Path
from lib.utils.video_cache import (
    get_video_metadata_hash,
    get_video_metadata,
    clear_cache,
)


class TestGetVideoMetadataHash:
    """Tests for get_video_metadata_hash function."""
    
    def test_get_video_metadata_hash_exists(self, temp_dir):
        """Test get_video_metadata_hash with existing file."""
        test_file = Path(temp_dir) / "test.mp4"
        test_file.write_text("dummy")
        
        hash_val = get_video_metadata_hash(str(test_file))
        assert isinstance(hash_val, str)
        assert len(hash_val) > 0
    
    def test_get_video_metadata_hash_not_exists(self, temp_dir):
        """Test get_video_metadata_hash with non-existing file."""
        hash_val = get_video_metadata_hash(str(Path(temp_dir) / "nonexistent.mp4"))
        assert hash_val == ""


class TestGetVideoMetadata:
    """Tests for get_video_metadata function."""
    
    def test_get_video_metadata_not_exists(self, temp_dir):
        """Test get_video_metadata with non-existing file."""
        metadata = get_video_metadata(str(Path(temp_dir) / "nonexistent.mp4"))
        
        assert isinstance(metadata, dict)
        assert metadata['total_frames'] == 0


class TestClearCache:
    """Tests for clear_cache function."""
    
    def test_clear_cache_basic(self):
        """Test clear_cache doesn't raise errors."""
        clear_cache()
        # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
